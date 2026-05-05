#!/usr/bin/env python3
"""Compare one pipeline result against a time-aligned ground-truth transcript.

The script accepts the formats used in this repo:
  - UI result.json with ``segments`` or ``transcription_json``
  - Audiofy/API style JSON with ``speaker_json``
  - A raw list of segment dicts
  - CSV with start,end,speaker,phrase columns

It reports time-weighted AGENT/CUSTOMER accuracy, role F1, WER, and the
largest mismatch intervals. Use it before and after every speaker-ID change.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROLE_AGENT = "AGENT"
ROLE_CUSTOMER = "CUSTOMER"
ROLE_UNKNOWN = "UNKNOWN"
ROLE_NONE = "NONE"


def ts_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    except ValueError:
        return None
    return None


def norm_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def wer(ref_text: str, hyp_text: str) -> float:
    ref = norm_words(ref_text)
    hyp = norm_words(hyp_text)
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            if ref_word == hyp_word:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(ref)


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_plain_transcript(path: Path) -> List[Dict[str, Any]]:
    """Parse simple transcript lines.

    Supported examples:
      [00:00:01.000 - 00:00:02.400] Agent: hello
      00:00:01.000 --> 00:00:02.400 Customer: yes
      Agent: hello
    """
    rows: List[Dict[str, Any]] = []
    timed_pattern = re.compile(
        r"^\s*\[?(?P<start>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)"
        r"\s*(?:-->|-|to)\s*"
        r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\]?\s*"
        r"(?P<speaker>[^:]+):\s*(?P<text>.*)$",
        re.IGNORECASE,
    )
    role_pattern = re.compile(r"^\s*(?P<speaker>Agent|Customer)\s*:\s*(?P<text>.*)$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        match = timed_pattern.match(line)
        if match:
            rows.append(
                {
                    "start": match.group("start"),
                    "end": match.group("end"),
                    "speaker": match.group("speaker").strip(),
                    "text": match.group("text").strip(),
                }
            )
            continue
        match = role_pattern.match(line)
        if match:
            rows.append(
                {
                    "speaker": match.group("speaker").strip(),
                    "text": match.group("text").strip(),
                }
            )
    return rows


def extract_segment_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []

    for key in ("speaker_json", "segments", "transcription_json"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    data_value = data.get("data")
    if isinstance(data_value, dict):
        extracted = extract_segment_list(data_value)
        if extracted:
            return extracted
    if isinstance(data_value, list):
        if len(data_value) == 1 and isinstance(data_value[0], dict):
            extracted = extract_segment_list(data_value[0])
            if extracted:
                return extracted
        return [x for x in data_value if isinstance(x, dict)]

    return []


def load_segments(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix in {".txt", ".md"}:
        plain = load_plain_transcript(path)
        if plain:
            return plain

    data = load_json_or_jsonl(path)
    return extract_segment_list(data)


def text_of(seg: Dict[str, Any]) -> str:
    return str(
        seg.get("text")
        or seg.get("phrase")
        or seg.get("transcript")
        or seg.get("utterance")
        or ""
    ).strip()


def role_from_segment(
    seg: Dict[str, Any],
    *,
    is_truth: bool,
    agent_name: str = "",
) -> str:
    labels = [
        seg.get("identified_speaker"),
        seg.get("display_speaker"),
        seg.get("speaker"),
        seg.get("agent_name"),
        seg.get("role"),
    ]
    joined = " ".join(str(x or "") for x in labels).strip()
    low = joined.lower()
    agent_low = agent_name.strip().lower()

    explicit = str(seg.get("identified_speaker") or seg.get("role") or "").upper()
    if explicit == ROLE_AGENT:
        return ROLE_AGENT
    if explicit == ROLE_CUSTOMER:
        return ROLE_CUSTOMER
    if "customer" in low or "client" in low or "caller" in low:
        return ROLE_CUSTOMER
    if agent_low and agent_low in low:
        return ROLE_AGENT
    if "unknown" in low or not low:
        return ROLE_UNKNOWN
    if is_truth:
        # In API ground truth, any named non-customer speaker is the agent.
        return ROLE_AGENT
    return ROLE_UNKNOWN


def normalise_segments(
    rows: Iterable[Dict[str, Any]],
    *,
    is_truth: bool,
    agent_name: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        start = ts_to_seconds(row.get("start") or row.get("start_time"))
        end = ts_to_seconds(row.get("end") or row.get("end_time"))
        if start is None or end is None or end <= start:
            continue
        role = role_from_segment(row, is_truth=is_truth, agent_name=agent_name)
        out.append(
            {
                "idx": idx,
                "start": float(start),
                "end": float(end),
                "role": role,
                "text": text_of(row),
                "raw": row,
            }
        )
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def normalise_turns(
    rows: Iterable[Dict[str, Any]],
    *,
    is_truth: bool,
    agent_name: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        text = text_of(row)
        if not text:
            continue
        role = role_from_segment(row, is_truth=is_truth, agent_name=agent_name)
        if role not in (ROLE_AGENT, ROLE_CUSTOMER):
            role = ROLE_UNKNOWN
        out.append({"idx": idx, "role": role, "text": text, "raw": row})
    return out


def token_role_stream(turns: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    stream: List[Tuple[str, str]] = []
    for turn in turns:
        for word in norm_words(turn["text"]):
            stream.append((word, turn["role"]))
    return stream


def align_token_roles(
    ref: List[Tuple[str, str]],
    hyp: List[Tuple[str, str]],
) -> List[Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]]:
    """Levenshtein alignment over words, preserving roles for scoring."""
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: List[List[str]] = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        ref_word = ref[i - 1][0]
        for j in range(1, m + 1):
            hyp_word = hyp[j - 1][0]
            sub_cost = 0 if ref_word == hyp_word else 1
            choices = (
                (dp[i - 1][j - 1] + sub_cost, "M"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            )
            cost, op = min(choices, key=lambda x: x[0])
            dp[i][j] = cost
            back[i][j] = op

    aligned: List[Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            aligned.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            aligned.append((ref[i - 1], None))
            i -= 1
        else:
            aligned.append((None, hyp[j - 1]))
            j -= 1
    aligned.reverse()
    return aligned


def compare_text_roles(
    truth_turns: List[Dict[str, Any]],
    pred_turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ref = token_role_stream(truth_turns)
    hyp = token_role_stream(pred_turns)
    if not ref:
        raise ValueError("Ground-truth transcript has no words to score.")
    if not hyp:
        raise ValueError("Prediction transcript has no words to score.")

    confusion: Dict[str, Dict[str, int]] = {
        ROLE_AGENT: {ROLE_AGENT: 0, ROLE_CUSTOMER: 0, ROLE_UNKNOWN: 0, ROLE_NONE: 0},
        ROLE_CUSTOMER: {ROLE_AGENT: 0, ROLE_CUSTOMER: 0, ROLE_UNKNOWN: 0, ROLE_NONE: 0},
    }
    insertions = {ROLE_AGENT: 0, ROLE_CUSTOMER: 0, ROLE_UNKNOWN: 0}
    examples: List[Dict[str, Any]] = []
    correct = total = 0
    correct_with_hyp = total_with_hyp = 0

    for pos, (ref_item, hyp_item) in enumerate(align_token_roles(ref, hyp)):
        if ref_item is None:
            hyp_role = hyp_item[1] if hyp_item else ROLE_UNKNOWN
            if hyp_role not in insertions:
                hyp_role = ROLE_UNKNOWN
            insertions[hyp_role] += 1
            continue

        ref_word, ref_role = ref_item
        hyp_word = hyp_item[0] if hyp_item else ""
        hyp_role = hyp_item[1] if hyp_item else ROLE_NONE
        if hyp_role not in (ROLE_AGENT, ROLE_CUSTOMER, ROLE_UNKNOWN, ROLE_NONE):
            hyp_role = ROLE_UNKNOWN

        if ref_role in (ROLE_AGENT, ROLE_CUSTOMER):
            confusion[ref_role][hyp_role] += 1
            total += 1
            if ref_role == hyp_role:
                correct += 1
            if hyp_item is not None:
                total_with_hyp += 1
                if ref_role == hyp_role:
                    correct_with_hyp += 1
            elif len(examples) < 40:
                examples.append(
                    {
                        "aligned_position": pos,
                        "truth_word": ref_word,
                        "pred_word": hyp_word,
                        "truth_role": ref_role,
                        "pred_role": hyp_role,
                    }
                )

    tp_agent = confusion[ROLE_AGENT][ROLE_AGENT]
    fp_agent = confusion[ROLE_CUSTOMER][ROLE_AGENT] + insertions[ROLE_AGENT]
    fn_agent = (
        confusion[ROLE_AGENT][ROLE_CUSTOMER]
        + confusion[ROLE_AGENT][ROLE_UNKNOWN]
        + confusion[ROLE_AGENT][ROLE_NONE]
    )
    tp_customer = confusion[ROLE_CUSTOMER][ROLE_CUSTOMER]
    fp_customer = confusion[ROLE_AGENT][ROLE_CUSTOMER] + insertions[ROLE_CUSTOMER]
    fn_customer = (
        confusion[ROLE_CUSTOMER][ROLE_AGENT]
        + confusion[ROLE_CUSTOMER][ROLE_UNKNOWN]
        + confusion[ROLE_CUSTOMER][ROLE_NONE]
    )

    ref_all = " ".join(turn["text"] for turn in truth_turns)
    hyp_all = " ".join(turn["text"] for turn in pred_turns)
    ref_agent = " ".join(turn["text"] for turn in truth_turns if turn["role"] == ROLE_AGENT)
    hyp_agent = " ".join(turn["text"] for turn in pred_turns if turn["role"] == ROLE_AGENT)
    ref_customer = " ".join(turn["text"] for turn in truth_turns if turn["role"] == ROLE_CUSTOMER)
    hyp_customer = " ".join(turn["text"] for turn in pred_turns if turn["role"] == ROLE_CUSTOMER)

    return {
        "comparison_mode": "text_aligned_roles",
        "truth_words": len(ref),
        "pred_words": len(hyp),
        "role_accuracy": round(correct / max(total, 1), 4),
        "role_accuracy_on_emitted_words": round(
            correct_with_hyp / max(total_with_hyp, 1),
            4,
        ),
        "confusion_words": confusion,
        "inserted_prediction_words": insertions,
        "agent": prf(tp_agent, fp_agent, fn_agent),
        "customer": prf(tp_customer, fp_customer, fn_customer),
        "wer": {
            "all": round(wer(ref_all, hyp_all), 4),
            "agent": round(wer(ref_agent, hyp_agent), 4),
            "customer": round(wer(ref_customer, hyp_customer), 4),
            "word_accuracy_all": round(1.0 - wer(ref_all, hyp_all), 4),
        },
        "mismatch_examples": examples,
    }


def segment_at(segments: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    for seg in segments:
        if seg["start"] <= t < seg["end"]:
            return seg
    return None


def merge_intervals(samples: List[Tuple[float, str, str]], step: float) -> List[Dict[str, Any]]:
    intervals: List[Dict[str, Any]] = []
    for t, truth, pred in samples:
        if (
            intervals
            and intervals[-1]["truth_role"] == truth
            and intervals[-1]["pred_role"] == pred
            and abs(intervals[-1]["end"] - t) <= step * 1.5
        ):
            intervals[-1]["end"] = round(t + step, 3)
            intervals[-1]["seconds"] = round(intervals[-1]["end"] - intervals[-1]["start"], 3)
        else:
            intervals.append(
                {
                    "start": round(t, 3),
                    "end": round(t + step, 3),
                    "seconds": round(step, 3),
                    "truth_role": truth,
                    "pred_role": pred,
                }
            )
    return intervals


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compare(
    truth_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    *,
    step: float,
) -> Dict[str, Any]:
    if not truth_segments:
        raise ValueError("No usable ground-truth segments with start/end timestamps.")
    if not pred_segments:
        raise ValueError("No usable prediction segments with start/end timestamps.")

    start = min(seg["start"] for seg in truth_segments)
    end = max(seg["end"] for seg in truth_segments)
    t = start

    confusion: Dict[str, Dict[str, int]] = {
        ROLE_AGENT: {ROLE_AGENT: 0, ROLE_CUSTOMER: 0, ROLE_UNKNOWN: 0, ROLE_NONE: 0},
        ROLE_CUSTOMER: {ROLE_AGENT: 0, ROLE_CUSTOMER: 0, ROLE_UNKNOWN: 0, ROLE_NONE: 0},
    }
    wrong_samples: List[Tuple[float, str, str]] = []
    total = correct = 0

    while t < end:
        gt = segment_at(truth_segments, t)
        if not gt or gt["role"] not in (ROLE_AGENT, ROLE_CUSTOMER):
            t += step
            continue

        pred = segment_at(pred_segments, t)
        pred_role = pred["role"] if pred else ROLE_NONE
        if pred_role not in (ROLE_AGENT, ROLE_CUSTOMER, ROLE_UNKNOWN):
            pred_role = ROLE_UNKNOWN

        truth_role = gt["role"]
        confusion[truth_role][pred_role] += 1
        total += 1
        if pred_role == truth_role:
            correct += 1
        else:
            wrong_samples.append((t, truth_role, pred_role))
        t += step

    tp_agent = confusion[ROLE_AGENT][ROLE_AGENT]
    fp_agent = confusion[ROLE_CUSTOMER][ROLE_AGENT]
    fn_agent = (
        confusion[ROLE_AGENT][ROLE_CUSTOMER]
        + confusion[ROLE_AGENT][ROLE_UNKNOWN]
        + confusion[ROLE_AGENT][ROLE_NONE]
    )
    tp_customer = confusion[ROLE_CUSTOMER][ROLE_CUSTOMER]
    fp_customer = confusion[ROLE_AGENT][ROLE_CUSTOMER]
    fn_customer = (
        confusion[ROLE_CUSTOMER][ROLE_AGENT]
        + confusion[ROLE_CUSTOMER][ROLE_UNKNOWN]
        + confusion[ROLE_CUSTOMER][ROLE_NONE]
    )

    ref_all = " ".join(seg["text"] for seg in truth_segments)
    hyp_all = " ".join(seg["text"] for seg in pred_segments)
    ref_agent = " ".join(seg["text"] for seg in truth_segments if seg["role"] == ROLE_AGENT)
    hyp_agent = " ".join(seg["text"] for seg in pred_segments if seg["role"] == ROLE_AGENT)
    ref_customer = " ".join(seg["text"] for seg in truth_segments if seg["role"] == ROLE_CUSTOMER)
    hyp_customer = " ".join(seg["text"] for seg in pred_segments if seg["role"] == ROLE_CUSTOMER)

    return {
        "sample_step_seconds": step,
        "scored_seconds": round(total * step, 3),
        "role_accuracy": round(correct / max(total, 1), 4),
        "confusion_samples": confusion,
        "agent": prf(tp_agent, fp_agent, fn_agent),
        "customer": prf(tp_customer, fp_customer, fn_customer),
        "wer": {
            "all": round(wer(ref_all, hyp_all), 4),
            "agent": round(wer(ref_agent, hyp_agent), 4),
            "customer": round(wer(ref_customer, hyp_customer), 4),
            "word_accuracy_all": round(1.0 - wer(ref_all, hyp_all), 4),
        },
        "mismatch_intervals": merge_intervals(wrong_samples, step),
    }


def print_report(report: Dict[str, Any], max_errors: int) -> None:
    print("\nGROUND TRUTH COMPARISON")
    if report.get("comparison_mode") == "text_aligned_roles":
        print("  Mode:          text-aligned roles (truth has no timestamps)")
        print(f"  Truth words:   {report['truth_words']}")
        print(f"  Pred words:    {report['pred_words']}")
        print(
            "  Role on ASR words: "
            f"{report['role_accuracy_on_emitted_words'] * 100:.2f}%"
        )
    else:
        print(f"  Scored seconds: {report['scored_seconds']}")
    print(f"  Role accuracy:  {report['role_accuracy'] * 100:.2f}%")
    print(
        "  Agent P/R/F1:   "
        f"{report['agent']['precision']:.3f} / "
        f"{report['agent']['recall']:.3f} / "
        f"{report['agent']['f1']:.3f}"
    )
    print(
        "  Cust P/R/F1:    "
        f"{report['customer']['precision']:.3f} / "
        f"{report['customer']['recall']:.3f} / "
        f"{report['customer']['f1']:.3f}"
    )
    print(
        "  WER all/agent/customer: "
        f"{report['wer']['all'] * 100:.2f}% / "
        f"{report['wer']['agent'] * 100:.2f}% / "
        f"{report['wer']['customer'] * 100:.2f}%"
    )

    if report.get("comparison_mode") == "text_aligned_roles":
        errors = report.get("mismatch_examples", [])
        if errors:
            print("\nFirst role mismatches in text alignment:")
            for item in errors[:max_errors]:
                print(
                    f"  word#{item['aligned_position']:5d} "
                    f"{item['truth_role']:>8}->{item['pred_role']:<8} "
                    f"truth='{item['truth_word']}' pred='{item['pred_word']}'"
                )
        else:
            print("\nNo role mismatches in text alignment.")
        return

    errors = sorted(
        report["mismatch_intervals"],
        key=lambda item: item["seconds"],
        reverse=True,
    )
    if errors:
        print("\nLargest role mismatches:")
        for item in errors[:max_errors]:
            print(
                f"  {item['start']:8.2f}-{item['end']:8.2f}s "
                f"{item['truth_role']:>8} -> {item['pred_role']:<8} "
                f"({item['seconds']:.2f}s)"
            )
    else:
        print("\nNo role mismatches at the selected time step.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Ground-truth JSON/CSV/TXT transcript")
    parser.add_argument("--result", required=True, help="Pipeline result JSON or segment list")
    parser.add_argument("--truth-agent-name", default="", help="Optional expected agent name")
    parser.add_argument("--pred-agent-name", default="", help="Optional predicted agent name override")
    parser.add_argument("--step", type=float, default=0.25, help="Time scoring step in seconds")
    parser.add_argument("--save-json", default="", help="Optional path to save metrics JSON")
    parser.add_argument("--max-errors", type=int, default=20, help="Largest mismatch intervals to print")
    args = parser.parse_args()

    truth_rows = load_segments(Path(args.truth))
    pred_rows = load_segments(Path(args.result))
    truth = normalise_segments(truth_rows, is_truth=True, agent_name=args.truth_agent_name)
    pred = normalise_segments(pred_rows, is_truth=False, agent_name=args.pred_agent_name)
    if truth and pred:
        report = compare(truth, pred, step=args.step)
        report["comparison_mode"] = "time_aligned_roles"
    else:
        truth_turns = normalise_turns(
            truth_rows,
            is_truth=True,
            agent_name=args.truth_agent_name,
        )
        pred_turns = normalise_turns(
            pred_rows,
            is_truth=False,
            agent_name=args.pred_agent_name,
        )
        report = compare_text_roles(truth_turns, pred_turns)
    report["truth_file"] = str(Path(args.truth))
    report["result_file"] = str(Path(args.result))
    report["truth_segments"] = len(truth) if truth else len(truth_rows)
    report["pred_segments"] = len(pred) if pred else len(pred_rows)

    print_report(report, args.max_errors)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved metrics: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
