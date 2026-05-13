#!/usr/bin/env python
"""Verify Zak role accuracy without index-based leakage.

This script scores predictions against Gemini labels by timestamp overlap. It
also reports the simple alternation baseline separately because the current Zak
Gemini label files alternate customer/agent perfectly; that is useful for data
QA but is not proof of voice-model accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR))
sys.path.insert(0, str(CALL_PROCESSOR_DIR / "scripts"))

from train_zak_pure_embeddings import load_labelled_calls  # noqa: E402


def norm_role(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"agent", "agent_00", "zak", "zak raissi", "zak raissi barnet"}:
        return "agent"
    if raw == "ag" or "agent" in raw:
        return "agent"
    if raw.startswith("customer") or raw in {"cust", "caller"}:
        return "customer"
    return "unknown"


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def pct(num: float, den: float) -> float:
    return round((num / den * 100.0) if den else 0.0, 2)


def prepared_gt_segments(call) -> list[dict]:
    rows = []
    audio_end = len(call.audio) / call.sr
    for idx, seg in enumerate(call.segments, start=1):
        role = norm_role(seg.get("speaker"))
        if role not in {"agent", "customer"}:
            continue
        start = float(seg.get("start") or 0.0) + call.offset_s
        end = float(seg.get("end") or 0.0) + call.offset_s
        if end <= start or start >= audio_end:
            continue
        rows.append(
            {
                "segment": idx,
                "start": start,
                "end": min(end, audio_end),
                "speaker": role,
                "text": str(seg.get("text") or ""),
            }
        )
    return rows


def role_for_prediction(seg: dict) -> str:
    return norm_role(
        seg.get("identified_speaker")
        or seg.get("role")
        or seg.get("speaker_role")
        or seg.get("display_speaker")
    )


def score_by_overlap(gt_segments: list[dict], pred_segments: list[dict]) -> dict:
    total = correct = 0
    role_total = Counter()
    role_correct = Counter()
    duration_total = Counter()
    duration_correct = Counter()
    no_overlap = 0
    errors = []

    for gt in gt_segments:
        gt_start = float(gt["start"])
        gt_end = float(gt["end"])
        gt_role = gt["speaker"]
        role_seconds = Counter()
        overlap_seconds = 0.0

        for pred in pred_segments:
            pred_start = float(pred.get("start") or 0.0)
            pred_end = float(pred.get("end") or pred_start)
            seconds = overlap(gt_start, gt_end, pred_start, pred_end)
            if seconds <= 0:
                continue
            role_seconds[role_for_prediction(pred)] += seconds
            overlap_seconds += seconds

        if role_seconds:
            pred_role = role_seconds.most_common(1)[0][0]
        else:
            pred_role = "unknown"
            no_overlap += 1

        ok = pred_role == gt_role
        total += 1
        correct += int(ok)
        role_total[gt_role] += 1
        role_correct[gt_role] += int(ok)
        gt_duration = max(gt_end - gt_start, 0.0)
        duration_total[gt_role] += gt_duration
        duration_correct[gt_role] += min(role_seconds.get(gt_role, 0.0), gt_duration)

        if not ok and len(errors) < 40:
            errors.append(
                {
                    "segment": gt["segment"],
                    "time": [round(gt_start, 2), round(gt_end, 2)],
                    "expected": gt_role,
                    "predicted": pred_role,
                    "overlap_seconds": round(overlap_seconds, 2),
                    "text": gt["text"][:140],
                }
            )

    return {
        "segments": total,
        "overall_accuracy": pct(correct, total),
        "agent_accuracy": pct(role_correct["agent"], role_total["agent"]),
        "customer_accuracy": pct(role_correct["customer"], role_total["customer"]),
        "agent_correct": int(role_correct["agent"]),
        "agent_total": int(role_total["agent"]),
        "customer_correct": int(role_correct["customer"]),
        "customer_total": int(role_total["customer"]),
        "duration_weighted": {
            "agent_accuracy": pct(duration_correct["agent"], duration_total["agent"]),
            "customer_accuracy": pct(duration_correct["customer"], duration_total["customer"]),
            "overall_accuracy": pct(
                duration_correct["agent"] + duration_correct["customer"],
                duration_total["agent"] + duration_total["customer"],
            ),
        },
        "no_overlap_segments": no_overlap,
        "errors": errors,
    }


def alternation_baseline(gt_segments: list[dict]) -> dict:
    if not gt_segments:
        return {"segments": 0, "accuracy": 0.0, "note": "no labels"}
    start_role = gt_segments[0]["speaker"]
    roles = [start_role if i % 2 == 0 else ("agent" if start_role == "customer" else "customer")
             for i in range(len(gt_segments))]
    correct = sum(int(role == gt["speaker"]) for role, gt in zip(roles, gt_segments))
    adjacent_same = sum(
        int(gt_segments[i]["speaker"] == gt_segments[i - 1]["speaker"])
        for i in range(1, len(gt_segments))
    )
    return {
        "segments": len(gt_segments),
        "accuracy": pct(correct, len(gt_segments)),
        "first_role": start_role,
        "adjacent_same_role_pairs": adjacent_same,
        "note": "label-structure baseline only; not voice-model accuracy",
    }


def load_result_segments(path: Path, field: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get(field)
    if rows is None and field == "segments":
        rows = data if isinstance(data, list) else []
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a list field named {field}")
    return rows


def infer_call_id_from_result_path(path: Path) -> str:
    stem = path.parent.name
    if stem.startswith("enhanced_"):
        stem = stem[len("enhanced_"):]
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    for suffix in (
        "_supervised_verify",
        "_zak_retrain",
        "_ui_parakeet",
        "_parakeet",
        "_retrain",
        "_ui",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.strip()


def resolve_call_id_for_result(data: dict, path: Path, calls_by_id: dict[str, object]) -> str:
    call_id = str(data.get("call_id") or data.get("target_call_id") or "").strip()
    if call_id in calls_by_id:
        return call_id

    candidates = [
        call_id,
        infer_call_id_from_result_path(path),
        str(data.get("audio_file") or ""),
        str(data.get("orig_file") or ""),
        str(data.get("asr_audio_file") or ""),
        str(data.get("diarization_audio_file") or ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        for known_id in calls_by_id:
            if known_id and known_id in candidate:
                return known_id
    return call_id or infer_call_id_from_result_path(path)


def score_saved_results(calls_by_id: dict[str, object], result_paths: Iterable[Path], field: str) -> dict:
    out = {}
    for path in result_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        call_id = resolve_call_id_for_result(data, path, calls_by_id)
        if not call_id:
            # Existing local call_02 reports do not carry call_id in the root.
            if "call02" in path.name.lower() or "call_02" in path.name.lower():
                call_id = "20260429T085638315_375379"
        call = calls_by_id.get(call_id)
        result_key = path.parent.name if path.name == "result.json" else path.name
        if call is None:
            out[result_key] = {"skipped": True, "reason": f"could not resolve call_id {call_id!r}"}
            continue
        pred_segments = data.get(field)
        if not isinstance(pred_segments, list):
            out[result_key] = {"skipped": True, "reason": f"missing list field {field!r}"}
            continue
        out[result_key] = {
            "call_id": call_id,
            "field": field,
            "score": score_by_overlap(prepared_gt_segments(call), pred_segments),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default=str(CALL_PROCESSOR_DIR / "data" / "training"))
    parser.add_argument("--audio-root", default=str(REPO_ROOT / "traning_data" / "zak_raissi"))
    parser.add_argument("--offset-mode", choices=("detected", "none", "auto-source"), default="detected")
    parser.add_argument("--include-source-prefix", default="")
    parser.add_argument("--include-call", action="append", default=[])
    parser.add_argument("--exclude-call", action="append", default=[])
    parser.add_argument("--result", action="append", default=[], help="Saved result JSON to score")
    parser.add_argument("--field", default="segments", help="Result JSON segment field to score")
    parser.add_argument(
        "--out",
        default=str(CALL_PROCESSOR_DIR / "data" / "agent_voiceprints" / "zak_corrected_verification.json"),
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir).resolve()
    audio_root = Path(args.audio_root).resolve()
    calls, skipped_calls = load_labelled_calls(
        labels_dir,
        audio_root,
        offset_mode=args.offset_mode,
        include_source_prefix=args.include_source_prefix,
        include_call_names=set(args.include_call),
        exclude_call_names=set(args.exclude_call),
    )
    calls_by_id = {call.call_id: call for call in calls}

    label_reports = {}
    all_gt_segments = []
    for call in calls:
        gt_segments = prepared_gt_segments(call)
        all_gt_segments.extend(gt_segments)
        label_reports[call.call_name] = {
            "call_id": call.call_id,
            "audio": str(call.audio_path),
            "offset_s": call.offset_s,
            "segments": len(gt_segments),
            "agent_segments": sum(1 for s in gt_segments if s["speaker"] == "agent"),
            "customer_segments": sum(1 for s in gt_segments if s["speaker"] == "customer"),
            "alternation_baseline": alternation_baseline(gt_segments),
        }

    result_paths = [Path(p).resolve() for p in args.result]
    saved_results = score_saved_results(calls_by_id, result_paths, args.field) if result_paths else {}

    report = {
        "labels_dir": str(labels_dir),
        "audio_root": str(audio_root),
        "offset_mode": args.offset_mode,
        "include_source_prefix": args.include_source_prefix,
        "include_calls": args.include_call,
        "exclude_calls": args.exclude_call,
        "skipped_calls": skipped_calls,
        "label_reports": label_reports,
        "all_labels_alternation_baseline": alternation_baseline(all_gt_segments),
        "saved_results": saved_results,
        "note": (
            "Correct model verification must use timestamp overlap and held-out calls. "
            "The alternation baseline is reported only to reveal label structure."
        ),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": report}, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
