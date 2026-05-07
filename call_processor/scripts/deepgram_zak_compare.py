#!/usr/bin/env python
"""Transcribe Zak training calls with Deepgram Nova 3 and score role accuracy.

Deepgram returns diarization speaker IDs, not business roles. For evaluation we
map each Deepgram speaker ID to agent/customer by maximum overlap with the local
Gemini data.json labels, then score time-aligned role accuracy. That makes this
an upper-bound diarization comparison, not a production agent-ID solution.
"""
from __future__ import annotations

import argparse
import itertools
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR / "scripts"))

from train_zak_pure_embeddings import (  # noqa: E402
    load_folder_data_json_calls,
    primary_audio_in_folder,
)

API_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_KEYTERMS = [
    "Zak Raissi",
    "Car Planet",
    "Barnet",
    "warranty",
    "finance",
    "deposit",
]


def load_dotenv_key() -> str:
    if os.environ.get("DEEPGRAM_API_KEY"):
        return os.environ["DEEPGRAM_API_KEY"].strip()
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return ""
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DEEPGRAM_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def mime_for(path: Path) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }.get(path.suffix.lower(), "application/octet-stream")


def call_deepgram(audio_path: Path, api_key: str, model: str, keyterms: list[str]) -> dict[str, Any]:
    params: list[tuple[str, str]] = [
        ("model", model),
        ("language", "en"),
        ("diarize", "true"),
        ("utterances", "true"),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("numerals", "true"),
        ("filler_words", "false"),
        ("mip_opt_out", "true"),
    ]
    for term in keyterms:
        if term.strip():
            params.append(("keyterm", term.strip()))
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        data=audio_path.read_bytes(),
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": mime_for(audio_path),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_segments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    res = raw.get("results") or {}
    utterances = res.get("utterances") or []
    out: list[dict[str, Any]] = []
    if utterances:
        for item in utterances:
            text = str(item.get("transcript") or "").strip()
            if not text:
                continue
            speaker = item.get("speaker", 0)
            out.append({
                "start": float(item.get("start") or 0.0),
                "end": float(item.get("end") or 0.0),
                "speaker": f"SPEAKER_{int(speaker):02d}",
                "text": text,
                "confidence": float(item.get("confidence") or 0.0),
            })
        return out

    channels = res.get("channels") or []
    alt = (channels[0].get("alternatives") or [{}])[0] if channels else {}
    words = alt.get("words") or []
    cur: dict[str, Any] | None = None
    for word in words:
        speaker = f"SPEAKER_{int(word.get('speaker') or 0):02d}"
        text = str(word.get("punctuated_word") or word.get("word") or "").strip()
        if not text:
            continue
        if cur is None or cur["speaker"] != speaker:
            if cur:
                out.append(cur)
            cur = {
                "start": float(word.get("start") or 0.0),
                "end": float(word.get("end") or 0.0),
                "speaker": speaker,
                "text": text,
                "confidence": 0.0,
            }
        else:
            cur["end"] = float(word.get("end") or cur["end"])
            cur["text"] += " " + text
    if cur:
        out.append(cur)
    return out


def role_at(segments: list[dict[str, Any]], t: float, field: str = "speaker") -> str:
    for seg in segments:
        try:
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or 0.0)
        except Exception:
            continue
        if start <= t < end:
            return str(seg.get(field) or "").strip().lower()
    return ""


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_speaker_role_map(pred: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, str]:
    overlaps: dict[str, dict[str, float]] = {}
    for pseg in pred:
        speaker = str(pseg.get("speaker") or "")
        ps = float(pseg.get("start") or 0.0)
        pe = float(pseg.get("end") or 0.0)
        bucket = overlaps.setdefault(speaker, {"agent": 0.0, "customer": 0.0})
        for tseg in truth:
            role = str(tseg.get("speaker") or "").lower().strip()
            if role not in {"agent", "customer"}:
                continue
            ts = float(tseg.get("start") or 0.0)
            te = float(tseg.get("end") or 0.0)
            bucket[role] += overlap_seconds(ps, pe, ts, te)
    speakers = sorted(overlaps)
    truth_roles = {
        str(seg.get("speaker") or "").lower().strip()
        for seg in truth
        if str(seg.get("speaker") or "").lower().strip() in {"agent", "customer"}
    }
    if not speakers:
        return {}

    best_map: dict[str, str] = {}
    best_score = -1.0
    roles = ("agent", "customer")
    for assignment in itertools.product(roles, repeat=len(speakers)):
        if len(speakers) >= 2 and truth_roles == {"agent", "customer"}:
            if "agent" not in assignment or "customer" not in assignment:
                continue
        score = 0.0
        for speaker, role in zip(speakers, assignment):
            score += overlaps[speaker].get(role, 0.0)
        if score > best_score:
            best_score = score
            best_map = dict(zip(speakers, assignment))

    if best_map:
        return best_map
    return {
        speaker: ("agent" if scores.get("agent", 0.0) >= scores.get("customer", 0.0) else "customer")
        for speaker, scores in overlaps.items()
    }


def score_roles(pred: list[dict[str, Any]], truth: list[dict[str, Any]], step_s: float) -> dict[str, Any]:
    role_map = build_speaker_role_map(pred, truth)
    pred_role_segments = [
        {**seg, "role": role_map.get(str(seg.get("speaker") or ""), "unknown")}
        for seg in pred
    ]
    max_t = max((float(seg.get("end") or 0.0) for seg in truth), default=0.0)
    total = correct = 0
    role_total = {"agent": 0, "customer": 0}
    role_correct = {"agent": 0, "customer": 0}
    unknown = 0
    t = 0.0
    while t < max_t:
        gt = role_at(truth, t)
        if gt in {"agent", "customer"}:
            pred_role = role_at(pred_role_segments, t, field="role")
            total += 1
            role_total[gt] += 1
            if pred_role == gt:
                correct += 1
                role_correct[gt] += 1
            elif pred_role not in {"agent", "customer"}:
                unknown += 1
        t += step_s

    def pct(num: int, den: int) -> float:
        return round((num / den * 100.0) if den else 0.0, 2)

    return {
        "role_map": role_map,
        "time_step_s": step_s,
        "ticks": total,
        "unknown_ticks": unknown,
        "overall_accuracy": pct(correct, total),
        "agent_accuracy": pct(role_correct["agent"], role_total["agent"]),
        "customer_accuracy": pct(role_correct["customer"], role_total["customer"]),
        "agent_ticks": role_total["agent"],
        "customer_ticks": role_total["customer"],
    }


def summarize_response(raw: dict[str, Any]) -> dict[str, Any]:
    res = raw.get("results") or {}
    alt = (((res.get("channels") or [{}])[0]).get("alternatives") or [{}])[0]
    return {
        "request_id": (raw.get("metadata") or {}).get("request_id"),
        "duration": (raw.get("metadata") or {}).get("duration"),
        "utterances": len(res.get("utterances") or []),
        "words": len(alt.get("words") or []),
        "transcript_chars": len(str(alt.get("transcript") or "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", default=str(REPO_ROOT / "traning_data" / "zak_raissi"))
    parser.add_argument("--out-dir", default=str(CALL_PROCESSOR_DIR / "data" / "deepgram_zak_eval" / "nova3_folder_data_20260506"))
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--step-s", type=float, default=0.25)
    parser.add_argument("--keyterm", action="append", default=DEFAULT_KEYTERMS)
    args = parser.parse_args()

    api_key = load_dotenv_key()
    if not api_key:
        print("[error] DEEPGRAM_API_KEY is not set in env or .env", file=sys.stderr)
        return 2

    audio_root = Path(args.audio_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"
    seg_dir = out_dir / "segments"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    calls, skipped, _ = load_folder_data_json_calls(audio_root, offset_mode="none")
    if args.limit > 0:
        calls = calls[: args.limit]
    print(f"[data] calls={len(calls)} skipped={len(skipped)} model={args.model}")
    for item in skipped:
        print(f"[skip] {item}")

    results = []
    for idx, call in enumerate(calls, start=1):
        audio_path = primary_audio_in_folder(call.audio_path.parent) or call.audio_path
        raw_path = raw_dir / f"{call.call_name}_{audio_path.stem}.deepgram.json"
        seg_path = seg_dir / f"{call.call_name}_{audio_path.stem}.segments.json"
        print(f"[{idx}/{len(calls)}] {call.call_name} {audio_path.name}", flush=True)
        t0 = time.time()
        if raw_path.exists() and not args.force:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            elapsed = 0.0
        else:
            try:
                raw = call_deepgram(audio_path, api_key, args.model, args.keyterm)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                print(f"[error] Deepgram HTTP {exc.code}: {body}", file=sys.stderr)
                return 3
            elapsed = time.time() - t0
            raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        pred_segments = normalize_segments(raw)
        seg_path.write_text(json.dumps(pred_segments, indent=2), encoding="utf-8")
        score = score_roles(pred_segments, call.segments, args.step_s)
        response_summary = summarize_response(raw)
        result = {
            "call": call.call_name,
            "call_id": call.call_id,
            "audio": str(audio_path),
            "audio_duration_s": round(len(call.audio) / call.sr, 2),
            "deepgram_elapsed_s": round(elapsed, 2),
            "raw_response": str(raw_path),
            "segments_json": str(seg_path),
            "truth_segments": len(call.segments),
            "pred_segments": len(pred_segments),
            "deepgram": response_summary,
            "role_accuracy": score,
        }
        results.append(result)
        print(
            "  acc={overall_accuracy}% agent={agent_accuracy}% customer={customer_accuracy}% "
            "pred_segments={pred_segments} utterances={utterances}".format(
                pred_segments=len(pred_segments),
                utterances=response_summary["utterances"],
                **score,
            ),
            flush=True,
        )

    total_ticks = sum(item["role_accuracy"]["ticks"] for item in results)
    total_unknown = sum(item["role_accuracy"]["unknown_ticks"] for item in results)
    weighted_overall = sum(
        item["role_accuracy"]["overall_accuracy"] * item["role_accuracy"]["ticks"]
        for item in results
    ) / max(total_ticks, 1)

    agent_ticks = sum(item["role_accuracy"]["agent_ticks"] for item in results)
    customer_ticks = sum(item["role_accuracy"]["customer_ticks"] for item in results)
    agent_weighted = sum(
        item["role_accuracy"]["agent_accuracy"] * item["role_accuracy"]["agent_ticks"]
        for item in results
    ) / max(agent_ticks, 1)
    customer_weighted = sum(
        item["role_accuracy"]["customer_accuracy"] * item["role_accuracy"]["customer_ticks"]
        for item in results
    ) / max(customer_ticks, 1)

    report = {
        "model": args.model,
        "audio_root": str(audio_root),
        "out_dir": str(out_dir),
        "calls": len(results),
        "skipped_calls": skipped,
        "comparison_note": (
            "Deepgram speaker IDs are mapped to agent/customer by maximum overlap "
            "with local data.json labels. This is an upper-bound diarization score."
        ),
        "weighted_role_accuracy": {
            "overall_accuracy": round(weighted_overall, 2),
            "agent_accuracy": round(agent_weighted, 2),
            "customer_accuracy": round(customer_weighted, 2),
            "ticks": total_ticks,
            "unknown_ticks": total_unknown,
        },
        "results": results,
    }
    report_path = out_dir / "deepgram_zak_nova3_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[out] {report_path}")
    print(
        f"[summary] overall={weighted_overall:.2f}% agent={agent_weighted:.2f}% "
        f"customer={customer_weighted:.2f}% calls={len(results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
