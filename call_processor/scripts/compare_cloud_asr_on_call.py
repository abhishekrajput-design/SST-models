#!/usr/bin/env python
"""Compare cloud ASR diarization against a local speaker-role reference.

This is diagnostic tooling, not training data generation. Deepgram and
AssemblyAI return generic speaker labels, so this script maps provider speaker
IDs to local AGENT/CUSTOMER regions by overlap and reports coverage.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
ASSEMBLY_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLY_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"


def load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


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


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def role_from_segment(seg: dict[str, Any]) -> str:
    speaker = str(seg.get("speaker") or "").strip().lower()
    if speaker == "agent":
        return "AGENT"
    if speaker == "customer":
        return "CUSTOMER"
    role = str(seg.get("identified_speaker") or "").upper()
    if role == "AGENT":
        return "AGENT"
    label = str(seg.get("display_speaker") or seg.get("speaker") or "").lower()
    if "zak" in label or "hussein" in label or "hussain" in label:
        return "AGENT"
    return "CUSTOMER"


def load_reference(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("speaker_segments") or data.get("segments") or []
    out = []
    for seg in raw:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        if end <= start:
            continue
        out.append({
            "start": start,
            "end": end,
            "speaker": str(seg.get("speaker") or ""),
            "display_speaker": str(seg.get("display_speaker") or seg.get("speaker") or ""),
            "role": role_from_segment(seg),
        })
    return sorted(out, key=lambda item: (item["start"], item["end"]))


def normalize_deepgram(raw: dict[str, Any]) -> list[dict[str, Any]]:
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


def normalize_assemblyai(raw: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw.get("utterances") or []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        speaker = str(item.get("speaker") or "A")
        out.append({
            "start": float(item.get("start") or 0.0) / 1000.0,
            "end": float(item.get("end") or 0.0) / 1000.0,
            "speaker": f"SPEAKER_{speaker}",
            "text": text,
            "confidence": float(item.get("confidence") or 0.0),
        })
    return out


def call_deepgram(audio: Path) -> dict[str, Any]:
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")
    params: list[tuple[str, str]] = [
        ("model", "nova-3"),
        ("language", "en"),
        ("diarize", "true"),
        ("utterances", "true"),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("numerals", "true"),
        ("filler_words", "false"),
        ("mip_opt_out", "true"),
        ("keyterm", "Zak"),
        ("keyterm", "Zak Raissi"),
        ("keyterm", "Car Planet"),
        ("keyterm", "finance"),
        ("keyterm", "deposit"),
    ]
    req = urllib.request.Request(
        f"{DEEPGRAM_URL}?{urllib.parse.urlencode(params)}",
        data=audio.read_bytes(),
        headers={"Authorization": f"Token {api_key}", "Content-Type": mime_for(audio)},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_assemblyai(audio: Path, min_speakers: int, max_speakers: int) -> dict[str, Any]:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set")
    upload_req = urllib.request.Request(
        ASSEMBLY_UPLOAD_URL,
        data=audio.read_bytes(),
        headers={"authorization": api_key, "content-type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(upload_req, timeout=180) as resp:
        upload_url = json.loads(resp.read().decode("utf-8"))["upload_url"]

    body = json.dumps({
        "audio_url": upload_url,
        "speech_models": ["universal-3-pro", "universal-2"],
        "speaker_labels": True,
        "speaker_options": {
            "min_speakers_expected": min_speakers,
            "max_speakers_expected": max_speakers,
        },
        "language_code": "en",
        "punctuate": True,
        "format_text": True,
    }).encode("utf-8")
    headers = {"authorization": api_key, "content-type": "application/json"}
    submit_req = urllib.request.Request(ASSEMBLY_TRANSCRIPT_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(submit_req, timeout=60) as resp:
        transcript_id = json.loads(resp.read().decode("utf-8"))["id"]

    poll_url = f"{ASSEMBLY_TRANSCRIPT_URL}/{transcript_id}"
    deadline = time.time() + 1200
    while time.time() < deadline:
        poll_req = urllib.request.Request(poll_url, headers=headers)
        with urllib.request.urlopen(poll_req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        status = data.get("status")
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(f"AssemblyAI error: {data.get('error')}")
        time.sleep(3)
    raise TimeoutError("AssemblyAI transcript timed out")


def compare_provider(name: str, segments: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    ref_by_role = {"AGENT": 0.0, "CUSTOMER": 0.0}
    for ref in reference:
        ref_by_role[ref["role"]] += max(ref["end"] - ref["start"], 0.0)

    speaker_overlap: dict[str, dict[str, float]] = {}
    captured_by_role = {"AGENT": 0.0, "CUSTOMER": 0.0}
    words_by_role = {"AGENT": 0, "CUSTOMER": 0}
    chars_by_role = {"AGENT": 0, "CUSTOMER": 0}
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        if end <= start:
            continue
        spk = str(seg.get("speaker") or "UNKNOWN")
        text = str(seg.get("text") or "")
        bucket = speaker_overlap.setdefault(spk, {"AGENT": 0.0, "CUSTOMER": 0.0})
        seg_role_overlap = {"AGENT": 0.0, "CUSTOMER": 0.0}
        for ref in reference:
            ov = overlap_seconds(start, end, ref["start"], ref["end"])
            if ov <= 0:
                continue
            role = ref["role"]
            bucket[role] += ov
            seg_role_overlap[role] += ov
        role = "AGENT" if seg_role_overlap["AGENT"] >= seg_role_overlap["CUSTOMER"] else "CUSTOMER"
        captured_by_role[role] += max(end - start, 0.0)
        words_by_role[role] += len(text.split())
        chars_by_role[role] += len(text)

    speaker_role_map = {}
    for spk, overlap in speaker_overlap.items():
        speaker_role_map[spk] = "AGENT" if overlap["AGENT"] >= overlap["CUSTOMER"] else "CUSTOMER"

    return {
        "provider": name,
        "segments": len(segments),
        "speakers": sorted({str(seg.get("speaker") or "UNKNOWN") for seg in segments}),
        "speaker_role_map_by_local_overlap": speaker_role_map,
        "reference_seconds": {k: round(v, 2) for k, v in ref_by_role.items()},
        "provider_segment_seconds_by_mapped_role": {k: round(v, 2) for k, v in captured_by_role.items()},
        "words_by_mapped_role": words_by_role,
        "chars_by_mapped_role": chars_by_role,
        "speaker_overlap_seconds": {
            spk: {role: round(seconds, 2) for role, seconds in data.items()}
            for spk, data in speaker_overlap.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-dir", default=str(CALL_PROCESSOR_DIR / "data" / "provider_compare"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--providers", default="deepgram,assemblyai")
    parser.add_argument("--min-speakers", type=int, default=3)
    parser.add_argument("--max-speakers", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    audio = Path(args.audio).resolve()
    reference_path = Path(args.reference).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(reference_path)
    selected = [p.strip().lower() for p in args.providers.split(",") if p.strip()]

    raw_outputs: dict[str, dict[str, Any]] = {}
    normalized: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for provider in selected:
        raw_path = out_dir / f"{audio.stem}.{provider}.raw.json"
        norm_path = out_dir / f"{audio.stem}.{provider}.segments.json"
        try:
            if provider == "deepgram":
                if raw_path.exists() and not args.force:
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                else:
                    print("[compare] Deepgram Nova-3...", flush=True)
                    raw = call_deepgram(audio)
                    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
                segs = normalize_deepgram(raw)
            elif provider == "assemblyai":
                if raw_path.exists() and not args.force:
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                else:
                    print("[compare] AssemblyAI Universal-3 Pro/2...", flush=True)
                    raw = call_assemblyai(audio, args.min_speakers, args.max_speakers)
                    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
                segs = normalize_assemblyai(raw)
            else:
                print(f"[compare] skipping unknown provider: {provider}", flush=True)
                continue
        except Exception as exc:
            errors[provider] = repr(exc)
            print(f"[compare] {provider} failed: {exc}", flush=True)
            continue
        norm_path.write_text(json.dumps(segs, indent=2, ensure_ascii=False), encoding="utf-8")
        raw_outputs[provider] = raw
        normalized[provider] = segs

    report = {
        "audio": str(audio),
        "reference": str(reference_path),
        "reference_turns": len(reference),
        "providers": {
            provider: compare_provider(provider, segs, reference)
            for provider, segs in normalized.items()
        },
        "errors": errors,
        "warning": (
            "Provider speaker IDs are generic diarization labels. Role mapping here "
            "uses overlap with local Zak/customer diarization and is diagnostic only."
        ),
    }
    report_path = out_dir / f"{audio.stem}.provider_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "providers": {
            k: {
                "segments": v["segments"],
                "speakers": v["speakers"],
                "words_by_mapped_role": v["words_by_mapped_role"],
            }
            for k, v in report["providers"].items()
        },
        "errors": errors,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
