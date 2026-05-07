#!/usr/bin/env python
"""Local Whisper transcript preview for training-call identity QA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR))

from src.transcribers.whisper_turbo import WhisperTurboTranscriber  # noqa: E402


def primary_audio(folder: Path) -> Path | None:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}:
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.audio_root).resolve()
    transcriber = WhisperTurboTranscriber(device=args.device, model_size=args.model)
    previews = []
    try:
        for folder in sorted(root.glob("call_*")):
            if not folder.is_dir():
                continue
            audio = primary_audio(folder)
            if audio is None:
                continue
            print(f"\n{folder.name} {audio.name}")
            segments = transcriber.transcribe(str(audio))[:args.max_segments]
            preview_segments = []
            joined = " ".join(seg["text"] for seg in segments).lower()
            has_name = all(part in joined for part in args.agent_name.lower().split())
            for seg in segments:
                item = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                preview_segments.append(item)
                print(f"  {seg['start']}-{seg['end']}: {seg['text'][:180]}")
            previews.append({
                "folder": folder.name,
                "audio": str(audio),
                "has_agent_name_in_preview": has_name,
                "segments": preview_segments,
            })
    finally:
        transcriber.unload()

    report = {
        "audio_root": str(root),
        "agent_name": args.agent_name,
        "model": args.model,
        "previews": previews,
    }
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
