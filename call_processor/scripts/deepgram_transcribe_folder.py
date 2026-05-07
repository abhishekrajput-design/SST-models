#!/usr/bin/env python
"""Transcribe local audio files with Deepgram Nova 3.

This produces transcript artifacts only. It does not convert Deepgram speaker IDs
to agent/customer labels, because those labels need verification before training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_PROCESSOR_DIR = REPO_ROOT / "call_processor"
sys.path.insert(0, str(CALL_PROCESSOR_DIR / "scripts"))

from deepgram_zak_compare import (  # noqa: E402
    DEFAULT_KEYTERMS,
    call_deepgram,
    load_dotenv_key,
    normalize_segments,
    summarize_response,
)

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}


def find_audio(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in AUDIO_EXTS:
        return [root]
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTS
        and not path.name.startswith("audio_16k")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Audio file or directory containing audio files")
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--suffix", default=".deepgram_nova3.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keyterm", action="append", default=[])
    args = parser.parse_args()

    api_key = load_dotenv_key()
    if not api_key:
        raise SystemExit("DEEPGRAM_API_KEY is not set")

    audio_paths = find_audio(Path(args.root).resolve())
    if not audio_paths:
        raise SystemExit("No audio files found")

    keyterms = [*DEFAULT_KEYTERMS, *args.keyterm]
    results = []
    for idx, audio_path in enumerate(audio_paths, start=1):
        out_path = audio_path.with_suffix(args.suffix)
        if out_path.exists() and not args.force:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            results.append({
                "audio": str(audio_path),
                "out": str(out_path),
                "cached": True,
                "segments": len(data.get("segments") or []),
            })
            continue
        print(f"[{idx}/{len(audio_paths)}] Deepgram {audio_path.name}", flush=True)
        raw = call_deepgram(audio_path, api_key, args.model, keyterms)
        segments = normalize_segments(raw)
        payload = {
            "audio": str(audio_path),
            "model": args.model,
            "source": "deepgram",
            "summary": summarize_response(raw),
            "segments": segments,
            "raw": raw,
            "training_warning": (
                "Deepgram speaker IDs are diarization IDs, not verified agent/customer "
                "roles. Do not use this file as Zak training labels without review."
            ),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        results.append({
            "audio": str(audio_path),
            "out": str(out_path),
            "cached": False,
            "segments": len(segments),
            **payload["summary"],
        })

    print(json.dumps({"count": len(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
