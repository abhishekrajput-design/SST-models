from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.transcribers import get_transcriber  # noqa: E402


def _word_count(text: str) -> int:
    return len((text or "").strip().split())


def _audio_stats(path: Path) -> dict:
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if len(data) == 0:
            return {"sample_rate": sr, "rms_db": None, "peak_db": None}
        rms = math.sqrt(float((data * data).mean()))
        peak = float(abs(data).max())
        return {
            "sample_rate": sr,
            "rms_db": round(20 * math.log10(max(rms, 1e-9)), 2),
            "peak_db": round(20 * math.log10(max(peak, 1e-9)), 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe local enhancement variants with one ASR load.")
    parser.add_argument(
        "--dir",
        default="data/processed/ekome8_enhance_tests",
        help="Directory containing WAV variants, relative to call_processor by default.",
    )
    parser.add_argument("--pattern", default="*360_480.wav")
    parser.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    test_dir = Path(args.dir)
    if not test_dir.is_absolute():
        test_dir = ROOT / test_dir
    paths = sorted(test_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {test_dir / args.pattern}")

    transcriber = get_transcriber(args.model, device=args.device)
    transcriber.load()

    report: list[dict] = []
    try:
        for path in paths:
            print(f"[variant] {path.name}", flush=True)
            segments = transcriber.transcribe(str(path), language="en")
            words = sum(_word_count(seg.get("text", "")) for seg in segments)
            non_empty = [seg for seg in segments if (seg.get("text") or "").strip()]
            report.append(
                {
                    "file": path.name,
                    "audio": _audio_stats(path),
                    "segments": len(segments),
                    "words": words,
                    "first_segments": non_empty[:8],
                    "all_text": " ".join((seg.get("text") or "").strip() for seg in non_empty),
                }
            )
    finally:
        unload = getattr(transcriber, "unload", None)
        if callable(unload):
            unload()

    out_path = Path(args.out) if args.out else test_dir / f"{args.model.replace('/', '_')}_{args.pattern.replace('*', 'star')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[report] {out_path}", flush=True)
    for row in report:
        print(
            f"{row['file']}: segments={row['segments']} words={row['words']} "
            f"rms={row['audio'].get('rms_db')} peak={row['audio'].get('peak_db')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
