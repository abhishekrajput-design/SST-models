"""
Test every registered transcriber on a single audio file (sequential, full GPU).

Run:
    python test_all_models.py --input data/raw_calls/audio_04_12_2026_12_28_59_vrcta2.mp3
    python test_all_models.py --only whisper-large-v3-turbo cohere-transcribe-03-2026
    python test_all_models.py --skip vibevoice-asr  # 18 GB model

Outputs:
    data/processed/<audio>__<model>/result.json
    data/processed/<audio>__<model>/transcript.txt
    data/model_comparison.json
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Load .env (HF_TOKEN)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.transcribers import TRANSCRIBERS, get_transcriber  # noqa: E402

MODELS_TO_TEST = [
    "whisper-large-v3-turbo",
    "cohere-transcribe-03-2026",
    "parakeet-tdt-0.6b-v3",
    "qwen3-asr-1.7b",
    "vibevoice-asr",
]


def vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def fmt_time(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def write_outputs(out_dir: str, audio_path: str, model_name: str,
                  segments: list, elapsed: float):
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "audio_file":              audio_path.replace("\\", "/"),
        "model":                   model_name,
        "processed_at":            datetime.now().isoformat(),
        "processing_time_seconds": round(elapsed, 2),
        "total_segments":          len(segments),
        "segments":                segments,
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [f"TRANSCRIPT — {os.path.basename(audio_path)}  ({model_name})", "=" * 70, ""]
    for s in segments:
        spk = s.get("identified_speaker") or s.get("speaker") or "?"
        lines.append(f"[{fmt_time(s['start'])} → {fmt_time(s['end'])}] [{spk}] {s['text']}")
    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_one(name: str, audio_path: str, language: str) -> dict:
    print(f"\n{'━' * 60}\n  {name}\n{'━' * 60}")
    vram_before = vram_mb()
    t_load = time.time()

    try:
        transcriber = get_transcriber(name, device="cuda")
        transcriber.load()
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return {"model": name, "status": "load_failed", "error": str(e)}

    load_time = time.time() - t_load
    vram_after_load = vram_mb()
    print(f"  Loaded in {load_time:.1f}s  ·  VRAM: +{vram_after_load - vram_before:.0f} MB")

    t_trans = time.time()
    try:
        segments = transcriber.transcribe(audio_path, language=language)
    except Exception as e:
        print(f"  TRANSCRIBE FAILED: {e}")
        traceback.print_exc()
        transcriber.unload()
        return {"model": name, "status": "transcribe_failed", "error": str(e),
                "load_time_s": round(load_time, 1)}
    trans_time = time.time() - t_trans
    vram_peak = vram_mb()
    print(f"  Transcribed in {trans_time:.1f}s  ·  VRAM peak: {vram_peak:.0f} MB  ·  {len(segments)} segments")

    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = os.path.join("data", "processed", f"{audio_name}__{name}")
    write_outputs(out_dir, audio_path, name, segments, trans_time)
    print(f"  Saved → {out_dir}/result.json")

    preview = segments[0]["text"] if segments else ""
    transcriber.unload()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return {
        "model":           name,
        "status":          "ok",
        "load_time_s":     round(load_time, 1),
        "transcribe_s":    round(trans_time, 1),
        "vram_mb":         round(vram_peak, 0),
        "segments":        len(segments),
        "preview":         preview[:120],
        "out_dir":         out_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to audio file")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    todo = []
    for m in MODELS_TO_TEST:
        if args.only and m not in args.only:
            continue
        if m in args.skip:
            continue
        todo.append(m)

    print(f"Audio: {args.input}")
    print(f"Models to test: {', '.join(todo)}")

    results = []
    for name in todo:
        results.append(run_one(name, args.input, args.language))

    # Comparison summary
    print(f"\n\n{'=' * 60}\n  COMPARISON\n{'=' * 60}")
    try:
        from tabulate import tabulate
        rows = []
        for r in results:
            if r["status"] != "ok":
                rows.append([r["model"], "—", "—", "—", "—", r["status"]])
            else:
                rows.append([r["model"], f"{r['load_time_s']}s",
                            f"{r['transcribe_s']}s", f"{r['vram_mb']:.0f}MB",
                            r["segments"], r["preview"][:60]])
        print(tabulate(rows, headers=["Model", "Load", "Transcribe", "VRAM", "Segs", "Preview"],
                       tablefmt="rounded_outline"))
    except ImportError:
        for r in results:
            print(f"  {r['model']:30s}  {r.get('status', '?')}")

    out = "data/model_comparison.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
