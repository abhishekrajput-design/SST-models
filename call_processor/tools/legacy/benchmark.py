"""
Pipeline Benchmarker — compare Whisper model configs on a single audio file.

Usage:
    python benchmark.py --input path/to/audio.wav
    python benchmark.py --input path/to/audio.wav --preset quality
    python benchmark.py --input path/to/audio.wav --language en

Presets (ordered fastest → best quality):
    tiny     — distil-small.en,   int8   (~500MB VRAM, 10x+ faster)
    fast     — distil-large-v3,   int8   (~2GB  VRAM, 5.8x faster)
    quality  — large-v3,          float16 (~3GB  VRAM, best accuracy)
    best     — large-v3-turbo,    float16 (~3.5GB VRAM, speed + quality)
"""

import argparse
import gc
import io
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Preset definitions ────────────────────────────────────────────────────────

PRESETS: Dict[str, Dict] = {
    "tiny": {
        "label": "Tiny (distil-small.en)",
        "model": "distil-small.en",
        "compute_type": "int8",
        "description": "~500MB VRAM — ultra fast, English only",
    },
    "fast": {
        "label": "Fast (distil-large-v3)",
        "model": "distil-large-v3",
        "compute_type": "int8",
        "description": "~2GB VRAM — 5.8x faster than large-v3, within 1% WER",
    },
    "quality": {
        "label": "Quality (large-v3 float16)",
        "model": "large-v3",
        "compute_type": "float16",
        "description": "~3GB VRAM — best accuracy, recommended for 6GB systems",
    },
    "best": {
        "label": "Best (large-v3-turbo float16)",
        "model": "large-v3-turbo",
        "compute_type": "float16",
        "description": "~3.5GB VRAM — speed + quality optimized turbo variant",
    },
}


def vram_mb() -> float:
    """Current GPU memory reserved in MB (captures both PyTorch and CTranslate2 allocations)."""
    if torch.cuda.is_available():
        # Use reserved (not just allocated) to capture CTranslate2's memory pool
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0.0


def run_preset(
    preset_key: str,
    audio_path: str,
    language: str,
    initial_prompt: str,
    download_root: str,
) -> Dict:
    """Run a single preset and return timing + transcript."""
    from faster_whisper import WhisperModel

    cfg = PRESETS[preset_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = cfg["compute_type"] if device == "cuda" else "float32"

    print(f"\n  Loading {cfg['label']}...")
    vram_before = vram_mb()
    t_load = time.time()

    model = WhisperModel(
        cfg["model"],
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )

    load_time = round(time.time() - t_load, 2)
    vram_after = vram_mb()
    vram_used = round(vram_after - vram_before, 1)

    print(f"  Loaded in {load_time}s | VRAM delta: {vram_used:.0f}MB")
    print(f"  Transcribing...")

    t_transcribe = time.time()
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        word_timestamps=False,
        temperature=0,
        no_speech_threshold=0.6,
    )

    text_parts = [seg.text.strip() for seg in segments_iter]
    transcript = " ".join(text_parts)
    transcribe_time = round(time.time() - t_transcribe, 2)

    # Unload
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "preset": preset_key,
        "label": cfg["label"],
        "model": cfg["model"],
        "compute_type": compute_type,
        "device": device,
        "load_time_s": load_time,
        "transcribe_time_s": transcribe_time,
        "total_time_s": round(load_time + transcribe_time, 2),
        "vram_delta_mb": vram_used,
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 3),
        "transcript": transcript,
    }


def print_results_table(results: List[Dict]):
    """Print a comparison table of benchmark results."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("\n[install tabulate for a formatted table: pip install tabulate]")
        for r in results:
            print(f"\n  {r['label']}: {r['total_time_s']}s | {r['vram_delta_mb']}MB VRAM")
            print(f"    {r['transcript'][:120]}...")
        return

    rows = []
    for r in results:
        rows.append([
            r["label"],
            f"{r['load_time_s']}s",
            f"{r['transcribe_time_s']}s",
            f"{r['total_time_s']}s",
            f"{r['vram_delta_mb']:.0f}MB",
            r["device"],
            r["transcript"][:80] + ("..." if len(r["transcript"]) > 80 else ""),
        ])

    headers = ["Preset", "Load", "Transcribe", "Total", "VRAM", "Device", "Transcript (first 80 chars)"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper pipeline presets on a call recording"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to audio file")
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()) + ["all"],
        default="all",
        help="Which preset to run (default: all)",
    )
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--initial-prompt",
        default="CarPlanet car dealership. Agent speaking with customer.",
        help="Domain prompt to prime the model",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/benchmark_results.json",
        help="Where to save results JSON (default: data/benchmark_results.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    download_root = os.path.join(PROJECT_ROOT, "models", "faster-whisper")

    presets_to_run = list(PRESETS.keys()) if args.preset == "all" else [args.preset]

    device_label = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"\nGPU: {gpu_name} ({vram_total:.0f}MB VRAM)")
    else:
        print("\nNo CUDA GPU detected — running on CPU")

    print(f"Audio: {args.input}")
    print(f"Presets to run: {', '.join(presets_to_run)}")
    print(f"Language: {args.language}")

    results = []
    for preset_key in presets_to_run:
        cfg = PRESETS[preset_key]
        print(f"\n{'='*60}")
        print(f"  PRESET: {cfg['label']}")
        print(f"  {cfg['description']}")
        print(f"{'='*60}")
        try:
            result = run_preset(
                preset_key=preset_key,
                audio_path=args.input,
                language=args.language,
                initial_prompt=args.initial_prompt,
                download_root=download_root,
            )
            results.append(result)
            print(f"\n  Transcript:\n    {result['transcript'][:200]}")
        except Exception as e:
            print(f"\n  ERROR running preset '{preset_key}': {e}")
            results.append({"preset": preset_key, "error": str(e)})

    print(f"\n\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print_results_table([r for r in results if "error" not in r])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
