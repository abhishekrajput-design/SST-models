"""
One-shot downloader for all 5 ASR models + neural enhancement model.

Run:
    python download_models.py                # all
    python download_models.py --only cohere  # one model
    python download_models.py --skip vibevoice  # skip the 18 GB model

All models go under ./models/ (organized by source).
"""
import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Load .env so HF_TOKEN is available for gated models
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

HF_CACHE = os.path.join(PROJECT_ROOT, "models", "hf")
FW_CACHE = os.path.join(PROJECT_ROOT, "models", "faster-whisper")
NEMO_CACHE = os.path.join(PROJECT_ROOT, "models", "nemo")
SB_CACHE = os.path.join(PROJECT_ROOT, "models", "sepformer-dns4")

# (key, hf_id_or_action, approx_size_gb, downloader)
MODELS = [
    ("whisper-large-v3-turbo", "Systran/faster-whisper-large-v3-turbo", 1.6, "faster_whisper"),
    ("cohere-transcribe",      "CohereLabs/cohere-transcribe-03-2026",  3.0, "hf_snapshot"),
    ("parakeet-tdt-v3",        "nvidia/parakeet-tdt-0.6b-v3",           1.2, "hf_snapshot"),
    ("qwen3-asr-1.7b",         "Qwen/Qwen3-ASR-1.7B",                   3.4, "hf_snapshot"),
    ("vibevoice-asr",          "microsoft/VibeVoice-ASR",              18.0, "hf_snapshot"),
    ("sepformer-dns4",         "speechbrain/sepformer-dns4-16k-enhancement", 0.25, "hf_snapshot"),
]


def fmt_size(bytes_n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_n < 1024:
            return f"{bytes_n:.1f} {unit}"
        bytes_n /= 1024
    return f"{bytes_n:.1f} TB"


def dir_size(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def download_faster_whisper(model_id: str) -> str:
    """faster-whisper triggers download by instantiating WhisperModel."""
    from faster_whisper import WhisperModel
    short = model_id.split("/")[-1].replace("faster-whisper-", "")
    print(f"  Loading {short} into faster-whisper cache...")
    m = WhisperModel(short, device="cpu", compute_type="int8", download_root=FW_CACHE)
    del m
    return FW_CACHE


def download_hf(repo_id: str) -> str:
    from huggingface_hub import snapshot_download
    target = os.path.join(HF_CACHE, repo_id.replace("/", "__"))
    os.makedirs(target, exist_ok=True)
    print(f"  → {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=target, local_dir_use_symlinks=False,
                      token=os.environ.get("HF_TOKEN"))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Download only these (e.g. --only cohere parakeet-tdt-v3)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip these (e.g. --skip vibevoice)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ASR Model Downloader")
    print("=" * 60)

    if not os.environ.get("HF_TOKEN"):
        print("  WARN: HF_TOKEN not set — gated models (e.g. pyannote) will fail.")

    todo = []
    for key, hf_id, size_gb, kind in MODELS:
        if args.only and key not in args.only:
            continue
        if key in args.skip:
            print(f"  [skip] {key} ({size_gb} GB)")
            continue
        todo.append((key, hf_id, size_gb, kind))

    total_planned = sum(s for _, _, s, _ in todo)
    print(f"\n  Planned: {len(todo)} models, ~{total_planned:.1f} GB")
    print()

    results = []
    for key, hf_id, size_gb, kind in todo:
        print(f"━━━ {key}  ({hf_id})  ~{size_gb} GB ━━━")
        t0 = time.time()
        try:
            if kind == "faster_whisper":
                path = download_faster_whisper(hf_id)
            elif kind == "hf_snapshot":
                path = download_hf(hf_id)
            else:
                path = "?"
            elapsed = time.time() - t0
            actual_bytes = dir_size(path)
            print(f"  ✓ done in {elapsed:.0f}s · disk: {fmt_size(actual_bytes)}\n")
            results.append((key, "ok", actual_bytes, elapsed))
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            results.append((key, f"failed: {e}", 0, time.time() - t0))

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_disk = 0
    for key, status, bytes_n, elapsed in results:
        flag = "✓" if status == "ok" else "✗"
        print(f"  {flag} {key:30s}  {fmt_size(bytes_n):>10s}  {elapsed:5.0f}s  {status}")
        total_disk += bytes_n
    print(f"\n  Total on disk: {fmt_size(total_disk)}")


if __name__ == "__main__":
    main()
