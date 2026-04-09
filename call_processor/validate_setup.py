"""
Quick Validation Script — checks that all dependencies and models are accessible.

Usage:
    python validate_setup.py

This does NOT process any audio. It simply verifies:
  1. All Python packages are installed
  2. GPU is available and has enough VRAM
  3. HuggingFace token is set
  4. Project directories exist
  5. Agent embeddings exist (if enrolled)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def check(name: str, passed: bool, detail: str = ""):
    status = "✅" if passed else "❌"
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def main():
    print("=" * 60)
    print("  Call Processor — Setup Validation")
    print("=" * 60)
    all_ok = True

    # ── 1. Python version ──────────────────────────────────────
    print("\n[1/6] Python Version")
    py_ver = sys.version_info
    all_ok &= check(
        f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
        py_ver >= (3, 10),
        "Requires 3.10+" if py_ver < (3, 10) else "OK"
    )

    # ── 2. Core packages ──────────────────────────────────────
    print("\n[2/6] Required Packages")
    packages = {
        "torch": "PyTorch",
        "torchaudio": "TorchAudio",
        "pyannote.audio": "pyannote.audio (diarization)",
        "speechbrain": "SpeechBrain (speaker embeddings)",
        "faster_whisper": "faster-whisper (transcription)",
        "soundfile": "SoundFile (audio I/O)",
        "numpy": "NumPy",
    }
    for pkg, label in packages.items():
        try:
            mod = __import__(pkg.replace(".", "_") if "." in pkg else pkg)
            ver = getattr(mod, "__version__", "installed")
            all_ok &= check(label, True, f"v{ver}")
        except ImportError:
            all_ok &= check(label, False, "NOT INSTALLED — run: pip install -r requirements.txt")

    # ── 3. GPU / CUDA ──────────────────────────────────────────
    print("\n[3/6] GPU / CUDA")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            all_ok &= check("CUDA available", True, f"{name} ({vram:.1f} GB VRAM)")
            all_ok &= check("VRAM >= 4GB", vram >= 4.0, f"{vram:.1f} GB" if vram < 4.0 else "OK")
        else:
            check("CUDA available", False, "Will use CPU (slower). Install CUDA toolkit for GPU.")
    except Exception as e:
        check("CUDA check", False, str(e))

    # ── 4. HuggingFace token ───────────────────────────────────
    print("\n[4/6] HuggingFace Token")
    hf_token = os.environ.get("HF_TOKEN", "")
    all_ok &= check(
        "HF_TOKEN environment variable",
        bool(hf_token),
        f"Set ({hf_token[:8]}...)" if hf_token else "NOT SET — required for pyannote.audio"
    )

    # ── 5. Directory structure ─────────────────────────────────
    print("\n[5/6] Project Directories")
    dirs = {
        "data/raw_calls": "Place call recordings here",
        "data/agent_samples": "Agent voice sample directories",
        "data/processed": "Pipeline output (auto-created)",
        "embeddings": "Agent embedding database",
        "models": "Cached model weights",
    }
    for rel_path, desc in dirs.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        exists = os.path.isdir(full_path)
        all_ok &= check(rel_path, exists, desc if exists else f"MISSING — mkdir {rel_path}")

    # ── 6. Agent enrollment ────────────────────────────────────
    print("\n[6/6] Agent Enrollment")
    emb_path = os.path.join(PROJECT_ROOT, "embeddings", "agent_embeddings.pkl")
    if os.path.exists(emb_path):
        import pickle
        with open(emb_path, "rb") as f:
            agents = pickle.load(f)
        check("Agent embeddings", True, f"{len(agents)} agents enrolled: {', '.join(agents.keys())}")
    else:
        check(
            "Agent embeddings",
            False,
            "Not enrolled yet — run: python enroll_agents.py"
        )

    # Check agent sample directories
    samples_dir = os.path.join(PROJECT_ROOT, "data", "agent_samples")
    if os.path.isdir(samples_dir):
        agent_dirs = [
            d for d in os.listdir(samples_dir)
            if os.path.isdir(os.path.join(samples_dir, d))
        ]
        if agent_dirs:
            for d in agent_dirs:
                audio_count = len([
                    f for f in os.listdir(os.path.join(samples_dir, d))
                    if f.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
                ])
                check(f"  {d}", audio_count > 0, f"{audio_count} audio files")
        else:
            check("Agent sample directories", False, "No agent_NAME/ directories found")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — Ready to process calls!")
        print("  Run: python main.py --input path/to/call.wav")
    else:
        print("  ⚠️  Some checks failed — fix the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
