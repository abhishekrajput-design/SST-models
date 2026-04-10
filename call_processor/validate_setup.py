"""
Quick validation script for the call processor environment.

Usage:
    python validate_setup.py
"""

import importlib
import os
import pickle
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "[OK]" if passed else "[FAIL]"
    message = f"  {status} {name}"
    if detail:
        message += f" -- {detail}"
    print(message)
    return passed


def main():
    print("=" * 60)
    print("  Call Processor - Setup Validation")
    print("=" * 60)
    all_ok = True

    print("\n[1/6] Python Version")
    py_ver = sys.version_info
    all_ok &= check(
        f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
        py_ver >= (3, 10),
        "Requires 3.10+" if py_ver < (3, 10) else "OK",
    )

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
    for module_name, label in packages.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            all_ok &= check(label, True, f"v{version}")
        except Exception as exc:
            all_ok &= check(label, False, str(exc))

    print("\n[3/6] GPU / CUDA")
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024 ** 3)
            all_ok &= check("CUDA available", True, f"{name} ({total_memory:.1f} GB VRAM)")
            all_ok &= check("VRAM >= 4GB", total_memory >= 4.0, "OK" if total_memory >= 4.0 else f"{total_memory:.1f} GB")
        else:
            all_ok &= check("CUDA available", False, "Will use CPU")
    except Exception as exc:
        all_ok &= check("CUDA check", False, str(exc))

    print("\n[4/6] HuggingFace Token")
    hf_token = os.environ.get("HF_TOKEN", "")
    all_ok &= check(
        "HF_TOKEN environment variable",
        bool(hf_token),
        f"Set ({hf_token[:8]}...)" if hf_token else "NOT SET - required for pyannote.audio diarization",
    )

    print("\n[5/6] Project Directories")
    dirs = {
        "data/raw_calls": "Place call recordings here",
        "data/agent_samples": "Agent voice sample directories",
        "data/processed": "Pipeline output",
        "embeddings": "Agent embedding database",
        "models": "Cached model weights",
    }
    for rel_path, desc in dirs.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        exists = os.path.isdir(full_path)
        all_ok &= check(rel_path, exists, desc if exists else f"Missing: {rel_path}")

    print("\n[6/6] Agent Enrollment")
    emb_path = os.path.join(PROJECT_ROOT, "embeddings", "agent_embeddings.pkl")
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as handle:
            agents = pickle.load(handle)
        all_ok &= check(
            "Agent embeddings",
            len(agents) > 0,
            f"{len(agents)} agents enrolled: {', '.join(sorted(agents.keys()))}",
        )
    else:
        all_ok &= check("Agent embeddings", False, "Not enrolled yet - run: python enroll_agents.py")

    samples_dir = os.path.join(PROJECT_ROOT, "data", "agent_samples")
    if os.path.isdir(samples_dir):
        agent_dirs = [
            name
            for name in sorted(os.listdir(samples_dir))
            if os.path.isdir(os.path.join(samples_dir, name))
        ]
        if agent_dirs:
            for name in agent_dirs:
                audio_count = len(
                    [
                        filename
                        for filename in os.listdir(os.path.join(samples_dir, name))
                        if filename.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
                    ]
                )
                check(name, audio_count > 0, f"{audio_count} audio files")
        else:
            check("Agent sample directories", False, "No agent directories found")

    print("\n" + "=" * 60)
    if all_ok:
        print("  [OK] All required checks passed.")
        print("  Run: python main.py --input path/to/call.wav")
    else:
        print("  [WARN] Some checks failed - fix the items above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
