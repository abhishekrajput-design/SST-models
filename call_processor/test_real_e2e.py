"""
test_real_e2e.py — End-to-end speaker ID test on real call recordings.

Tests Zak and Hussein agent identification using:
  - 512-dim CAM++ voiceprints enrolled from desk recordings
  - Parakeet TDT v3 transcription
  - CAM++ diarization
  - Multi-agent identify_agent_name

Usage:
  python call_processor/test_real_e2e.py
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(str(SCRIPT_DIR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FFMPEG = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

# Real call recordings to test
CALLS = [
    {
        "label":    "Zak",
        # Held-out: NOT used in enrollment (enrollment used 69e3b205 and 69e4a441)
        "audio":    str(SCRIPT_DIR / "data" / "audiofy" / "zak_raissi_barnet" / "audio"
                        / "call_69e3afc81bbc87d03ab29ae6.mp3"),
        "expected": "zak",
    },
]

OUT_DIR = SCRIPT_DIR / "data" / "processed" / "real_e2e_test"


def normalize(src: str, dest: str) -> None:
    subprocess.run(
        [FFMPEG, "-y", "-i", src, "-ac", "1", "-ar", "16000", dest],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=120, check=True,
    )


def clear_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def run_call(call: dict, norm_wav: str) -> dict:
    label    = call["label"]
    expected = call["expected"]
    t_start  = time.time()

    print(f"\n{'='*60}")
    print(f"  Testing: {label}")
    print(f"  Audio: {Path(call['audio']).name}")
    print(f"{'='*60}\n")

    # 1 — Normalize
    print(f"[1/3] Normalizing to 16kHz mono ...", flush=True)
    normalize(call["audio"], norm_wav)
    import soundfile as sf
    info = sf.info(norm_wav)
    print(f"      {info.duration:.0f}s @ {info.samplerate}Hz", flush=True)

    # 2 — Transcribe
    print(f"\n[2/3] Parakeet transcription ...", flush=True)
    t0 = time.time()
    from src.transcribers import get_transcriber
    tr = get_transcriber("parakeet-tdt-0.6b-v3", device="cuda")
    tr.load()
    segments = tr.transcribe(norm_wav, language="en")
    tr.unload()
    clear_gpu()
    print(f"      {len(segments)} segments in {time.time()-t0:.0f}s", flush=True)

    # 3 — Diarize
    print(f"\n[3/3] CAM++ diarization + agent identification ...", flush=True)
    t0 = time.time()

    from src.diar_campp import diarize_segments_campp
    segments = diarize_segments_campp(segments, norm_wav, num_speakers=2)

    spk_time: dict = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        spk_time[spk] = spk_time.get(spk, 0.0) + (float(seg["end"]) - float(seg["start"]))

    print(f"      Speakers: {spk_time}", flush=True)

    from src.speaker_role import identify_agent_name
    agent_spk, agent_name, agent_sim = identify_agent_name(segments, norm_wav, spk_time)

    total = time.time() - t_start
    passed = expected.lower() in agent_name.lower()

    print(f"\n{'='*60}")
    print(f"  Result: {label}")
    print(f"  Agent speaker : {agent_spk}")
    print(f"  Agent name    : {agent_name}")
    print(f"  Cosine sim    : {agent_sim:.4f}")
    print(f"  Total time    : {total:.0f}s")
    if passed:
        print(f"  [PASS] Correctly identified as {agent_name}")
    else:
        print(f"  [FAIL] Expected '{expected}' in name, got '{agent_name}'")
    print(f"{'='*60}")

    return {
        "label":      label,
        "agent_name": agent_name,
        "agent_sim":  round(agent_sim, 4),
        "agent_spk":  agent_spk,
        "passed":     passed,
        "time_s":     round(total, 1),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify voiceprints exist
    vp_dir = SCRIPT_DIR / "data" / "agent_voiceprints"
    import numpy as np
    for slug, name in [("zak_raissi_barnet", "Zak"), ("hussein_mohamed", "Hussein")]:
        vp = vp_dir / f"{slug}.npy"
        if vp.exists():
            arr = np.load(vp)
            print(f"[check] {name} voiceprint: {arr.shape}", flush=True)
        else:
            print(f"[WARN] {name} voiceprint missing: {vp}", flush=True)

    results = []
    for call in CALLS:
        if not Path(call["audio"]).exists():
            print(f"[SKIP] Audio not found: {call['audio']}", flush=True)
            continue
        fd, norm_wav = tempfile.mkstemp(suffix=f"_{call['label']}.wav", dir=str(OUT_DIR))
        os.close(fd)
        try:
            result = run_call(call, norm_wav)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {call['label']}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            try:
                os.unlink(norm_wav)
            except OSError:
                pass
        clear_gpu()

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['label']:12} → {r['agent_name']}  (sim={r['agent_sim']:.3f}, {r['time_s']}s)")

    out_file = OUT_DIR / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results → {out_file}")


if __name__ == "__main__":
    main()
