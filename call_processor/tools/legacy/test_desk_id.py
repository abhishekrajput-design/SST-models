"""
test_desk_id.py — Test agent identification on 1 desk recording each for Zak and Hussein.

Desk recordings are 30-min ambient captures. We trim to first 3 minutes for speed,
run the full pipeline, and check if the correct agent is identified.
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

CACHE = SCRIPT_DIR / "data" / "desk_recordings_cache"

TESTS = [
    {
        "label":    "Zak",
        "src":      str(CACHE / "zak_raissi_barnet_00_audio_03_15_2026_10_08_04_qbpe8e.mp3"),
        "expected": "zak",
    },
    {
        "label":    "Hussein",
        "src":      str(CACHE / "hussein_mohamed_00_audio_04_05_2026_20_37_05_z2l7zh.mp3"),
        "expected": "hussein",
    },
]

TRIM_SECONDS = 180   # test on first 3 minutes (desk recordings are 30 min each)
OUT_DIR = SCRIPT_DIR / "data" / "processed" / "desk_id_test"


def normalize_and_trim(src: str, dest: str, trim_s: int = TRIM_SECONDS) -> float:
    """FFmpeg: trim to first N seconds + normalize to 16kHz mono."""
    subprocess.run(
        [FFMPEG, "-y", "-i", src,
         "-t", str(trim_s),
         "-ac", "1", "-ar", "16000", dest],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=60, check=True,
    )
    import soundfile as sf
    return sf.info(dest).duration


def clear_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def run_test(t: dict, norm_wav: str) -> dict:
    label    = t["label"]
    expected = t["expected"]
    t_start  = time.time()

    print(f"\n{'='*60}")
    print(f"  DESK RECORDING TEST: {label}")
    print(f"  File: {Path(t['src']).name}")
    print(f"  Trim: first {TRIM_SECONDS}s")
    print(f"{'='*60}\n")

    # 1 — Normalize + trim
    print("[1/3] Normalize + trim to 16kHz mono ...", flush=True)
    dur = normalize_and_trim(t["src"], norm_wav)
    print(f"      {dur:.0f}s ready", flush=True)

    # 2 — Transcribe
    print("\n[2/3] Parakeet transcription ...", flush=True)
    t0 = time.time()
    from src.transcribers import get_transcriber
    tr = get_transcriber("parakeet-tdt-0.6b-v3", device="cuda")
    tr.load()
    segments = tr.transcribe(norm_wav, language="en")
    tr.unload()
    clear_gpu()
    print(f"      {len(segments)} segments in {time.time()-t0:.0f}s", flush=True)

    if not segments:
        print("  [WARN] No speech detected in this recording segment", flush=True)

    # 3 — Diarize + identify
    print("\n[3/3] CAM++ diarization + agent identification ...", flush=True)
    t0 = time.time()

    from src.diar_campp import diarize_segments_campp
    segments = diarize_segments_campp(segments, norm_wav, num_speakers=2)

    spk_time: dict = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        spk_time[spk] = spk_time.get(spk, 0.0) + float(seg["end"]) - float(seg["start"])

    print(f"      Speakers: { {k: round(v,1) for k,v in spk_time.items()} }", flush=True)

    from src.speaker_role import identify_agent_name
    agent_spk, agent_name, agent_sim = identify_agent_name(segments, norm_wav, spk_time)

    total = time.time() - t_start
    passed = expected.lower() in agent_name.lower()

    print(f"\n{'='*60}")
    print(f"  Result  : {label}")
    print(f"  Identified as : {agent_name}")
    print(f"  Speaker       : {agent_spk}")
    print(f"  Cosine sim    : {agent_sim:.4f}")
    print(f"  Time          : {total:.0f}s")
    print(f"  {'[PASS]' if passed else '[FAIL]'} Expected '{expected}' in name")
    print(f"{'='*60}")

    # Print first 8 transcript lines
    print("\n  First 8 segments:")
    for seg in segments[:8]:
        spk  = seg.get("display_speaker") or seg.get("speaker", "?")
        txt  = seg.get("text", "").strip()[:70]
        ts   = f"{float(seg['start']):.1f}s"
        print(f"    [{ts:>6}] {spk}: {txt}")

    return {
        "label":      label,
        "agent_name": agent_name,
        "agent_sim":  round(agent_sim, 4),
        "passed":     passed,
        "n_segments": len(segments),
        "time_s":     round(total, 1),
        "spk_time":   {k: round(v, 1) for k, v in spk_time.items()},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import numpy as np
    vp_dir = SCRIPT_DIR / "data" / "agent_voiceprints"
    for slug, name in [("zak_raissi_barnet", "Zak"), ("hussein_mohamed", "Hussein"),
                        ("mohammed_hussein_al_khwildi", "Hussein (API slug)")]:
        vp = vp_dir / f"{slug}.npy"
        if vp.exists():
            print(f"[check] {name}: {np.load(vp).shape}", flush=True)

    results = []
    for t in TESTS:
        if not Path(t["src"]).exists():
            print(f"[SKIP] File not found: {t['src']}", flush=True)
            continue

        fd, norm_wav = tempfile.mkstemp(suffix=f"_{t['label']}.wav", dir=str(OUT_DIR))
        os.close(fd)
        try:
            result = run_test(t, norm_wav)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {t['label']}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            try:
                os.unlink(norm_wav)
            except OSError:
                pass
        clear_gpu()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['label']:10} → {r['agent_name']}  "
              f"(sim={r['agent_sim']:.3f}, {r['time_s']}s, {r['n_segments']} segs)")

    out = OUT_DIR / "results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
