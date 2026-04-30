"""
Test agent/customer identification on Agents-recoding/sd_agent_customer_audio.
Runs: FFmpeg-norm -> Parakeet transcribe -> diarize_multi.
Reports which enrolled agent was identified, similarity, and time/turn split.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.diar_multi import diarize_multi
from src.transcribers import get_transcriber

SD_DIR = ROOT.parent / "Agents-recoding" / "sd_agent_customer_audio"
OUT_DIR = ROOT / "data" / "processed" / "sd_id_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NORM_AF = (
    "aresample=44100,"
    "loudnorm=I=-16:TP=-1.5:LRA=11,"
    "dynaudnorm=p=0.9:m=100:s=5"
)


def make_norm_wav(src: Path, dst: Path) -> None:
    """Mono-only normalisation — same recipe as ui.py _make_norm_wav."""
    af = f"aformat=channel_layouts=mono,{NORM_AF}"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-af", af, str(dst)],
        check=True,
        capture_output=True,
        timeout=600,
    )


def test_one(audio_path: Path, model_name: str = "parakeet-tdt-0.6b-v3") -> dict:
    print(f"\n{'='*70}\n[TEST] {audio_path.name}\n{'='*70}")
    base = audio_path.stem
    norm_wav = OUT_DIR / f"norm_{base}.wav"

    if not norm_wav.exists():
        print(f"[1/3] FFmpeg-norm -> {norm_wav.name}")
        t0 = time.time()
        make_norm_wav(audio_path, norm_wav)
        print(f"      done in {time.time() - t0:.1f}s")
    else:
        print(f"[1/3] reusing {norm_wav.name}")

    print(f"[2/3] Transcribing with {model_name}")
    t0 = time.time()
    tr = get_transcriber(model_name, device="cuda")
    tr.load()
    segs = tr.transcribe(str(norm_wav), language="en")
    tr.unload()
    print(f"      {len(segs)} segments, {time.time() - t0:.1f}s")

    print(f"[3/3] diarize_multi (voiceprint matching against enrolled agents)")
    t0 = time.time()
    res = diarize_multi(segs, str(norm_wav), force_cpu=True)
    print(f"      done in {time.time() - t0:.1f}s")

    out_segs = res["segments"]
    agent_t = sum(float(s["end"]) - float(s["start"])
                  for s in out_segs if s.get("identified_speaker") == "AGENT")
    cust_t = sum(float(s["end"]) - float(s["start"])
                 for s in out_segs if s.get("identified_speaker") == "CUSTOMER")
    speakers = sorted({s.get("display_speaker", "?") for s in out_segs})
    n_segs_agent = sum(1 for s in out_segs if s.get("identified_speaker") == "AGENT")
    n_segs_cust = sum(1 for s in out_segs if s.get("identified_speaker") == "CUSTOMER")

    print(f"\n  AGENT IDENTIFIED: {res.get('agent_name', '?')}")
    print(f"  Agent similarity (top-30% mean): {res.get('agent_similarity', 0.0):.3f}")
    print(f"  n_speakers: {res.get('n_speakers', '?')}")
    print(f"  Speakers in output: {speakers}")
    print(f"  AGENT  : {agent_t:6.1f}s  ({n_segs_agent} segs)")
    print(f"  CUSTOMER: {cust_t:6.1f}s  ({n_segs_cust} segs)")

    summary = {
        "file": audio_path.name,
        "agent_name": res.get("agent_name"),
        "agent_similarity": float(res.get("agent_similarity", 0.0)),
        "n_speakers": res.get("n_speakers"),
        "match_counts": res.get("match_counts", {}),
        "speakers": speakers,
        "agent_time_s": round(agent_t, 1),
        "customer_time_s": round(cust_t, 1),
        "total_segments": len(out_segs),
    }
    with open(OUT_DIR / f"{base}__summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    files = sorted(SD_DIR.glob("*.mp3"))
    if not files:
        print(f"No MP3 files in {SD_DIR}")
        return
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        files = [f for f in files if target in f.name]
        if not files:
            print(f"No files match '{target}'")
            return

    print(f"Found {len(files)} test file(s):")
    for f in files:
        print(f"  - {f.name}")

    all_summaries = []
    for f in files:
        try:
            s = test_one(f)
            all_summaries.append(s)
        except Exception as e:
            import traceback
            print(f"[ERROR] {f.name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for s in all_summaries:
        print(f"  {s['file']:50s}  agent={s['agent_name']:25s}  sim={s['agent_similarity']:.3f}  "
              f"agent={s['agent_time_s']}s  cust={s['customer_time_s']}s")

    with open(OUT_DIR / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
