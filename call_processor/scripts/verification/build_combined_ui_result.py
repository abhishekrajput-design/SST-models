"""
build_combined_ui_result.py
Transcribes + diarizes the combined multi-speaker test WAV and writes a
result.json that the UI can load directly (no re-enhancement needed).

Source : data/processed/enhanced_combined/norm_enhanced_combined.wav
Output : data/processed/combined_3agents__parakeet-tdt-0.6b-v3/result.json
"""
from __future__ import annotations
import gc, json, os, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(str(SCRIPT_DIR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

NORM_WAV   = str(SCRIPT_DIR / "data" / "processed" / "combined_speaker_test" / "combined.wav")
OUT_DIR    = str(SCRIPT_DIR / "data" / "processed" / "combined_3agents__parakeet-tdt-0.6b-v3")
AUDIO_SRC  = str(SCRIPT_DIR / "data" / "processed" / "combined_speaker_test" / "combined.wav")

FFMPEG = (
    r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

_TRANSCRIBE = r"""
import gc, json, os, sys
sys.path.insert(0, r"{script_dir}")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from src.transcribers import get_transcriber
tr = get_transcriber("parakeet-tdt-0.6b-v3", device="cuda")
tr.load()
segs = tr.transcribe(sys.argv[1], language="en")
tr.unload(); gc.collect()
print(json.dumps(segs, ensure_ascii=False))
sys.stdout.flush()
"""

_DIARIZE = r"""
import json, logging, os, sys
sys.path.insert(0, r"{script_dir}")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
from src.diar_multi import diarize_multi
with open(sys.argv[2], encoding="utf-8") as f:
    segs = json.load(f)
out = diarize_multi(segs, sys.argv[1], threshold=0.35, force_cpu=False)
print(json.dumps(out, ensure_ascii=False, default=str))
sys.stdout.flush()
"""


def _write_tmp(body: str) -> str:
    body = body.replace("{script_dir}", str(SCRIPT_DIR).replace("\\", "\\\\"))
    fd, path = tempfile.mkstemp(suffix=".py", dir=str(SCRIPT_DIR))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    # ── Verify source WAV ─────────────────────────────────────────────────────
    if not os.path.exists(NORM_WAV):
        print(f"[ERROR] combined WAV not found: {NORM_WAV}")
        print("  Run test_combined_speakers.py first to build it.")
        return

    import soundfile as sf
    dur = sf.info(NORM_WAV).duration
    print(f"[1/2] Transcribing {dur:.0f}s WAV with Parakeet TDT v3 ...")

    script = _write_tmp(_TRANSCRIBE)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", script, NORM_WAV],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600, cwd=str(SCRIPT_DIR))
    os.unlink(script)
    if r.returncode != 0:
        print(f"[ERROR] Transcription failed:\n{r.stderr[-800:]}")
        return
    # Extract JSON — last line starting with "["
    raw = next((l for l in reversed(r.stdout.splitlines()) if l.strip().startswith("[")), "[]")
    segments = json.loads(raw)
    print(f"    {len(segments)} segments in {time.time()-t0:.0f}s")

    seg_fd, seg_json = tempfile.mkstemp(suffix=".json")
    with os.fdopen(seg_fd, "w", encoding="utf-8") as f:
        json.dump(segments, f)

    print(f"\n[2/2] Running diar_multi on all {len(segments)} segments ...")
    script = _write_tmp(_DIARIZE)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", script, NORM_WAV, seg_json],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600, cwd=str(SCRIPT_DIR))
    os.unlink(script)
    os.unlink(seg_json)
    if r.returncode != 0:
        print(f"[ERROR] Diarization failed:\n{r.stderr[-800:]}")
        return

    raw_d = next((l for l in reversed(r.stdout.splitlines()) if l.strip().startswith("{")), "{}")
    diar = json.loads(raw_d)
    segs_out = diar.get("segments", segments)
    agent_name = diar.get("agent_name", "Unknown Agent")
    agent_sim  = diar.get("agent_similarity", 0.0)
    per_spk    = diar.get("per_speaker", {})
    print(f"    Done in {time.time()-t0:.0f}s")
    print(f"    Primary agent : {agent_name}  (avg cosine {agent_sim:.3f})")
    print(f"    Speakers      : {list(per_spk.keys())}")

    # ── Build transcription_json (HH:MM:SS.mmm format) ───────────────────────
    def _fmt(s: float) -> str:
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}"

    transcription_json = []
    for seg in segs_out:
        transcription_json.append({
            "start":              _fmt(float(seg["start"])),
            "end":                _fmt(float(seg["end"])),
            "speaker":            seg.get("speaker", "SPEAKER_00"),
            "phrase":             seg.get("text", ""),
            "avg_score":          seg.get("confidence", 0.0),
            "identified_speaker": seg.get("identified_speaker", "CUSTOMER"),
            "display_speaker":    seg.get("display_speaker", "Customer"),
            "agent_name":         seg.get("agent_name"),
        })

    # ── speaker_stats for UI header bar ──────────────────────────────────────
    agent_time = sum(float(s["end"]) - float(s["start"])
                     for s in segs_out if s.get("identified_speaker") == "AGENT")
    cust_time  = sum(float(s["end"]) - float(s["start"])
                     for s in segs_out if s.get("identified_speaker") != "AGENT")
    total_time = agent_time + cust_time or 1.0
    speaker_stats = {
        "agent_time_s":    round(agent_time, 1),
        "customer_time_s": round(cust_time, 1),
        "agent_pct":       round(100 * agent_time / total_time),
        "customer_pct":    round(100 * cust_time  / total_time),
        "agent_turns":     sum(1 for s in segs_out if s.get("identified_speaker") == "AGENT"),
        "customer_turns":  sum(1 for s in segs_out if s.get("identified_speaker") != "AGENT"),
        "per_speaker":     per_spk,
    }

    # ── Build trimmed_audio.mp3 from combined.wav ─────────────────────────────
    trimmed_mp3 = os.path.join(OUT_DIR, "trimmed_audio.mp3")
    subprocess.run([FFMPEG, "-y", "-i", AUDIO_SRC, "-b:a", "128k", trimmed_mp3],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    result = {
        "audio_file":              AUDIO_SRC.replace("\\", "/"),
        "trimmed_audio_file":      trimmed_mp3.replace("\\", "/"),
        "model":                   "parakeet-tdt-0.6b-v3",
        "processed_at":            datetime.utcnow().isoformat() + "Z",
        "processing_time_seconds": round(time.time() - t_total, 1),
        "total_segments":          len(segs_out),
        "segments":                segs_out,
        "transcription_json":      transcription_json,
        "diarization":             "diar_multi_voiceprint",
        "speaker_stats":           speaker_stats,
        "identified_agent":        agent_name,
        "agent_similarity":        agent_sim,
        "note": "Synthetic test: Zak(0-40s) | Random(42-72s) | Omar(74-114s) | Random(116-146s) | Hussein(148-188s)",
        "roles_swapped":           False,
        "n_speakers_detected":     len(per_spk),
    }

    result_path = os.path.join(OUT_DIR, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n    Saved → {result_path}")
    print(f"    Total time: {time.time()-t_total:.0f}s")
    print(f"\n  Open http://localhost:8080 and click 'combined_3agents__parakeet-tdt-0.6b-v3'")


if __name__ == "__main__":
    main()
