"""
Standalone transcription job — hardcoded paths, no stdin dependency.
Run: python transcribe_job.py
"""
import sys, os, time, json, gc

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

NORM_WAV    = r"C:\Users\abhis\Desktop\SST-models\call_processor\data\processed\audio_04_12_2026_12_28_59_vrcta2\norm_audio_04_12_2026_12_28_59_vrcta2.wav"
OUT_DIR     = r"C:\Users\abhis\Desktop\SST-models\call_processor\data\processed\audio_04_12_2026_12_28_59_vrcta2"
MODEL_DIR   = r"C:\Users\abhis\Desktop\SST-models\call_processor\models\faster-whisper"
INITIAL     = "CarPlanet car dealership. Agent speaking with customer."
LOG_PATH    = os.path.join(OUT_DIR, "transcribe_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log(f"[{time.strftime('%H:%M:%S')}] Starting transcription job")
log(f"Input: {NORM_WAV}")
log(f"Output dir: {OUT_DIR}")

if not os.path.exists(NORM_WAV):
    log(f"ERROR: norm wav not found!")
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

import torch
from faster_whisper import WhisperModel

device  = "cuda" if torch.cuda.is_available() else "cpu"
compute = "float16" if device == "cuda" else "float32"
log(f"Device: {device} | Compute: {compute}")

log("Loading Whisper large-v3...")
t_load = time.time()
model = WhisperModel("large-v3", device=device, compute_type=compute, download_root=MODEL_DIR)
log(f"Model loaded in {time.time()-t_load:.1f}s")

log("Transcribing (VAD-filtered, 30-min recording)...")
t0 = time.time()
segs_iter, info = model.transcribe(
    NORM_WAV,
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
    condition_on_previous_text=True,
    initial_prompt=INITIAL,
    word_timestamps=True,
    temperature=0,
    no_speech_threshold=0.6,
)

segments = [{"start": round(s.start,2), "end": round(s.end,2), "text": s.text.strip()} for s in segs_iter]
elapsed = time.time() - t0
log(f"Transcribed {len(segments)} segments in {elapsed:.1f}s")
log(f"Language: {info.language} ({info.language_probability:.0%})")

del model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

def fmt(sec):
    m,s = divmod(int(sec),60)
    h,m2 = divmod(m,60)
    return f"{h:02d}:{m2:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

json_path = os.path.join(OUT_DIR, "result.json")
txt_path  = os.path.join(OUT_DIR, "transcript.txt")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"audio_file": NORM_WAV, "total_segments": len(segments), "segments": segments}, f, indent=2, ensure_ascii=False)
log(f"Saved JSON: {json_path} ({os.path.getsize(json_path)} bytes)")

lines = [f"TRANSCRIPT — audio_04_12_2026_12_28_59_vrcta2.mp3", "="*70, ""]
for seg in segments:
    lines.append(f"[{fmt(seg['start'])} → {fmt(seg['end'])}]  {seg['text']}")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log(f"Saved transcript: {txt_path} ({os.path.getsize(txt_path)} bytes)")

log(f"\n--- PREVIEW (first 30 segments) ---")
for seg in segments[:30]:
    log(f"  [{fmt(seg['start'])} → {fmt(seg['end'])}]  {seg['text']}")
if len(segments) > 30:
    log(f"  ... and {len(segments)-30} more segments")

log(f"\n[DONE] Total: {(time.time()-t0):.0f}s")
