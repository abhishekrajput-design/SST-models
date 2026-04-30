"""
Quick batch test: run transcription on enhanced audio files with specific models.
Writes result.json to data/processed/{base}__{model}/ so the UI picks them up.
"""
import os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env (same as ui.py does)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

TESTS = [
    # vrcta2 (high quality, 51 kbps) - deepgram only
    ("data/raw_calls/enhanced_audio_04_12_2026_12_28_59_vrcta2.mp3", "deepgram-nova-3"),
]

import subprocess

def normalize_wav(audio_path: str, norm_wav: str):
    # Require at least 10MB for a ~30-min recording at 16kHz mono 16-bit
    min_size = 10_000_000
    if os.path.exists(norm_wav) and os.path.getsize(norm_wav) > min_size:
        print(f"  [skip norm] {os.path.basename(norm_wav)} already exists ({os.path.getsize(norm_wav)//1024//1024}MB)")
        return
    print(f"  [norm] -> {os.path.basename(norm_wav)}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path,
         "-ar", "16000", "-ac", "1",
         "-af", "loudnorm=I=-23:TP=-1:LRA=7",
         norm_wav],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

results_summary = []

for audio_path, model in TESTS:
    base     = os.path.splitext(os.path.basename(audio_path))[0]
    dir_name = f"{base}__{model}"
    out_dir  = os.path.join("data", "processed", dir_name)
    result_path = os.path.join(out_dir, "result.json")

    if os.path.exists(result_path):
        try:
            d = json.load(open(result_path))
            segs = len(d.get("segments", []))
            if segs > 0:
                print(f"[SKIP] {dir_name}: already done ({segs} segs)")
                results_summary.append((dir_name, model, "skip", segs, 0))
                continue
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"TEST: {dir_name}")
    print(f"  audio: {audio_path}")
    print(f"  model: {model}")
    os.makedirs(out_dir, exist_ok=True)

    # Normalize audio once per base file
    norm_dir = os.path.join("data", "processed", base)
    os.makedirs(norm_dir, exist_ok=True)
    norm_wav = os.path.join(norm_dir, f"norm_{base}.wav")
    try:
        normalize_wav(audio_path, norm_wav)
    except Exception as e:
        print(f"  [ERROR] normalization failed: {e}")
        results_summary.append((dir_name, model, "error", 0, 0))
        continue

    from src.transcribers import get_transcriber
    print(f"  [load] loading {model}...")
    t_load = time.time()
    try:
        transcriber = get_transcriber(model, device="cuda")
        transcriber.load()
        load_time = round(time.time() - t_load, 1)
        print(f"  [load] done in {load_time}s")
    except Exception as e:
        print(f"  [ERROR] load failed: {e}")
        results_summary.append((dir_name, model, "error-load", 0, 0))
        continue

    print(f"  [transcribe] running...")
    t0 = time.time()
    try:
        segments = transcriber.transcribe(norm_wav, language="en")
        elapsed = round(time.time() - t0, 2)
    except Exception as e:
        print(f"  [ERROR] transcribe failed: {e}")
        transcriber.unload()
        results_summary.append((dir_name, model, "error-transcribe", 0, 0))
        continue

    transcriber.unload()

    result = {
        "audio_file":            audio_path.replace("\\", "/"),
        "model":                 model,
        "processed_at":          datetime.now().isoformat(),
        "processing_time_seconds": elapsed,
        "total_segments":        len(segments),
        "segments":              segments,
        "note":                  f"Batch test via run_batch_tests.py",
        "enhancements": {"ffmpeg": audio_path.replace("\\", "/")},
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  [done] {len(segments)} segs in {elapsed}s  -> {result_path}")
    results_summary.append((dir_name, model, "done", len(segments), elapsed))

    # Write transcript.txt
    txt_path = os.path.join(out_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in segments:
            start = s.get("start", 0)
            end   = s.get("end", 0)
            text  = s.get("text", "")
            spk   = s.get("speaker") or s.get("identified_speaker") or ""
            f.write(f"[{start:.1f}s - {end:.1f}s] {spk + ': ' if spk else ''}{text}\n")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Test':<55} {'Status':<12} {'Segs':>6} {'Time':>8}")
print("-"*85)
for (name, model, status, segs, t) in results_summary:
    short = name[:52]
    print(f"{short:<55} {status:<12} {segs:>6} {str(t)+'s':>8}")
