"""Quick transcription-only test (no diarization) for all key models.

Usage (from call_processor/):
    python test_all_no_diar.py
"""
import sys, os, time, traceback, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3.5",
    "parakeet-tdt-0.6b-v3",
]

AUDIO = "data/raw_calls/anchor_test.mp3"
SECONDS = 60


def prepare_wav(audio_path: str, max_seconds: int) -> str:
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-t", str(max_seconds),
         "-ac", "1", "-ar", "16000", wav],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[:300])
    return wav


def test_model(name, wav_path):
    print(f"\n{'='*60}\n  Model: {name}\n{'='*60}")
    try:
        from src.transcribers import get_transcriber
        t = get_transcriber(name, device="cuda")
        t0 = time.time()
        t.load()
        print(f"  Load: {time.time()-t0:.1f}s")
        t0 = time.time()
        segs = t.transcribe(wav_path)
        print(f"  Transcribe: {time.time()-t0:.1f}s — {len(segs)} segments")
        if segs:
            print(f"  First: {segs[0].get('text','')[:80]!r}")
            print(f"  Last:  {segs[-1].get('text','')[:80]!r}")
        t.unload()
        print(f"  PASS")
        return True
    except Exception:
        traceback.print_exc()
        print(f"  FAIL")
        return False


def main():
    if not os.path.isfile(AUDIO):
        print(f"Audio not found: {AUDIO}")
        sys.exit(1)
    print(f"Audio: {AUDIO}  ({SECONDS}s clip)")
    wav = prepare_wav(AUDIO, SECONDS)
    results = {}
    try:
        for m in MODELS:
            results[m] = test_model(m, wav)
    finally:
        try:
            os.unlink(wav)
        except Exception:
            pass
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for m, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {m}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
