"""
Diagnostic script — run directly on the server to get full tracebacks.

Usage (from call_processor/ directory):
    python test_models_local.py
    python test_models_local.py --model qwen3-asr-1.7b
    python test_models_local.py --model vibevoice-asr
"""
import sys, os, argparse, traceback, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = ["qwen3-asr-1.7b", "vibevoice-asr"]

def test_model(name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")
    try:
        from src.transcribers import get_transcriber
        t = get_transcriber(name)
        t_load = time.time()
        print(f"  Loading model...")
        t.load()
        print(f"  Load OK in {time.time()-t_load:.1f}s  backend={getattr(t, '_backend', 'n/a')}")

        # Find a test audio file
        test_audio = None
        for candidate in [
            os.path.join(os.path.dirname(__file__), "data", "raw_calls"),
        ]:
            if os.path.isdir(candidate):
                for f in sorted(os.listdir(candidate)):
                    if f.endswith((".mp3", ".wav")):
                        test_audio = os.path.join(candidate, f)
                        break
            if test_audio:
                break

        if test_audio:
            print(f"  Transcribing: {os.path.basename(test_audio)}")
            t_inf = time.time()
            segs = t.transcribe(test_audio)
            print(f"  Transcribe OK in {time.time()-t_inf:.1f}s  segments={len(segs)}")
            if segs:
                print(f"  First segment: {segs[0]}")
        else:
            print("  No test audio found — skipping transcription test")

        t.unload()
        print(f"  PASS")
        return True

    except Exception:
        print(f"\n  !! FULL TRACEBACK:")
        traceback.print_exc()
        print(f"\n  FAIL: {name}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Test one model only")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    results = {}
    for m in models:
        results[m] = test_model(m)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for m, ok in results.items():
        print(f"  {'OK' if ok else '!!'} {m}")
    print(f"{'='*60}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
