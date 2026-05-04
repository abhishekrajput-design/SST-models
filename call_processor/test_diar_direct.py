#!/usr/bin/env python3
"""
Direct test of diar_multi without UI - test multi-voiceprint matching.
"""
import json
import time
from src.diar_multi import diarize_multi
from src.process_audio import load_mp3_mono_16k

API_DATA_DIR = "data/audiofy/_dataset"

def main():
    print("[test] Direct diar_multi test\n")

    # Load API data
    with open(f"{API_DATA_DIR}/index.json") as f:
        api_data = json.load(f)

    # Pick a few calls to test
    test_calls = api_data[:5]

    results = []

    for i, call in enumerate(test_calls, 1):
        call_id = call.get("_id")
        agent_name = call.get("agent_name", "Unknown")
        audio_file = f"{API_DATA_DIR}/audio/{call_id}.mp3"

        print(f"[{i}] {agent_name:25s} ... ", end="", flush=True)

        try:
            # Load audio
            audio, sr = load_mp3_mono_16k(audio_file)
            print(f"{len(audio)/sr:.1f}s ", end="", flush=True)

            # Run diarization
            t0 = time.time()
            result = diarize_multi(None, audio, force_cpu=False, use_waveform=True)
            elapsed = time.time() - t0

            identified = (result.get("agent_name") or "").lower().replace(" ", "_")
            expected = (agent_name or "").lower().replace(" ", "_")

            is_match = identified == expected or expected in identified

            print(f"→ {identified:25s} ", end="")

            if is_match:
                print(f"OK ({elapsed:.1f}s)")
            else:
                print(f"WRONG ({elapsed:.1f}s)")

            results.append({
                "call_id": call_id,
                "agent_name": agent_name,
                "identified": identified,
                "correct": is_match,
                "time_s": elapsed,
                "n_segments": len(result.get("segments", [])),
                "backend_dim": result.get("matched_backend_dim"),
            })

        except Exception as e:
            print(f"ERROR: {str(e)[:60]}")

    # Summary
    print("\n" + "="*80)
    if results:
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        accuracy = correct / total * 100
        times = [r["time_s"] for r in results]
        avg_time = sum(times) / len(times)

        print(f"\n[DIAR_MULTI DIRECT TEST]")
        print(f"  Calls tested: {total}")
        print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"  Avg processing time: {avg_time:.1f}s")
        print(f"  Min/Max: {min(times):.1f}s / {max(times):.1f}s")

        # Per-call details
        print(f"\nDetails:")
        for r in results:
            status = "OK" if r["correct"] else "WRONG"
            print(f"  {r['agent_name']:25s} → {r['identified']:25s} {status:5s} {r['time_s']:6.1f}s ({r['n_segments']} segs, dim={r['backend_dim']})")

        # Save
        with open("diar_direct_results.json", "w") as f:
            json.dump({
                "accuracy_pct": accuracy,
                "correct": correct,
                "total": total,
                "avg_time_s": avg_time,
                "results": results
            }, f, indent=2)

        print(f"\nResults saved to diar_direct_results.json")

if __name__ == "__main__":
    main()
