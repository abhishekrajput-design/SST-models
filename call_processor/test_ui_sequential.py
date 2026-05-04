#!/usr/bin/env python3
"""
Sequential UI test - upload calls one at a time, wait for completion.
"""
import json
import time
import os
import requests

UI_URL = "http://localhost:8080"
API_DATA_DIR = "data/audiofy/_dataset"

def main():
    print("[test-ui] Sequential upload test\n")

    # Load API data
    with open(f"{API_DATA_DIR}/index.json") as f:
        api_data = json.load(f)

    # Load trained IDs
    with open("data/agent_voiceprints/agents.json") as f:
        agents = json.load(f)

    trained_ids = set()
    for agent_data in agents.values():
        audit = agent_data.get("per_call_snr", [])
        for entry in audit:
            trained_ids.add(entry.get("_id"))

    # Find held-out calls
    held_out = [c for c in api_data if c.get("_id") not in trained_ids]

    print(f"Testing {len(held_out[:10])} calls from {len(api_data)} total\n")

    results = []

    for i, call in enumerate(held_out[:10], 1):
        call_id = call.get("_id")
        agent_name = call.get("agent_name", "Unknown")
        audio_path = f"{API_DATA_DIR}/audio/{call_id}.mp3"

        if not os.path.exists(audio_path):
            continue

        print(f"[{i:2d}] {agent_name:25s}", end=" ... ", flush=True)

        # Upload
        t0 = time.time()
        with open(audio_path, "rb") as f:
            resp = requests.post(f"{UI_URL}/api/upload", files={"file": f}, timeout=900)

        print(f"HTTP {resp.status_code:3d} ... ", end="", flush=True)

        if resp.status_code != 200:
            print(f"SKIP")
            continue

        try:
            data = resp.json()
        except:
            print(f"INVALID_JSON")
            continue

        result_id = data.get("result_id")
        if not result_id:
            print(f"NO_RESULT_ID")
            continue

        # Now poll for actual result
        print(f"waiting ", end="", flush=True)

        for attempt in range(300):  # Up to 10 minutes
            r2 = requests.get(f"{UI_URL}/api/call/{result_id}")
            if r2.status_code == 200:
                result = r2.json()
                elapsed = time.time() - t0

                # Check identification
                ui_agent = (result.get("identified_agent") or "").lower().replace(" ", "_")
                api_agent = (agent_name or "").lower().replace(" ", "_")

                is_correct = ui_agent == api_agent or api_agent in ui_agent

                print(f"{elapsed:6.1f}s", end=" ")
                if is_correct:
                    print(f"OK")
                    results.append(("OK", elapsed, ui_agent, api_agent))
                else:
                    print(f"WRONG ({ui_agent})")
                    results.append(("WRONG", elapsed, ui_agent, api_agent))

                break

            time.sleep(2)

        else:
            print(f"TIMEOUT")

    # Summary
    print("\n" + "="*80)
    if results:
        correct = sum(1 for r in results if r[0] == "OK")
        total = len(results)
        accuracy = correct / total * 100
        times = [r[1] for r in results]
        avg_time = sum(times) / len(times)

        print(f"\n[UI TEST SUMMARY]")
        print(f"  Calls tested: {total}")
        print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"  Avg processing time: {avg_time:.1f}s")
        if times:
            print(f"  Min/Max time: {min(times):.1f}s / {max(times):.1f}s")

        # Save results
        with open("ui_sequential_results.json", "w") as f:
            json.dump({
                "accuracy_pct": accuracy,
                "correct": correct,
                "total": total,
                "avg_time_s": avg_time,
                "min_time_s": min(times) if times else 0,
                "max_time_s": max(times) if times else 0,
                "results": [
                    {"status": r[0], "time_s": r[1], "ui_agent": r[2], "api_agent": r[3]}
                    for r in results
                ]
            }, f, indent=2)

        print(f"\nResults saved to ui_sequential_results.json")

if __name__ == "__main__":
    main()
