#!/usr/bin/env python3
"""
Focused UI test - upload 5 API calls, measure accuracy + speed.
"""
import json
import time
import os
import requests
from collections import defaultdict

UI_URL = "http://localhost:8080"
API_DATA_DIR = "data/audiofy/_dataset"

def load_api_data():
    with open(f"{API_DATA_DIR}/index.json") as f:
        return json.load(f)

def main():
    print("[test-ui] Loading API data...")
    api_data = load_api_data()

    # Load agents to find trained vs held-out
    with open("data/agent_voiceprints/agents.json") as f:
        agents = json.load(f)

    trained_ids = set()
    for agent_data in agents.values():
        audit = agent_data.get("per_call_snr", [])
        for entry in audit:
            trained_ids.add(entry.get("_id"))

    print(f"[*] Trained calls: {len(trained_ids)}")
    print(f"[*] Total API calls: {len(api_data)}")

    # Find held-out calls
    held_out = [c for c in api_data if c.get("_id") not in trained_ids]
    print(f"[*] Held-out calls: {len(held_out)}\n")

    results = []
    times = []
    correct = 0

    for i, call in enumerate(held_out[:5], 1):
        call_id = call.get("_id")
        agent_name = call.get("agent_name", "Unknown")
        audio_path = f"{API_DATA_DIR}/audio/{call_id}.mp3"

        if not os.path.exists(audio_path):
            print(f"[{i}] {call_id[:12]} - SKIP (no file)")
            continue

        print(f"[{i}] {agent_name:25s} ... ", end="", flush=True)

        # Upload
        t0 = time.time()
        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    f"{UI_URL}/api/upload",
                    files={"file": f},
                    timeout=600
                )
                if resp.status_code != 200:
                    print(f"HTTP {resp.status_code}")
                    continue

            result_id = resp.json().get("result_id")
            if not result_id:
                print("NO_ID")
                continue

            # Poll for result
            while time.time() - t0 < 600:
                resp2 = requests.get(f"{UI_URL}/api/call/{result_id}")
                if resp2.status_code == 200:
                    break
                time.sleep(2)

            elapsed = time.time() - t0
            times.append(elapsed)

            if resp2.status_code != 200:
                print(f"TIMEOUT")
                continue

            result = resp2.json()

            # Compare
            ui_agent = (result.get("identified_agent") or "").lower().replace(" ", "_")
            api_agent = (agent_name or "").lower().replace(" ", "_")

            is_match = ui_agent == api_agent or api_agent in ui_agent
            if is_match:
                correct += 1
                print(f"OK              {elapsed:6.1f}s")
            else:
                print(f"WRONG ({ui_agent:20s}) {elapsed:6.1f}s")

            results.append({
                "call_id": call_id,
                "correct": is_match,
                "ui_agent": ui_agent,
                "api_agent": api_agent,
                "time_s": elapsed
            })

        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")

    # Summary
    print("\n" + "="*80)
    if results:
        accuracy = correct / len(results) * 100
        avg_time = sum(times) / len(times) if times else 0

        print(f"\n[UI TEST RESULTS]")
        print(f"  Calls tested: {len(results)}")
        print(f"  Accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")
        if times:
            print(f"  Avg speed: {avg_time:.1f}s")
            print(f"  Min/Max: {min(times):.1f}s / {max(times):.1f}s")

        with open("ui_test_summary.json", "w") as f:
            json.dump({
                "accuracy_pct": accuracy,
                "correct": correct,
                "total": len(results),
                "avg_time_s": avg_time,
                "results": results
            }, f, indent=2)

        print(f"\nSummary saved to ui_test_summary.json")

if __name__ == "__main__":
    main()
