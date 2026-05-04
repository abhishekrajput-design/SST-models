#!/usr/bin/env python3
"""
UI test with queued uploads (UI processes one at a time).
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

def upload_to_ui(audio_path, max_retries=10):
    """Upload to UI, retry on 409 (conflict)."""
    for attempt in range(max_retries):
        with open(audio_path, "rb") as f:
            try:
                resp = requests.post(f"{UI_URL}/api/upload", files={"file": f}, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 409:
                    # Busy - wait and retry
                    time.sleep(2)
                    continue
                else:
                    return {"error": f"HTTP {resp.status_code}"}
            except Exception as e:
                return {"error": str(e)}
    return {"error": "Max retries exceeded (UI still busy)"}

def get_result(result_id, max_wait=600):
    """Poll for result."""
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(f"{UI_URL}/api/call/{result_id}")
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2)
    return None

def main():
    print("[ui-test] Loading API data...", flush=True)
    api_data = load_api_data()

    # Get agents and their trained calls
    with open("data/agent_voiceprints/agents.json") as f:
        agents = json.load(f)

    trained_ids = set()
    for agent_data in agents.values():
        audit = agent_data.get("per_call_snr", [])
        for entry in audit:
            trained_ids.add(entry.get("_id"))

    # Find held-out calls
    held_out = [c for c in api_data if c.get("_id") not in trained_ids]
    print(f"[ui-test] Held-out calls: {len(held_out)} / {len(api_data)}", flush=True)

    results = []
    times = []
    correct = 0

    for i, call in enumerate(held_out[:20], 1):  # Test first 20
        call_id = call.get("_id")
        agent_name = call.get("agent_name", "Unknown")
        audio_path = f"{API_DATA_DIR}/audio/{call_id}.mp3"

        if not os.path.exists(audio_path):
            continue

        print(f"[{i:2d}] {call_id[:8]} ({agent_name:22s}) ... ", end="", flush=True)

        # Upload & time
        t0 = time.time()
        upload_resp = upload_to_ui(audio_path)

        if "error" in upload_resp:
            print(f"ERROR: {upload_resp['error'][:40]}")
            continue

        result_id = upload_resp.get("result_id")
        if not result_id:
            print(f"NO_ID")
            continue

        print(f"waiting ... ", end="", flush=True)

        # Wait for result
        result = get_result(result_id)
        elapsed = time.time() - t0
        times.append(elapsed)

        if not result:
            print(f"TIMEOUT")
            continue

        # Compare
        ui_agent = (result.get("identified_agent") or "").lower().replace(" ", "_")
        api_agent = (agent_name or "").lower().replace(" ", "_")

        is_match = ui_agent == api_agent or api_agent in ui_agent
        if is_match:
            correct += 1
            status = "OK"
        else:
            status = f"WRONG ({ui_agent})"

        results.append({
            "call_id": call_id,
            "correct": is_match,
            "ui_agent": ui_agent,
            "api_agent": api_agent,
            "time_s": elapsed
        })

        print(f"{status:25s} {elapsed:7.1f}s")

    # Summary
    print("\n" + "="*80)
    if results:
        accuracy = correct / len(results) * 100
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n[UI TEST RESULTS]")
        print(f"  Calls tested: {len(results)}")
        print(f"  Accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")
        print(f"  Avg time: {avg_time:.1f}s")
        print(f"  Min/Max time: {min_time:.1f}s / {max_time:.1f}s")

        # Per-agent
        by_agent = defaultdict(list)
        for r in results:
            by_agent[r["api_agent"]].append(r["correct"])

        print(f"\n  Per-agent breakdown:")
        for agent in sorted(by_agent.keys()):
            correct_list = by_agent[agent]
            pct = sum(correct_list) / len(correct_list) * 100 if correct_list else 0
            print(f"    {agent:25s} {sum(correct_list):2d}/{len(correct_list)} ({pct:5.1f}%)")

        # Save
        with open("ui_test_results.json", "w") as f:
            json.dump({
                "accuracy_pct": accuracy,
                "correct": correct,
                "total": len(results),
                "avg_time_s": avg_time,
                "min_time_s": min_time,
                "max_time_s": max_time,
                "results": results
            }, f, indent=2)

        print(f"\nResults saved to ui_test_results.json")

if __name__ == "__main__":
    main()
