#!/usr/bin/env python3
"""
Test UI multi-voiceprint accuracy against held-out API calls.
Uploads calls to UI, measures processing time, compares against API ground truth.
"""
import json
import time
import os
import sys
import requests
from pathlib import Path
from collections import defaultdict

UI_URL = "http://localhost:8080"
API_DATA_DIR = "data/audiofy/_dataset"
AGENTS_JSON = "data/agent_voiceprints/agents.json"

def load_agents():
    """Load trained agents."""
    with open(AGENTS_JSON) as f:
        return json.load(f)

def load_api_data():
    """Load API index with ground truth speaker labels."""
    with open(f"{API_DATA_DIR}/index.json") as f:
        return json.load(f)

def get_held_out_calls(api_data, agents, max_calls=50):
    """Get calls NOT used for training (held-out test set)."""
    # Get agent IDs used in training
    trained_ids = set()
    for agent_data in agents.values():
        audit = agent_data.get("per_call_snr", [])
        for entry in audit:
            trained_ids.add(entry.get("_id"))

    # Filter to held-out calls
    held_out = []
    for call in api_data:
        if call.get("_id") not in trained_ids:
            held_out.append(call)
        if len(held_out) >= max_calls:
            break

    return held_out

def upload_to_ui(audio_path, timeout=600):
    """Upload audio to UI, return result."""
    with open(audio_path, "rb") as f:
        files = {"file": f}
        try:
            resp = requests.post(
                f"{UI_URL}/api/upload",
                files=files,
                timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

def get_result(result_id, max_wait=600):
    """Poll for result until ready."""
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(f"{UI_URL}/api/call/{result_id}")
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2)
    return None

def extract_agent_segments(result, agent_name):
    """Count agent vs customer segments."""
    segments = result.get("segments", [])
    agent_count = sum(1 for s in segments if s.get("speaker") == "AGENT")
    customer_count = sum(1 for s in segments if s.get("speaker") == "CUSTOMER")
    return agent_count, customer_count

def get_api_agent_segments(call, agent_name):
    """Get agent vs customer segment count from API ground truth."""
    speaker_json = call.get("speaker_json", {})
    if not speaker_json:
        return None, None

    segments = speaker_json.get("segments", [])
    agent_count = sum(1 for s in segments if s.get("speaker") == agent_name)
    customer_count = sum(1 for s in segments if s.get("speaker") != agent_name)
    return agent_count, customer_count

def evaluate_call(ui_result, api_call, call_id):
    """Compare UI result against API ground truth."""
    if not api_call.get("speaker_json"):
        return None

    ui_agent = ui_result.get("identified_agent", "Unknown").lower().replace(" ", "_")
    api_agent = api_call.get("agent_name", "").lower().replace(" ", "_")

    # Check if identified correctly
    identified_correct = ui_agent == api_agent or api_agent in ui_agent

    # Count segments
    ui_agent_segs, ui_cust_segs = extract_agent_segments(ui_result, ui_agent)
    api_agent_segs, api_cust_segs = get_api_agent_segments(api_call, api_call.get("agent_name"))

    return {
        "call_id": call_id,
        "identified_correct": identified_correct,
        "ui_agent": ui_agent,
        "api_agent": api_agent,
        "ui_agent_segs": ui_agent_segs,
        "ui_cust_segs": ui_cust_segs,
        "api_agent_segs": api_agent_segs,
        "api_cust_segs": api_cust_segs,
    }

def main():
    print("[test-ui] Loading agents...", flush=True)
    agents = load_agents()

    print("[test-ui] Loading API data...", flush=True)
    api_data = load_api_data()

    print("[test-ui] Finding held-out calls...", flush=True)
    held_out = get_held_out_calls(api_data, agents, max_calls=20)
    print(f"[test-ui] Found {len(held_out)} held-out calls to test", flush=True)

    results = []
    times = []
    correct = 0

    for i, call in enumerate(held_out, 1):
        call_id = call.get("_id")
        agent_name = call.get("agent_name", "Unknown")
        audio_file = call.get("audio_file", "")

        if not audio_file:
            print(f"[{i:2d}] {call_id[:8]} - SKIP (no audio file)")
            continue

        audio_path = f"{API_DATA_DIR}/audio/{call_id}.mp3"
        if not os.path.exists(audio_path):
            print(f"[{i:2d}] {call_id[:8]} - SKIP (audio not found)")
            continue

        print(f"[{i:2d}] {call_id[:8]} ({agent_name:20s}) ... ", end="", flush=True)

        # Upload to UI
        t0 = time.time()
        upload_resp = upload_to_ui(audio_path)

        if "error" in upload_resp:
            print(f"UPLOAD_ERROR: {upload_resp['error']}")
            continue

        result_id = upload_resp.get("result_id")
        if not result_id:
            print(f"NO_RESULT_ID")
            continue

        # Wait for result
        result = get_result(result_id)
        elapsed = time.time() - t0
        times.append(elapsed)

        if not result:
            print(f"TIMEOUT ({elapsed:.1f}s)")
            continue

        # Evaluate
        eval_result = evaluate_call(result, call, call_id)
        if eval_result:
            results.append(eval_result)
            is_correct = eval_result["identified_correct"]
            if is_correct:
                correct += 1
                status = "OK"
            else:
                status = f"WRONG (got {eval_result['ui_agent']})"
            print(f"{status:30s} ({elapsed:6.1f}s)")

    # Summary
    print("\n" + "="*80)
    if results:
        accuracy = correct / len(results) * 100
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nUI ACCURACY TEST RESULTS")
        print(f"  Tested: {len(results)} calls")
        print(f"  Correct: {correct}/{len(results)} ({accuracy:.1f}%)")
        print(f"  Processing time (avg/min/max): {avg_time:.1f}s / {min_time:.1f}s / {max_time:.1f}s")

        # Per-agent breakdown
        by_agent = defaultdict(list)
        for r in results:
            by_agent[r["api_agent"]].append(r["identified_correct"])

        print(f"\nPer-agent breakdown:")
        for agent, correct_list in sorted(by_agent.items()):
            pct = sum(correct_list) / len(correct_list) * 100
            print(f"  {agent:25s} {sum(correct_list):2d}/{len(correct_list):2d} ({pct:5.1f}%)")

        # Save detailed results
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
        print(f"\nDetailed results saved to ui_test_results.json")
    else:
        print("No results to report")

if __name__ == "__main__":
    main()
