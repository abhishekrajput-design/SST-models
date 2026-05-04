#!/usr/bin/env python3
"""
Test top 5 agents on UI with 3-minute test recordings.
"""
import json
import time
import requests
import os
from collections import defaultdict

UI_URL = "http://localhost:8080"
API_DATA_DIR = "data/audiofy/_dataset"

# Top 5 agents to test
TOP_AGENTS = [
    "haris_bajwa",
    "kowsar_alam",
    "omar_el_harchaoui",
    "janusaan_jeyachandran",
    "ideal_dacaj"
]

def get_test_calls():
    """Get test calls for top 5 agents."""
    with open(f"{API_DATA_DIR}/index.json") as f:
        api_data = json.load(f)

    # Get trained IDs
    with open("data/agent_voiceprints/agents.json") as f:
        agents = json.load(f)

    trained_ids = set()
    for agent_data in agents.values():
        audit = agent_data.get('per_call_snr', [])
        for entry in audit:
            trained_ids.add(entry.get('_id'))

    # Find test calls for each agent
    test_calls_by_agent = defaultdict(list)
    for call in api_data:
        agent_name = call.get('agent_name', '').lower().replace(' ', '_')
        agent_slug = agent_name

        # Convert to slug format for matching
        for slug in TOP_AGENTS:
            if slug in agent_name or agent_name in slug:
                call_id = call.get('_id')
                if call_id not in trained_ids:  # Held-out calls
                    test_calls_by_agent[slug].append({
                        'id': call_id,
                        'agent_name': call.get('agent_name'),
                        'duration': call.get('duration_seconds', 0)
                    })
                break

    return test_calls_by_agent

def upload_and_test(audio_path, expected_agent):
    """Upload audio and test identification."""
    print(f"  Uploading... ", end="", flush=True)

    t0 = time.time()
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{UI_URL}/api/upload",
                files={"file": f},
                timeout=900
            )

        if resp.status_code != 200:
            print(f"FAILED (HTTP {resp.status_code})")
            return None

        result_id = resp.json().get("result_id")
        if not result_id:
            print(f"NO_ID")
            return None

        print(f"ID={result_id[:20]}... waiting... ", end="", flush=True)

        # Poll for result
        for attempt in range(300):
            resp2 = requests.get(f"{UI_URL}/api/call/{result_id}")
            if resp2.status_code == 200:
                result = resp2.json()
                elapsed = time.time() - t0

                identified = result.get("identified_agent", "Unknown").lower().replace(" ", "_")
                expected = expected_agent.lower().replace(" ", "_")

                is_correct = identified == expected or expected in identified or identified in expected

                sim = result.get("agent_similarity")
                warning = result.get("speaker_id_warning")

                status = "OK" if is_correct else "WRONG"
                print(f"{status:5s} ({elapsed:6.1f}s, sim={sim})")

                return {
                    "correct": is_correct,
                    "identified": identified,
                    "expected": expected,
                    "similarity": sim,
                    "time_s": elapsed,
                    "warning": warning
                }

            time.sleep(2)

    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")
        return None

    print("TIMEOUT")
    return None

def main():
    print("[TEST] UI Accuracy Test - Top 5 Agents")
    print("="*80)
    print()

    test_calls = get_test_calls()

    overall_results = []
    agent_results = defaultdict(list)

    for agent_slug in TOP_AGENTS:
        calls = test_calls.get(agent_slug, [])
        if not calls:
            print(f"[SKIP] {agent_slug.upper():30s} - no held-out calls")
            continue

        print(f"[TEST] {agent_slug.upper():30s} ({len(calls)} held-out calls)")
        print("-"*80)

        tested = 0
        for i, call in enumerate(calls[:3], 1):  # Test first 3 calls
            call_id = call['id']
            agent_name = call['agent_name']
            duration = call['duration']

            audio_path = f"{API_DATA_DIR}/audio/{call_id}.mp3"
            if not os.path.exists(audio_path):
                print(f"  [{i}] {call_id[:12]} - SKIP (no file)")
                continue

            print(f"  [{i}] {call_id[:12]} {duration:5.0f}s ({agent_name:25s}) ", end="", flush=True)

            result = upload_and_test(audio_path, agent_slug)
            if result:
                agent_results[agent_slug].append(result)
                overall_results.append((agent_slug, result))
                tested += 1

        print()

    # Summary
    print("\n" + "="*80)
    print("[RESULTS] Agent Accuracy Summary")
    print("="*80)
    print()

    for agent_slug in TOP_AGENTS:
        results = agent_results.get(agent_slug, [])
        if not results:
            continue

        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total * 100 if total > 0 else 0

        avg_sim = sum(r['similarity'] or 0 for r in results) / len(results) if results else 0
        avg_time = sum(r['time_s'] for r in results) / len(results)

        agent_name = agent_slug.replace("_", " ").title()

        print(f"{agent_name:30s} | Accuracy: {correct}/{total} ({accuracy:5.1f}%) | Avg Sim: {avg_sim:.3f} | Time: {avg_time:.0f}s")

        # Show results detail
        for i, r in enumerate(results, 1):
            status = "OK" if r['correct'] else "WRONG"
            expected_name = r['expected'].replace("_", " ").title()
            identified_name = r['identified'].replace("_", " ").title()
            print(f"     [{i}] {expected_name:25s} -> {identified_name:25s} | sim={r['similarity']:.3f} | {status}")

    # Overall
    print()
    print("-"*80)
    if overall_results:
        overall_correct = sum(1 for _, r in overall_results if r['correct'])
        overall_total = len(overall_results)
        overall_accuracy = overall_correct / overall_total * 100

        print(f"\nOVERALL ACCURACY: {overall_correct}/{overall_total} ({overall_accuracy:.1f}%)")
        print(f"Expected: 80-90%")

        if overall_accuracy >= 80:
            print("Status: PASSED")
        else:
            print("Status: NEEDS INVESTIGATION")

    # Save results
    with open("top_5_agents_ui_test.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_accuracy": overall_accuracy if overall_results else 0,
            "total_tested": overall_total if overall_results else 0,
            "results_by_agent": {
                agent: [
                    {
                        "correct": r['correct'],
                        "identified": r['identified'],
                        "expected": r['expected'],
                        "similarity": r['similarity'],
                        "time_s": r['time_s']
                    }
                    for r in agent_results[agent]
                ]
                for agent in TOP_AGENTS
            }
        }, f, indent=2)

    print(f"\nDetailed results saved to: top_5_agents_ui_test.json")

if __name__ == "__main__":
    main()
