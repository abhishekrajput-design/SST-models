import json
import requests
import time
from pathlib import Path

# Load API data to get ground truth
with open('data/audiofy/_dataset/index.json') as f:
    all_calls = json.load(f)

# Find 3+ minute calls from top agents only
top_agents = ['Ideal Dacaj', 'Kowsar Alam', 'Haris Bajwa', 'Anoush Sefatzadeh']
long_calls_to_test = []

for call_data in all_calls:
    duration = call_data.get('duration', 0)
    agent = call_data.get('agent_name', 'Unknown')

    # Only 3+ minute calls from top agents
    if duration >= 180 and agent in top_agents:
        call_id = call_data.get('_id', 'unknown')
        audio_path = call_data.get('audio_path')

        # Make sure audio file exists
        if audio_path and Path(audio_path).exists():
            long_calls_to_test.append({
                'call_id': call_id,
                'agent': agent,
                'duration': duration,
                'audio_path': audio_path,
                'ground_truth': call_data.get('speaker_json')
            })

# Limit to top 5 unique calls
long_calls_to_test = long_calls_to_test[:5]

print('=' * 100)
print('TESTING 3+ MINUTE LONG CALLS VIA LIVE API')
print('=' * 100)
print()

# Test against live API
API_URL = "http://13.42.127.218:8080"
results = []

for i, call_info in enumerate(long_calls_to_test, 1):
    duration_min = call_info['duration'] / 60
    print(f"[{i}] {call_info['agent']:<30} | {duration_min:5.1f} min")

    # Upload audio file
    with open(call_info['audio_path'], 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(f"{API_URL}/api/upload", files=files, timeout=300)
            if response.status_code == 200:
                result = response.json()
                result_id = result.get('result_id')

                # Poll for result
                for attempt in range(60):  # 10 minutes timeout
                    result_response = requests.get(f"{API_URL}/api/call/{result_id}", timeout=10)
                    if result_response.status_code == 200:
                        result_data = result_response.json()

                        if result_data.get('identified_agent'):
                            identified = result_data['identified_agent']
                            expected = call_info['agent']
                            sim = result_data.get('agent_similarity', 0)
                            match = "✓ OK" if identified == expected else "✗ WRONG"

                            print(f"    {match} | Expected: {expected:<30} | Got: {identified:<30} | Sim: {sim:.3f}")
                            results.append({
                                'agent': expected,
                                'duration_min': duration_min,
                                'correct': identified == expected,
                                'similarity': sim
                            })
                            break

                    time.sleep(10)
            else:
                print(f"    Upload failed: {response.status_code}")
        except Exception as e:
            print(f"    Error: {str(e)}")

# Summary
print()
print('=' * 100)
print('SUMMARY')
print('=' * 100)
if results:
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = 100 * correct / total if total > 0 else 0
    avg_sim = sum(r['similarity'] for r in results) / len(results) if results else 0

    print(f"Correct: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Avg Similarity: {avg_sim:.3f}")
    print()
    print("Per-Call Results:")
    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"  {status} {r['agent']:<30} | {r['duration_min']:5.1f} min | Sim: {r['similarity']:.3f}")
