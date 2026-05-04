import json
import os

# Load API data
with open('data/audiofy/_dataset/index.json') as f:
    calls = json.load(f)

# Filter for calls 3+ minutes
long_calls = []
for call_data in calls:
    duration = call_data.get('duration', 0)
    if duration >= 180:  # 3+ minutes
        agent = call_data.get('agent_name', 'Unknown')
        call_id = call_data.get('_id', 'unknown')
        long_calls.append({
            'id': call_id,
            'agent': agent,
            'duration': duration,
        })

# Sort by duration descending
long_calls.sort(key=lambda x: x['duration'], reverse=True)

print(f'Found {len(long_calls)} calls with 3+ minute duration')
print('=' * 90)
print()
for i, call in enumerate(long_calls[:15], 1):
    mins = call['duration'] / 60
    print(f'[{i:2d}] {call["id"]:<12} | {call["agent"]:<30} | {mins:5.1f} min')
