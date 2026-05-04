import json
with open('data/audiofy/_dataset/index.json', encoding='utf-8') as f:
    index = json.load(f)
omar_calls = [r for r in index if 'omar' in r.get('agent_name', '').lower()]
print(f'Total Omar calls in index: {len(omar_calls)}')
for i, r in enumerate(omar_calls[:15]):
    print(f'  {i+1}. {r["_id"][:8]} - {r.get("duration_s")}s - phrases={r.get("n_agent_phrases", 0)}')
