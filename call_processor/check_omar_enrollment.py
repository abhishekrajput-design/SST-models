import json
with open('data/agent_voiceprints/agents.json', encoding='utf-8') as f:
    agents = json.load(f)
omar = agents.get('omar_el_harchaoui', {})
used_calls = omar.get('per_call_snr', [])
used_ids = set(r.get('_id') for r in used_calls if r.get('_id'))
print(f'Omar enrollment used: {len(used_ids)} calls')
for r in used_calls[:8]:
    print(f'  {r.get("_id", "?")[:8]} - {r.get("snr_db", 0):.1f}dB ({r.get("bucket", "?")})')
print(f'\nOmar has {len(omar.get("voiceprints", []))} voiceprints:')
for vp in omar.get('voiceprints', []):
    print(f'  {vp.get("bucket")} - {vp.get("n_clips")} clips, {vp.get("snr_db", 0):.1f}dB')
