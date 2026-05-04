"""
Audit: How much customer voice contaminated the current enrollment?
Check each training call's speaker_json to measure agent vs customer time share.
"""
import json
from pathlib import Path

AGENTS_JSON = Path(__file__).parent / "data/agent_voiceprints/agents.json"
INDEX_JSON = Path(__file__).parent / "data/audiofy/_dataset/index.json"

with open(AGENTS_JSON, encoding='utf-8') as f:
    agents = json.load(f)

with open(INDEX_JSON, encoding='utf-8') as f:
    index = json.load(f)

# Build call ID -> call record map
call_map = {rec['_id']: rec for rec in index}

print("=== ENROLLMENT PURITY AUDIT ===\n")

for agent_slug, info in agents.items():
    if not isinstance(info, dict):
        continue

    agent_name = info.get('agent_name', agent_slug)
    used_calls = info.get('per_call_snr', [])

    if not used_calls:
        continue

    print(f"{agent_name} ({len(used_calls)} calls)")

    total_agent_phrases = 0
    total_customer_phrases = 0

    for call_rec in used_calls:
        call_id = call_rec.get('_id')
        if not call_id or call_id not in call_map:
            continue

        api_rec = call_map[call_id]
        speaker_json = api_rec.get('speaker_json', [])

        agent_phrases = 0
        customer_phrases = 0

        for phrase in speaker_json:
            if not isinstance(phrase, dict):
                continue
            speaker = (phrase.get('speaker') or '').strip()
            is_agent = bool(speaker) and speaker.lower() != 'customer'

            if is_agent:
                agent_phrases += 1
            else:
                customer_phrases += 1

        total_agent_phrases += agent_phrases
        total_customer_phrases += customer_phrases

        contamination = 100.0 * customer_phrases / max(agent_phrases + customer_phrases, 1)
        print(f"  {call_id[:8]} - {agent_phrases} agent, {customer_phrases} customer "
              f"({contamination:.0f}% contamination)")

    if total_agent_phrases + total_customer_phrases == 0:
        continue

    purity = 100.0 * total_agent_phrases / (total_agent_phrases + total_customer_phrases)
    print(f"  TOTAL: {total_agent_phrases} agent, {total_customer_phrases} customer "
          f"({purity:.1f}% purity)\n")
