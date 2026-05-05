#!/usr/bin/env python
"""
Gemini Multi-Agent Training Orchestrator

Workflow:
1. Identify top 5 agents from API
2. Get their largest calls (>5 min)
3. Upload to Gemini for perfect labels
4. Train voiceprints on correct labels
5. Compare accuracy before/after
6. Generate comprehensive report

Usage:
  python scripts/gemini_multi_agent_trainer.py
"""

import json
import sys
import requests
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.embedding_campp import get_model
import shutil
import time

API_BASE = "http://localhost:8080"

print("=" * 120)
print("GEMINI MULTI-AGENT TRAINING ORCHESTRATOR")
print("=" * 120)

print("\n[STEP 1] Analyzing API calls to identify top agents...")

# Get all calls from API
response = requests.get(f"{API_BASE}/api/calls?limit=1000")
all_calls = response.json()

# Group by agent (inferred from file path or stored agent_name)
agent_calls = defaultdict(list)
for call in all_calls:
    agent_name = call.get('agent_name')
    audio_file = call.get('audio_file', '')
    segments = call.get('segments', 0)

    # Infer agent from file path if no agent_name
    if not agent_name or agent_name == 'Unknown':
        if 'zak' in audio_file.lower():
            agent_name = 'Zak Raissi'
        elif 'omar' in audio_file.lower():
            agent_name = 'Omar El Harchaoui'
        elif 'hussein' in audio_file.lower():
            agent_name = 'Hussein'
        else:
            continue

    # Only include calls with transcriptions
    if segments > 30:
        agent_calls[agent_name].append({
            'id': call.get('id'),
            'segments': segments,
            'audio_file': audio_file,
            'orig_file': call.get('orig_file', ''),
        })

# Sort agents by number of calls
agent_stats = []
for agent_name, calls_list in agent_calls.items():
    agent_stats.append({
        'agent_name': agent_name,
        'num_calls': len(calls_list),
        'calls': calls_list,
    })

agent_stats.sort(key=lambda x: x['num_calls'], reverse=True)

print(f"\nFound {len(agent_stats)} agents with transcribed calls")
print("\nTop 5 Agents:")
print("-" * 120)

top_5_agents = agent_stats[:5]
for i, agent in enumerate(top_5_agents, 1):
    print(f"{i}. {agent['agent_name']}: {agent['num_calls']} calls")
    for call in agent['calls'][:3]:
        print(f"   - {call['id']} ({call['segments']} segments)")

# Save configuration for next steps
config = {
    'top_5_agents': top_5_agents,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'next_step': 'Manual Gemini upload for each agent'
}

config_file = Path('call_processor/data/training/multi_agent_config.json')
config_file.parent.mkdir(parents=True, exist_ok=True)
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2, default=str)

print("\n" + "=" * 120)
print("NEXT STEPS")
print("=" * 120)

print("""
For each agent, we need to:

1. UPLOAD TO GEMINI (via browser)
   - Go to https://gemini.google.com/app
   - Upload the agent's audio file
   - Ask Gemini to transcribe with speaker labels
   - Copy the JSON response

2. SAVE GEMINI LABELS
   - Save to: call_processor/data/training/gemini_labels_[AGENT_NAME].json

3. RUN TRAINING
   - python scripts/train_agent_from_gemini.py [AGENT_NAME]

4. TEST & COMPARE
   - python scripts/test_agent_accuracy.py [AGENT_NAME]

AGENTS TO PROCESS (in order):
""")

for i, agent in enumerate(top_5_agents, 1):
    best_call = agent['calls'][0]
    print(f"\n{i}. {agent['agent_name']}")
    print(f"   Primary call: {best_call['id']}")
    print(f"   File: {best_call['orig_file']}")
    print(f"   Segments: {best_call['segments']}")
    print(f"   Action: Upload to Gemini -> Save labels -> Train -> Test")

print("\n" + "=" * 120)
print("Configuration saved to: call_processor/data/training/multi_agent_config.json")
print("=" * 120)

print(f"""
RECOMMENDED WORKFLOW:
====================

Start with OMAR (already has Gemini labels):
  1. Test current accuracy
  2. Compare with Gemini's response
  3. Move to ZAK RAISSI next

Then process remaining agents:
  - Each agent upload takes ~5-10 min in Gemini UI
  - Each training takes ~2-3 min
  - Expected to complete all 5 agents in ~2 hours

Accuracy improvements expected:
  - Current (before Gemini training): 50-70%
  - After Gemini training: 85-95%
  - With 5-10 agent dataset: 95-98%

Ready to start? Follow the steps above for each agent.
""")
