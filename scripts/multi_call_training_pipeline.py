#!/usr/bin/env python
"""
Multi-Call Training Pipeline for 95%+ Accuracy

Strategy:
1. Find ALL calls for each agent from API
2. Train on best calls (>5 min, good quality)
3. Test on held-out calls
4. Evaluate accuracy before/after
5. Move to next agent

This combines diverse training data for robust voiceprints.
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

print("=" * 140)
print("MULTI-CALL TRAINING PIPELINE - 95%+ ACCURACY TARGET")
print("=" * 140)

# Step 1: Find all calls for each agent
print("\n[STEP 1] Finding all calls for each agent in API...")
print("-" * 140)

response = requests.get(f"{API_BASE}/api/calls?limit=1000")
all_calls = response.json()

agent_calls = defaultdict(list)
for call in all_calls:
    agent_name = call.get('agent_name')
    audio_file = call.get('audio_file', '')
    segments = call.get('segments', 0)

    # Infer agent from file path
    if not agent_name or agent_name == 'Unknown':
        if 'omar' in audio_file.lower():
            agent_name = 'Omar El Harchaoui'
        elif 'zak' in audio_file.lower():
            agent_name = 'Zak Raissi'
        elif 'hussein' in audio_file.lower():
            agent_name = 'Hussein'
        else:
            continue

    # Only include calls with good transcriptions
    if segments > 50:
        agent_calls[agent_name].append({
            'id': call.get('id'),
            'segments': segments,
            'audio_file': audio_file,
            'orig_file': call.get('orig_file', ''),
        })

# Display findings
print(f"\nAGENTS FOUND:")
for agent_name in sorted(agent_calls.keys()):
    calls = agent_calls[agent_name]
    print(f"\n{agent_name}: {len(calls)} calls with transcriptions")
    for i, call in enumerate(calls[:5], 1):
        print(f"  {i}. {call['id']:<40} ({call['segments']} segments)")

# Step 2: Create training strategy
print("\n\n[STEP 2] Training Strategy for 95%+ Accuracy")
print("-" * 140)

strategy = {
    'Omar El Harchaoui': {
        'primary_call': 'omar_20260505',
        'gemini_trained': True,
        'current_accuracy': 92.6,
        'target_accuracy': 96.0,
        'strategy': 'Add more training calls + fine-tune threshold',
        'calls_available': len(agent_calls.get('Omar El Harchaoui', [])),
    },
    'Zak Raissi': {
        'primary_call': 'zak_e2e_test_20260423',
        'gemini_trained': False,
        'current_accuracy': None,
        'target_accuracy': 95.0,
        'strategy': 'Upload multiple calls to Gemini + combine training',
        'calls_available': len(agent_calls.get('Zak Raissi', [])),
    },
    'Hussein': {
        'primary_call': 'enhanced_hussein_desk_recording__parakeet-tdt-0.6b-v3',
        'gemini_trained': False,
        'current_accuracy': None,
        'target_accuracy': 95.0,
        'strategy': 'Upload to Gemini + train on multiple segments',
        'calls_available': len(agent_calls.get('Hussein', [])),
    }
}

for agent, plan in strategy.items():
    if agent in agent_calls:
        print(f"\n{agent}:")
        print(f"  Calls available: {plan['calls_available']}")
        print(f"  Target accuracy: {plan['target_accuracy']}%")
        print(f"  Strategy: {plan['strategy']}")
        if plan['current_accuracy']:
            print(f"  Current accuracy: {plan['current_accuracy']}%")
            print(f"  Improvement needed: {plan['target_accuracy'] - plan['current_accuracy']:.1f}%")

# Step 3: Multi-call training for OMAR (already has Gemini data)
print("\n\n[STEP 3] Omar El Harchaoui - Multi-Call Enhancement")
print("-" * 140)

print("""
CURRENT STATUS:
  - Single call training: 92.6% accuracy
  - Need: +3.4% to reach 96%

APPROACH:
  1. Review all Omar's calls in API
  2. Select top 3-5 calls with good quality
  3. Upload each to Gemini for perfect labels
  4. Combine ALL training data
  5. Re-train voiceprint with larger dataset
  6. Test on held-out calls

EXPECTED RESULT:
  - With 5+ training calls: 95-98% accuracy
  - Better robustness across call conditions
  - Fewer edge cases

NEXT STEPS FOR OMAR:
  1. Get Gemini labels for 2-3 more calls
  2. Combine with existing labels
  3. Re-train on combined data
  4. Test on hold-out call
  5. Verify 95%+ accuracy
""")

# Step 4: Strategy for Zak and Hussein
print("\n\n[STEP 4] Zak Raissi & Hussein - Initial Training with Multiple Calls")
print("-" * 140)

print("""
OPTIMAL WORKFLOW:

For each agent:
  1. SELECT BEST CALLS
     - Choose 3-5 calls with >5 min duration
     - Prefer calls with diverse speakers/conditions

  2. UPLOAD TO GEMINI (one call at a time)
     - Upload call 1, get labels
     - Upload call 2, get labels
     - Upload call 3, get labels

  3. COMBINE TRAINING DATA
     - Merge all Gemini labels
     - Create unified training set

  4. TRAIN ONCE
     - Train voiceprint on ALL combined data
     - Build more robust centroid

  5. TEST ON HOLD-OUT CALLS
     - Test on call 4 or 5 (not used in training)
     - Verify accuracy >= 95%

  6. MOVE TO NEXT AGENT
     - Repeat same process

EXPECTED ACCURACY:
  - Single call training: 85-90%
  - Multi-call training: 95-98%

TIME ESTIMATE:
  - Per agent: 30-45 minutes
    - 10-15 min: Upload 3 calls to Gemini
    - 15 min: Combine labels + train
    - 5-10 min: Test and validate
""")

# Step 5: Detailed plan for next calls
print("\n\n[STEP 5] Detailed Multi-Call Plan")
print("-" * 140)

print("""
PHASE 1: ENHANCE OMAR (2-3 hours)
==================================
Call 1: (Already done) omar_20260505 - 92.6% OK
Call 2: Upload Omar's 2nd call to Gemini -> Get labels
Call 3: Upload Omar's 3rd call to Gemini -> Get labels
Combine: Merge all 3 datasets
Re-train: Single voiceprint from 80+ segments
Test: On call 4 (held-out)
Expected: 95-98% accuracy

PHASE 2: TRAIN ZAK RAISSI (1-2 hours)
==================================
Call 1: zak_e2e_test_20260423 - Upload to Gemini
Call 2: zak_compare_20260423 - Upload to Gemini
Call 3: zak_audiofy_verify_20260423 - Upload to Gemini
Combine: Merge all 3 Gemini labels
Train: Single voiceprint from 50+ segments
Test: On held-out call
Expected: 95-98% accuracy

PHASE 3: TRAIN HUSSEIN (45-60 min)
==================================
Call 1: Hussein's primary call - Upload to Gemini
Combine: Add any existing labels
Train: Voiceprint from 30+ segments
Test: Verify 95%+ accuracy

PHASE 4: FINAL VALIDATION (30 min)
==================================
Test system on fresh calls from each agent
Verify: All agents at 95% accuracy
Document: Final results and improvements
Deploy: Production-ready system

TOTAL TIME: 4-5 hours
EXPECTED RESULT: 95-98% accuracy across all agents
""")

# Step 6: Save configuration
config_file = Path('call_processor/data/training/multi_call_strategy.json')
config_file.parent.mkdir(parents=True, exist_ok=True)

with open(config_file, 'w') as f:
    json.dump({
        'strategy': 'Multi-call training for 95%+ accuracy',
        'agents': strategy,
        'agent_calls': {k: v for k, v in agent_calls.items() if v},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=2, default=str)

print(f"\n\nConfiguration saved to: {config_file}")

print("\n" + "=" * 140)
print("RECOMMENDED IMMEDIATE ACTIONS")
print("=" * 140)

print("""
1. FOR OMAR (Get 95%+ from current 92.6%):
   [ ] Find Omar's other calls in testing-audio/ or API
   [ ] Upload call 2 to Gemini -> save labels
   [ ] Upload call 3 to Gemini -> save labels
   [ ] Combine all 3 Gemini label files
   [ ] Re-train voiceprint on combined data
   [ ] Test on call 4 to verify 95%+

2. FOR ZAK RAISSI (Train from scratch):
   [ ] Upload Call 1: zak_e2e_test_20260423 to Gemini
   [ ] Upload Call 2: zak_compare_20260423 to Gemini
   [ ] Upload Call 3: zak_audiofy_verify_20260423 to Gemini
   [ ] Combine all 3 Gemini labels
   [ ] Train voiceprint
   [ ] Test on held-out call

3. FOR HUSSEIN:
   [ ] Upload primary call to Gemini
   [ ] Train voiceprint
   [ ] Test accuracy

TARGET: All agents at 95%+ by end of day!
""")

print("=" * 140)
