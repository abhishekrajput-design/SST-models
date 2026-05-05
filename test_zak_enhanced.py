#!/usr/bin/env python
"""
Enhanced test for Zak using clustered voiceprints
Tests against multiple cluster centroids (max similarity wins)
"""

import json
import sys
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.diar_multi import diarize_multi

print("=" * 160)
print("ENHANCED TEST: ZAK RAISSI - MULTI-CLUSTER VOICEPRINTS")
print("=" * 160)

GT_FILE = Path("traning_data/zak_raissi/call_01/data.json")
AUDIO_FILE = Path("traning_data/zak_raissi/call_01/audio_16k.wav")

if not GT_FILE.exists() or not AUDIO_FILE.exists():
    print(f"ERROR: Missing files")
    sys.exit(1)

with open(GT_FILE) as f:
    gt_data = json.load(f)

gt_segments = gt_data.get('segments', [])
print(f"Ground Truth: {len(gt_segments)} segments")

agent_count = sum(1 for s in gt_segments if s.get('speaker') == 'agent')
customer_count = sum(1 for s in gt_segments if s.get('speaker') == 'customer')
print(f"  AGENT: {agent_count}, CUSTOMER: {customer_count}")

# Run diarization
print(f"\n[Running diarization with enhanced voiceprints...]")

parakeet_segments = []
for seg in gt_segments:
    parakeet_segments.append({
        'start': seg['start'],
        'end': seg['end'],
        'text': seg['text'],
    })

result = diarize_multi(
    segments=parakeet_segments,
    norm_wav=str(AUDIO_FILE),
    force_cpu=True
)

our_segments = result.get('segments', [])

# Compare
print("\n" + "=" * 160)
print("RESULTS")
print("=" * 160)

correct_count = 0
incorrect_count = 0
accuracy_by_role = {'agent': {'correct': 0, 'total': 0}, 'customer': {'correct': 0, 'total': 0}}
errors = []

for i in range(len(gt_segments)):
    gt_seg = gt_segments[i]
    our_seg = our_segments[i] if i < len(our_segments) else {}

    gt_speaker = gt_seg.get('speaker', 'unknown').lower()
    gt_display = 'AGENT' if gt_speaker == 'agent' else 'CUSTOMER'

    our_speaker = our_seg.get('identified_speaker', 'UNKNOWN').upper()
    if 'AGENT' in our_speaker:
        our_display = 'AGENT'
    elif 'CUSTOMER' in our_speaker:
        our_display = 'CUSTOMER'
    else:
        our_display = 'UNKNOWN'

    match = our_display == gt_display
    if match:
        correct_count += 1
    else:
        incorrect_count += 1
        sim = our_seg.get('_best_sim', 0.0)
        dur = gt_seg.get('end', 0) - gt_seg.get('start', 0)
        text = gt_seg.get('text', '')[:35]
        errors.append({
            'i': i+1,
            'gt': gt_display,
            'pred': our_display,
            'sim': sim,
            'dur': dur,
            'text': text
        })

    accuracy_by_role[gt_speaker]['total'] += 1
    if match:
        accuracy_by_role[gt_speaker]['correct'] += 1

total = len(gt_segments)
overall_accuracy = (correct_count / total * 100) if total > 0 else 0

print(f"\nOVERALL ACCURACY: {overall_accuracy:.1f}% ({correct_count}/{total})")
print(f"  AGENT:    {accuracy_by_role['agent']['correct']/max(1,accuracy_by_role['agent']['total'])*100:.1f}% ({accuracy_by_role['agent']['correct']}/{accuracy_by_role['agent']['total']})")
print(f"  CUSTOMER: {accuracy_by_role['customer']['correct']/max(1,accuracy_by_role['customer']['total'])*100:.1f}% ({accuracy_by_role['customer']['correct']}/{accuracy_by_role['customer']['total']})")

print(f"\nERRORS: {incorrect_count}")
for e in errors[:15]:
    print(f"  [{e['i']:3d}] GT={e['gt']:8s} Pred={e['pred']:8s} sim={e['sim']:.3f} dur={e['dur']:.1f}s | {e['text']}")

print(f"\n" + "=" * 160)
if overall_accuracy >= 95:
    print(f"[SUCCESS] Reached 95%+ accuracy! ({overall_accuracy:.1f}%)")
elif overall_accuracy >= 85:
    print(f"[GOOD] Significant improvement ({overall_accuracy:.1f}%)")
elif overall_accuracy >= 75:
    print(f"[BETTER] Improved to {overall_accuracy:.1f}%")
else:
    print(f"[NEEDS MORE WORK] Currently at {overall_accuracy:.1f}%")
print("=" * 160)
