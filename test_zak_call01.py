#!/usr/bin/env python
"""
Test Zak's trained model on call_01
Compare system output vs Gemini ground truth labels
"""

import json
import sys
import numpy as np
from pathlib import Path
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.diar_multi import diarize_multi

print("=" * 160)
print("TESTING ZAK RAISSI ON CALL_01 - GROUND TRUTH COMPARISON")
print("=" * 160)

# Load ground truth (Gemini labels)
GT_FILE = Path("traning_data/zak_raissi/call_01/data.json")
AUDIO_FILE = Path("traning_data/zak_raissi/call_01/20260503T104319184_615752.mp3")

if not GT_FILE.exists():
    print(f"ERROR: Ground truth file not found: {GT_FILE}")
    sys.exit(1)

if not AUDIO_FILE.exists():
    print(f"ERROR: Audio file not found: {AUDIO_FILE}")
    sys.exit(1)

with open(GT_FILE) as f:
    gt_data = json.load(f)

gt_segments = gt_data.get('segments', [])
print(f"\nGround Truth (Gemini Labels):")
print(f"  Call ID: {gt_data.get('call_id')}")
print(f"  Agent: {gt_data.get('agent_name')}")
print(f"  Total segments: {len(gt_segments)}")

# Count ground truth
gt_agent = sum(1 for s in gt_segments if s.get('speaker') == 'agent')
gt_customer = sum(1 for s in gt_segments if s.get('speaker') == 'customer')
print(f"  Agent segments: {gt_agent}")
print(f"  Customer segments: {gt_customer}")

# Run diarization with trained Zak
print(f"\n[Running diarization with trained Zak model...]")

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
print(f"System output: {len(our_segments)} segments")

# Detailed comparison
print("\n" + "=" * 160)
print("SEGMENT-BY-SEGMENT COMPARISON")
print("=" * 160)

print(f"\n{'#':<3} {'GT':<12} {'System':<12} {'Match':<7} {'Similarity':<10} {'Duration':<8} {'Text':<40}")
print("-" * 160)

correct_count = 0
incorrect_count = 0
accuracy_by_role = {'agent': {'correct': 0, 'total': 0}, 'customer': {'correct': 0, 'total': 0}}

comparison_data = []

for i in range(len(gt_segments)):
    gt_seg = gt_segments[i]
    our_seg = our_segments[i] if i < len(our_segments) else {}

    # Ground truth label
    gt_speaker = gt_seg.get('speaker', 'unknown').lower()
    gt_display = 'AGENT' if gt_speaker == 'agent' else 'CUSTOMER'

    # System label
    our_speaker = our_seg.get('identified_speaker', 'UNKNOWN').upper()
    if 'AGENT' in our_speaker:
        our_display = 'AGENT'
    elif 'CUSTOMER' in our_speaker:
        our_display = 'CUSTOMER'
    else:
        our_display = 'UNKNOWN'

    # Compare
    match = our_display == gt_display
    match_str = "[OK]" if match else "[XX]"

    if match:
        correct_count += 1
    else:
        incorrect_count += 1

    # Track per-role accuracy
    accuracy_by_role[gt_speaker]['total'] += 1
    if match:
        accuracy_by_role[gt_speaker]['correct'] += 1

    # Details
    sim = our_seg.get('_best_sim', 0.0)
    dur = gt_seg.get('end', 0) - gt_seg.get('start', 0)
    text = gt_seg.get('text', '')[:35]

    print(f"{i+1:<3} {gt_display:<12} {our_display:<12} {match_str:<7} {sim:<10.3f} {dur:<8.2f} {text:<40}")

    comparison_data.append({
        'segment_num': i + 1,
        'gt_role': gt_display,
        'system_role': our_display,
        'match': match,
        'similarity': sim,
        'duration': dur,
        'text': gt_seg.get('text', '')
    })

# Summary
print("\n" + "=" * 160)
print("ACCURACY SUMMARY")
print("=" * 160)

total = len(gt_segments)
overall_accuracy = (correct_count / total * 100) if total > 0 else 0

print(f"\nOVERALL ACCURACY: {overall_accuracy:.1f}% ({correct_count}/{total} correct)")
print(f"  Correct:   {correct_count}")
print(f"  Incorrect: {incorrect_count}")

print(f"\nPER-ROLE ACCURACY:")
for role in ['agent', 'customer']:
    stats = accuracy_by_role[role]
    if stats['total'] > 0:
        role_acc = (stats['correct'] / stats['total'] * 100)
        print(f"  {role.upper():<10}: {role_acc:6.1f}% ({stats['correct']}/{stats['total']} correct)")

# Error analysis
print(f"\n" + "=" * 160)
print("ERROR ANALYSIS")
print("=" * 160)

agent_errors = [d for d in comparison_data if d['gt_role'] == 'AGENT' and not d['match']]
customer_errors = [d for d in comparison_data if d['gt_role'] == 'CUSTOMER' and not d['match']]

print(f"\nAGENT segments misclassified as CUSTOMER: {len(agent_errors)}")
if agent_errors:
    sims = [e['similarity'] for e in agent_errors]
    print(f"  Average similarity: {np.mean(sims):.3f}")
    print(f"  Examples:")
    for e in agent_errors[:3]:
        print(f"    - '{e['text'][:40]}' (sim={e['similarity']:.3f})")

print(f"\nCUSTOMER segments misclassified as AGENT: {len(customer_errors)}")
if customer_errors:
    sims = [e['similarity'] for e in customer_errors]
    print(f"  Average similarity: {np.mean(sims):.3f}")
    print(f"  Examples:")
    for e in customer_errors[:3]:
        print(f"    - '{e['text'][:40]}' (sim={e['similarity']:.3f})")

# Save results
results_file = Path("call_processor/data/training/zak_call01_test_results.json")
results_file.parent.mkdir(parents=True, exist_ok=True)

with open(results_file, 'w') as f:
    json.dump({
        'call': 'call_01',
        'overall_accuracy': overall_accuracy,
        'correct': correct_count,
        'incorrect': incorrect_count,
        'per_role_accuracy': accuracy_by_role,
        'agent_errors': len(agent_errors),
        'customer_errors': len(customer_errors),
        'segments': comparison_data
    }, f, indent=2)

print(f"\n" + "=" * 160)
print(f"Results saved to: {results_file}")
print("=" * 160)

# Conclusion
if overall_accuracy >= 95:
    print(f"\n[SUCCESS] Zak achieved 95%+ accuracy! ({overall_accuracy:.1f}%)")
    print("Ready for production use!")
elif overall_accuracy >= 90:
    print(f"\n[GOOD] Zak achieved 90%+ accuracy ({overall_accuracy:.1f}%)")
    print("Needs 1-2 more calls for 95%+")
else:
    print(f"\n[NEEDS IMPROVEMENT] Current accuracy: {overall_accuracy:.1f}%")
    print(f"Need to add more training calls or refine labels")
