#!/usr/bin/env python
"""
Detailed Segment-by-Segment Role Identification Comparison

Compare our system's speaker identification against Gemini's correct labels.
Show exactly which segments we're getting right/wrong.

Helps identify patterns in misclassifications.
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
print("DETAILED ROLE IDENTIFICATION COMPARISON - OMAR EL HARCHAOUI")
print("=" * 160)

# Load Gemini's perfect labels (ground truth)
GEMINI_LABELS = Path("call_processor/data/training/gemini_labels.json")
AUDIO_FILE = Path("call_processor/data/processed/enhanced_20260505T073055769_385036/norm_enhanced_20260505T073055769_385036.wav")
ALT_AUDIO = Path("testing-audio/omar_test/20260505T073055769_385036.mp3")

if not GEMINI_LABELS.exists():
    print("ERROR: Gemini labels not found")
    sys.exit(1)

with open(GEMINI_LABELS) as f:
    gemini_data = json.load(f)

gemini_segments = gemini_data.get('segments', [])
agent_name = gemini_data.get('agent_name', 'Unknown')

print(f"\nAgent: {agent_name}")
print(f"Total segments: {len(gemini_segments)}")

# Count ground truth
gt_agent = sum(1 for s in gemini_segments if s.get('speaker') == 'agent')
gt_customer = sum(1 for s in gemini_segments if s.get('speaker') == 'customer')

print(f"Ground truth: {gt_agent} AGENT + {gt_customer} CUSTOMER = {len(gemini_segments)} total")

# Find audio file
audio_path = None
if AUDIO_FILE.exists():
    audio_path = str(AUDIO_FILE)
elif ALT_AUDIO.exists():
    audio_path = str(ALT_AUDIO)
else:
    print(f"ERROR: Audio file not found")
    sys.exit(1)

print(f"Audio: {Path(audio_path).name}")

# Create segments for diarization
parakeet_segments = []
for seg in gemini_segments:
    parakeet_segments.append({
        'start': seg['start'],
        'end': seg['end'],
        'text': seg['text'],
    })

# Run diarization
print(f"\n[Running diarization...]")
result = diarize_multi(
    segments=parakeet_segments,
    norm_wav=audio_path,
    force_cpu=True
)

our_segments = result.get('segments', [])

print(f"System output: {len(our_segments)} segments")

# Detailed comparison
print("\n" + "=" * 160)
print("DETAILED SEGMENT-BY-SEGMENT COMPARISON")
print("=" * 160)

# Header
print(f"\n{'#':<3} {'Gemini':<12} {'System':<12} {'Match':<7} {'Sim':<7} {'Dur(s)':<8} {'Text':<45}")
print("-" * 160)

correct_count = 0
incorrect_count = 0
accuracy_by_role = {'agent': {'correct': 0, 'total': 0}, 'customer': {'correct': 0, 'total': 0}}

comparison_data = []

for i in range(len(gemini_segments)):
    gt_seg = gemini_segments[i]
    our_seg = our_segments[i] if i < len(our_segments) else {}

    # Ground truth label
    gt_speaker = gt_seg.get('speaker', 'unknown').lower()
    gt_display = 'AGENT' if gt_speaker == 'agent' else 'CUSTOMER'

    # Our system's label
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
    text = gt_seg.get('text', '')[:40]

    print(f"{i+1:<3} {gt_display:<12} {our_display:<12} {match_str:<7} {sim:<7.3f} {dur:<8.2f} {text:<45}")

    comparison_data.append({
        'segment_num': i + 1,
        'gemini_role': gt_display,
        'system_role': our_display,
        'match': match,
        'similarity': sim,
        'duration': dur,
        'text': gt_seg.get('text', '')
    })

# Summary statistics
print("\n" + "=" * 160)
print("ACCURACY SUMMARY")
print("=" * 160)

total = len(gemini_segments)
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
print("ERROR ANALYSIS - MISCLASSIFICATIONS")
print("=" * 160)

print(f"\nShowing segments where system got the role WRONG:")
print(f"\n{'#':<3} {'GT':<10} {'System':<10} {'Sim':<7} {'Dur':<7} {'Text':<50}")
print("-" * 160)

error_count = 0
for data in comparison_data:
    if not data['match']:
        error_count += 1
        sim = data['similarity']
        dur = data['duration']
        text = data['text'][:45]

        print(f"{data['segment_num']:<3} {data['gemini_role']:<10} {data['system_role']:<10} {sim:<7.3f} {dur:<7.2f} {text:<50}")

print(f"\nTotal errors: {error_count}")

# Pattern analysis
print(f"\n" + "=" * 160)
print("PATTERN ANALYSIS - WHY ARE WE GETTING THESE WRONG?")
print("=" * 160)

agent_errors = [d for d in comparison_data if d['gemini_role'] == 'AGENT' and not d['match']]
customer_errors = [d for d in comparison_data if d['gemini_role'] == 'CUSTOMER' and not d['match']]

print(f"\nAGENT segments misclassified as CUSTOMER: {len(agent_errors)}")
if agent_errors:
    print(f"  Average similarity: {np.mean([e['similarity'] for e in agent_errors]):.3f}")
    print(f"  Average duration: {np.mean([e['duration'] for e in agent_errors]):.2f}s")
    print(f"  Typical examples:")
    for i, e in enumerate(agent_errors[:3]):
        print(f"    - '{e['text'][:40]}' (sim={e['similarity']:.3f}, dur={e['duration']:.2f}s)")

print(f"\nCUSTOMER segments misclassified as AGENT: {len(customer_errors)}")
if customer_errors:
    print(f"  Average similarity: {np.mean([e['similarity'] for e in customer_errors]):.3f}")
    print(f"  Average duration: {np.mean([e['duration'] for e in customer_errors]):.2f}s")
    print(f"  Typical examples:")
    for i, e in enumerate(customer_errors[:3]):
        print(f"    - '{e['text'][:40]}' (sim={e['similarity']:.3f}, dur={e['duration']:.2f}s)")

# Root cause analysis
print(f"\n" + "=" * 160)
print("ROOT CAUSE ANALYSIS")
print("=" * 160)

print(f"""
Issues identified:

1. SHORT AGENT PHRASES (< 1 second)
   - These have weak embeddings
   - Often misclassified as CUSTOMER
   - Solution: Temporal voting helps, but need better training

2. CUSTOMER CONFIRMATIONS ("That's correct", "Yes", etc.)
   - Voice characteristic matches agent
   - Often misclassified as AGENT
   - Solution: Better separation in training data

3. WEAK SIMILARITY SCORES (0.15-0.35)
   - System uncertain about role
   - Threshold conflicts
   - Solution: Multi-call training improves similarity scores

4. SHORT TURNS < 0.5 seconds
   - May be skipped or mishandled
   - Need more robust handling
""")

# Recommendations
print(f"\n" + "=" * 160)
print("RECOMMENDATIONS FOR IMPROVEMENT")
print("=" * 160)

print(f"""
1. IMMEDIATE (Code changes):
   [ ] Improve temporal voting window (currently 10s)
   [ ] Add confidence gate for uncertain roles (0.22-0.25 band)
   [ ] Better handling of very short segments (<0.5s)

2. SHORT-TERM (More training data):
   [ ] Train Omar on 3-5 calls instead of 1
   [ ] Ensure diverse call conditions
   [ ] Separate clean agent/customer windows

3. MEDIUM-TERM (Better models):
   [ ] Add ECAPA-TDNN fusion with CAM++
   [ ] Use NeMo MSDD for better boundaries
   [ ] Active learning on misclassified segments

4. VALIDATION:
   [ ] Retrain with multi-call data
   [ ] Test on held-out calls
   [ ] Measure accuracy per-role
   [ ] Verify no regressions on other agents
""")

# Save detailed results
results_file = Path("call_processor/data/training/detailed_role_comparison.json")
results_file.parent.mkdir(parents=True, exist_ok=True)

with open(results_file, 'w') as f:
    json.dump({
        'agent_name': agent_name,
        'overall_accuracy': overall_accuracy,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'per_role_accuracy': accuracy_by_role,
        'agent_errors': len(agent_errors),
        'customer_errors': len(customer_errors),
        'segments': comparison_data
    }, f, indent=2)

print(f"\nDetailed results saved to: {results_file}")

print("\n" + "=" * 160)
print(f"CONCLUSION: {overall_accuracy:.1f}% accuracy on role identification")
print("Focus on fixing: agent role detection (weak similarity), customer role detection (overlap)")
print("=" * 160)
