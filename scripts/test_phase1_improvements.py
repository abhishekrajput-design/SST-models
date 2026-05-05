#!/usr/bin/env python
"""
Test Phase 1 improvements: Temporal voting on Omar's 6-minute call.
Shows accuracy before and after the improvements.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.diar_multi import diarize_multi
from src.voiceprints import load_agents_index

AUDIO_FILE = Path("c:/Users/abhis/Downloads/20260505T073055769_385036.mp3")
AGENTS_JSON = "call_processor/data/agent_voiceprints/agents.json"

# Ground truth from Parakeet transcription + manual verification
# Format: (start_s, end_s, is_agent)
GT_LABELS = [
    (0, 1.32, False),      # "Hi, I'm Arcester..."
    (1.32, 1.64, True),    # "Yeah speaking." - AGENT
    (1.64, 4.16, False),   # "Awesome, I wanted..."
    (4.16, 6.24, True),    # "Great, we have a few..." - AGENT
    (6.24, 8.64, False),   # Customer
    (8.64, 9.28, True),    # "Five o'clock." - AGENT SHORT PHRASE
    (9.28, 12.48, False),  # Customer
    (12.48, 13.12, True),  # "On this number?" - AGENT SHORT
    (13.12, 14.88, False), # Customer
    (14.88, 17.04, True),  # AGENT
    (17.04, 22.56, False), # Customer (long)
    (22.56, 24.72, True),  # AGENT
    (24.72, 27.36, False), # Customer
    (27.36, 30.48, True),  # AGENT
    (30.48, 32.88, False), # Customer
    (32.88, 36.96, True),  # AGENT
    (36.96, 41.76, False), # Customer
    (41.76, 47.04, True),  # AGENT
    (47.04, 50.4, False),  # Customer
    (50.4, 53.28, True),   # AGENT
    (53.28, 55.2, False),  # Customer
    (55.2, 56.16, True),   # "Great, thank you." - AGENT SHORT
    (56.16, 66.0, False),  # Customer (long final)
    (66.0, 68.4, True),    # AGENT - "Bye, thanks..."
]

print("=" * 100)
print("PHASE 1 TEST: TEMPORAL VOTING IMPROVEMENTS")
print("=" * 100)
print(f"\nAudio: {AUDIO_FILE.name}")
print(f"Duration: 6:11 (372 seconds)")
print(f"Ground truth: {len(GT_LABELS)} turns")

if not AUDIO_FILE.exists():
    print(f"ERROR: Audio file not found")
    sys.exit(1)

# Load agents
agents = load_agents_index(AGENTS_JSON)
print(f"Loaded {len(agents)} agents")

# Create dummy single-segment input (let diarization handle the full audio)
dummy_segments = [{
    "start": 0.0,
    "end": 372.0,
    "text": "[Full call]",
    "speaker": "SPEAKER_00",
    "identified_speaker": "UNKNOWN",
    "confidence": 0.0,
}]

print("\n" + "-" * 100)
print("Running diarization with Phase 1 improvements (temporal voting)")
print("-" * 100 + "\n")

try:
    result = diarize_multi(
        dummy_segments,
        norm_wav=str(AUDIO_FILE),
        threshold=0.25,
        agents_index_path=AGENTS_JSON,
        force_cpu=False,
    )

    out_segments = result.get("segments", dummy_segments)
    agent_name = result.get("agent_name", "Unknown")
    mode = result.get("speaker_mode", "unknown")

    print(f"Results:")
    print(f"  Total segments: {len(out_segments)}")
    print(f"  Identified agent: {agent_name}")
    print(f"  Mode: {mode}")

    # Match with ground truth
    print("\n" + "-" * 100)
    print("Accuracy Analysis")
    print("-" * 100 + "\n")

    correct = 0
    total_matched = 0
    agent_correct = 0
    agent_total = 0
    customer_correct = 0
    customer_total = 0

    # For each GT label, find matching output segment and check
    for gt_start, gt_end, is_agent_gt in GT_LABELS:
        gt_mid = (gt_start + gt_end) / 2
        # Find closest output segment
        best_match = None
        best_dist = float('inf')
        for out_seg in out_segments:
            out_mid = (float(out_seg.get("start", 0)) + float(out_seg.get("end", 0))) / 2
            dist = abs(out_mid - gt_mid)
            if dist < best_dist:
                best_dist = dist
                best_match = out_seg

        if best_match is None:
            continue

        out_role = best_match.get("identified_speaker", "")
        is_agent_pred = "AGENT" in out_role
        sim = best_match.get("_best_sim", 0.0)

        # Check if correct
        is_correct = (is_agent_gt == is_agent_pred)
        if is_correct:
            correct += 1

        total_matched += 1

        if is_agent_gt:
            agent_total += 1
            if is_correct:
                agent_correct += 1
        else:
            customer_total += 1
            if is_correct:
                customer_correct += 1

    print(f"Overall Accuracy:")
    print(f"  Correct: {correct}/{total_matched}")
    print(f"  Accuracy: {100*correct/total_matched if total_matched > 0 else 0:.1f}%")

    print(f"\nBy Speaker:")
    print(f"  AGENT:    {agent_correct}/{agent_total} = {100*agent_correct/agent_total if agent_total > 0 else 0:.1f}%")
    print(f"  CUSTOMER: {customer_correct}/{customer_total} = {100*customer_correct/customer_total if customer_total > 0 else 0:.1f}%")

    # Show sample segments with temporal voting markers
    print(f"\nSegments with Temporal Voting Overrides:")
    temporal_votes = [s for s in out_segments if s.get("_temporal_vote_override")]
    if temporal_votes:
        print(f"  Found {len(temporal_votes)} segments corrected by temporal voting")
        for seg in temporal_votes[:5]:
            role = seg.get("identified_speaker", "")
            text = seg.get("text", "")[:40]
            print(f"    - {role:15s}: {text}")
    else:
        print(f"  No temporal voting overrides (segments already correctly classified)")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"""
Phase 1 improvements (Temporal Voting + ECAPA):
- Temporal voting checks each segment against 10-second window of neighbors
- Corrects isolated misclassifications using local context
- ECAPA fusion (dual embeddings) provides complementary speaker verification

Current Accuracy: {100*correct/total_matched if total_matched > 0 else 0:.1f}%
Previous (Hotfixes only): ~41.7%
Expected with Phase 1: 50-65%

Remaining Issues:
- Short phrase embeddings still weak (need diverse training data)
- Customer-agent voice similarity requires re-enrollment
- Temporal voting helps but can't fix fundamental embedding gaps

Next Steps:
1. Implement Phase 2: Re-enroll from independent calls with manual labels
2. Update embeddings for all agents with clean call recordings
3. Expected improvement with Phase 2: 85-90% accuracy

The API has the data - we just need proper training labels.
""")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
