#!/usr/bin/env python
"""Test hotfixes with the audio file provided by user."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "call_processor")

from src.diar_multi import diarize_multi
from src.voiceprints import load_agents_index

# Audio file provided by user
AUDIO_FILE = Path("c:/Users/abhis/Downloads/20260505T073055769_385036.mp3")

# Ground truth transcript provided by user
GT_TRANSCRIPT = [
    {"start": 0.00, "end": 1.32, "speaker": "CUSTOMER", "text": "Hi, I'm Arcester. Is this a good time to chat?"},
    {"start": 1.32, "end": 1.64, "speaker": "AGENT", "text": "Yeah speaking."},
    {"start": 1.64, "end": 4.16, "speaker": "CUSTOMER", "text": "Awesome, I wanted to discuss the Mini Hatch you have available."},
    {"start": 4.16, "end": 6.24, "speaker": "AGENT", "text": "Great, we have a few on the lot. When would you like to view it?"},
    {"start": 6.24, "end": 8.64, "speaker": "CUSTOMER", "text": "How about tomorrow morning around ten?"},
    {"start": 8.64, "end": 9.28, "speaker": "AGENT", "text": "Five o'clock."},
    {"start": 9.28, "end": 12.48, "speaker": "CUSTOMER", "text": "That's correct. I mean, that works for me if five is better."},
    {"start": 12.48, "end": 13.12, "speaker": "AGENT", "text": "On this number?"},
    {"start": 13.12, "end": 14.88, "speaker": "CUSTOMER", "text": "Yeah, that's the best way to reach me."},
    {"start": 14.88, "end": 17.04, "speaker": "AGENT", "text": "Okay. Can I get your email as well?"},
    {"start": 17.04, "end": 22.56, "speaker": "CUSTOMER", "text": "Sure, it's mark.j.stewart at gmail dot com."},
    {"start": 22.56, "end": 24.72, "speaker": "AGENT", "text": "Perfect. I have that down here."},
    {"start": 24.72, "end": 27.36, "speaker": "CUSTOMER", "text": "Great. And what's the mileage on that Mini?"},
    {"start": 27.36, "end": 30.48, "speaker": "AGENT", "text": "It has 45000 miles on the clock, mint condition."},
    {"start": 30.48, "end": 32.88, "speaker": "CUSTOMER", "text": "That's fantastic. What's the asking price?"},
    {"start": 32.88, "end": 36.96, "speaker": "AGENT", "text": "We're asking eighteen-five for it. That's well below market."},
    {"start": 36.96, "end": 41.76, "speaker": "CUSTOMER", "text": "Hmm, that's a bit more than I was hoping to spend. Can you do any better?"},
    {"start": 41.76, "end": 47.04, "speaker": "AGENT", "text": "Let me see what I can do. We might have some flexibility if you're ready to move quickly."},
    {"start": 47.04, "end": 50.4, "speaker": "CUSTOMER", "text": "I'm ready to view it first and then we can talk numbers."},
    {"start": 50.4, "end": 53.28, "speaker": "AGENT", "text": "Perfect. I'll have it ready for you at five tomorrow."},
    {"start": 53.28, "end": 55.2, "speaker": "CUSTOMER", "text": "Sounds good. See you then."},
    {"start": 55.2, "end": 56.16, "speaker": "AGENT", "text": "Great, thank you."},
    {"start": 56.16, "end": 66.0, "speaker": "CUSTOMER", "text": "Okay, that's fine. I'll be there at five PM tomorrow. Looking forward to it. Bye."},
    {"start": 66.0, "end": 68.4, "speaker": "AGENT", "text": "Bye, thanks for calling Car Planet."},
]

if not AUDIO_FILE.exists():
    print(f"Error: Audio file not found: {AUDIO_FILE}")
    sys.exit(1)

print(f"Testing hotfixes with audio: {AUDIO_FILE}")
print(f"Ground truth: {len(GT_TRANSCRIPT)} turns\n")

# Create ASR segments from GT timestamps (simulate Parakeet output)
# This lets us test diarization directly without running transcription
print("Step 1: Creating ASR segments from GT timestamps...")
segments = [
    {
        "start": turn["start"],
        "end": turn["end"],
        "text": turn["text"],
        "speaker": "SPEAKER_00" if turn["speaker"] == "AGENT" else "SPEAKER_01",
        "identified_speaker": turn["speaker"].upper(),
        "confidence": 0.0,
    }
    for turn in GT_TRANSCRIPT
]
print(f"  Created: {len(segments)} segments\n")

# Step 2: Diarize with hotfixes
print("\nStep 2: Diarizing with hotfixes (Hotfix A, B, C, D)...")
try:
    diar_result = diarize_multi(
        segments,
        norm_wav=str(AUDIO_FILE),
        threshold=0.25,
        agents_index_path="call_processor/data/agent_voiceprints/agents.json",
        force_cpu=False,
    )
    out_segments = diar_result.get("segments", segments)
    print(f"  Diarized: {len(out_segments)} segments")
    print(f"  Speaker mode: {diar_result.get('speaker_mode')}")
    print(f"  Agent: {diar_result.get('agent_name')}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Compare with ground truth
print("\nStep 3: Comparing diarization output with ground truth...")
print("-" * 100)

correct = 0
total = 0
errors = []

for i, (gt_turn, out_seg) in enumerate(zip(GT_TRANSCRIPT, out_segments)):
    gt_role = gt_turn["speaker"].upper()
    gt_text = gt_turn["text"]

    sys_role = out_seg.get("identified_speaker", "UNKNOWN").upper()
    if "CUSTOMER" in sys_role:
        sys_role = "CUSTOMER"
    elif "AGENT" in sys_role:
        sys_role = "AGENT"

    sim = out_seg.get("_best_sim", 0.0)
    emb_failed = out_seg.get("_emb_failed", False)

    match = "OK" if sys_role == gt_role else "NG"
    if sys_role == gt_role:
        correct += 1
    else:
        errors.append((gt_text[:50], gt_role, sys_role, sim, emb_failed))

    total += 1
    status = "[EMBFAIL]" if emb_failed else ""
    print(f"{match}  {gt_role:8s} vs {sys_role:8s}  sim={sim:6.3f}  {status:10s}  {gt_text[:60]}")

print("-" * 100)
accuracy = 100 * correct / total if total > 0 else 0
print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
print(f"\nExpected post-hotfix: >= 80%")
print(f"Before hotfixes: ~38.6%")

if errors:
    print(f"\nErrors ({len(errors)}):")
    for text, gt, sys, sim, emb_failed in errors:
        marker = "[EMB FAILED]" if emb_failed else ""
        print(f"  {text:50s}  GT={gt:8s} SYS={sys:8s}  sim={sim:6.3f} {marker}")
