#!/usr/bin/env python
"""Test hotfixes by re-processing Omar's Mini Hatch call."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "call_processor")

from src.diar_multi import diarize_multi
from src.voiceprints import load_agents_index

# Paths
call_dir = Path("call_processor/data/processed/enhanced_20260505T073055769_385036__parakeet-tdt-0.6b-v3")
result_path = call_dir / "result.json"
audio_path = call_dir / "trimmed_audio.mp3"

# Load agents index
agents = load_agents_index("call_processor/data/agent_voiceprints/agents.json")
omar_data = agents.get("omar_el_harchaoui", {})

print(f"Omar voiceprints loaded: {len(omar_data.get('voiceprints', []))} buckets")
print(f"  mean_inside_sim: {omar_data.get('mean_inside_sim', 'N/A')}")
print(f"  max_outside_sim: {omar_data.get('max_outside_sim', 'N/A')}")

# Load result.json with ASR segments
with open(result_path) as f:
    result = json.load(f)

segments = result.get("segments", [])
print(f"\nProcessing {len(segments)} ASR segments...")

# Run diarizer with new enrollment
print("\nCalling diarize_multi...")
diar_info = diarize_multi(
    segments,
    norm_wav=str(audio_path),
    threshold=0.25,
    agents_index_path="call_processor/data/agent_voiceprints/agents.json",
    force_cpu=False,
)

out_segments = diar_info.get("segments", segments)

print(f"Output: {len(out_segments)} segments")
print(f"\nDiarization info keys: {list(diar_info.keys())}")
print(f"  speaker_mode: {diar_info.get('speaker_mode')}")
print(f"  agent_slug: {diar_info.get('agent_slug')}")
print(f"  num_agent: {diar_info.get('num_agent')}")

# Compute accuracy vs ground truth
GT_TURNS = [
    ("CUSTOMER", "Hi, I'm Arcester. Is this a good time to chat?"),
    ("AGENT", "Yeah speaking."),
    ("CUSTOMER", "Awesome, I wanted to discuss the Mini Hatch you have available."),
    ("AGENT", "Great, we have a few on the lot. When would you like to view it?"),
    ("CUSTOMER", "How about tomorrow morning around ten?"),
    ("AGENT", "Five o'clock."),
    ("CUSTOMER", "That's correct. I mean, that works for me if five is better."),
    ("AGENT", "On this number?"),
    ("CUSTOMER", "Yeah, that's the best way to reach me."),
    ("AGENT", "Okay. Can I get your email as well?"),
    ("CUSTOMER", "Sure, it's mark.j.stewart at gmail dot com."),
    ("AGENT", "Perfect. I have that down here."),
    ("CUSTOMER", "Great. And what's the mileage on that Mini?"),
    ("AGENT", "It has 45000 miles on the clock, mint condition."),
    ("CUSTOMER", "That's fantastic. What's the asking price?"),
    ("AGENT", "We're asking eighteen-five for it. That's well below market."),
    ("CUSTOMER", "Hmm, that's a bit more than I was hoping to spend. Can you do any better?"),
    ("AGENT", "Let me see what I can do. We might have some flexibility if you're ready to move quickly."),
    ("CUSTOMER", "I'm ready to view it first and then we can talk numbers."),
    ("AGENT", "Perfect. I'll have it ready for you at five tomorrow."),
    ("CUSTOMER", "Sounds good. See you then."),
    ("AGENT", "Great, thank you."),
    ("CUSTOMER", "Okay, that's fine. I'll be there at five PM tomorrow. Looking forward to it. Bye."),
    ("AGENT", "Bye, thanks for calling Car Planet."),
]

# Match output segments to GT turns by text
correct = 0
total = 0
errors = []

print(f"\n\nAccuracy vs Ground Truth ({len(GT_TURNS)} turns):")
print("-" * 80)

for gt_role, gt_text in GT_TURNS:
    # Find matching output segment
    best_match = None
    best_score = 0
    for seg in out_segments:
        seg_text = seg.get("text", "").strip()
        if seg_text and gt_text.lower() in seg_text.lower():
            # Simple text match
            best_match = seg
            break

    if best_match:
        sys_role = "AGENT" if best_match.get("identified_speaker") == "AGENT" else "CUSTOMER"
        sim = best_match.get("_best_sim", 0.0)
        emb_failed = best_match.get("_emb_failed", False)

        match = "OK" if sys_role == gt_role else "NG"
        if sys_role == gt_role:
            correct += 1
        else:
            errors.append((gt_text[:40], gt_role, sys_role, sim, emb_failed))

        total += 1
        print(f"{match} {gt_role:8s} vs {sys_role:8s}  sim={sim:6.3f}  _emb_failed={emb_failed}  {gt_text[:50]}")

print("-" * 80)
print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

if errors:
    print(f"\nErrors ({len(errors)}):")
    for text, gt, sys, sim, emb_failed in errors:
        marker = "[EMB FAILED]" if emb_failed else ""
        print(f"  {text:40s}  GT={gt:8s} SYS={sys:8s}  sim={sim:6.3f} {marker}")
