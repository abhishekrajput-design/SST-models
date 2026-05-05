#!/usr/bin/env python
"""Test hotfixes by re-processing Omar's Mini Hatch call — improved version."""
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
print(f"  mean_inside_sim: {omar_data.get('mean_inside_sim', 'N/A'):.4f}")
print(f"  max_outside_sim: {omar_data.get('max_outside_sim', 'N/A'):.4f}")

# Load result.json with ASR segments
with open(result_path) as f:
    result = json.load(f)

segments = result.get("segments", [])
print(f"\nProcessing {len(segments)} ASR segments...")

# Run diarizer with new enrollment
print("Calling diarize_multi...")
diar_info = diarize_multi(
    segments,
    norm_wav=str(audio_path),
    threshold=0.25,
    agents_index_path="call_processor/data/agent_voiceprints/agents.json",
    force_cpu=False,
)

out_segments = diar_info.get("segments", segments)
print(f"Output: {len(out_segments)} segments\n")

# Count AGENT vs CUSTOMER
agent_count = sum(1 for s in out_segments if s.get("identified_speaker") == "AGENT")
customer_count = len(out_segments) - agent_count
print(f"Role distribution:")
print(f"  AGENT: {agent_count}")
print(f"  CUSTOMER: {customer_count}")

# Key checks for hotfixes
print(f"\n\nKey segments for hotfix validation:")
print("-" * 100)

key_checks = {
    "Yeah speaking.": ("AGENT", 0.731, False, "Hotfix A: short AGENT phrase, was demoted in pass 3"),
    "Five o'clock.": ("AGENT", 0.685, False, "Hotfix A: short AGENT phrase, was demoted in pass 3"),
    "On this number?": ("AGENT", 0.600, True, "Hotfix A: short AGENT, emb_failed, was demoted"),
    "Okay. Can I get your email as well?": ("AGENT", 0.700, False, "longer AGENT phrase"),
    "That's correct": ("CUSTOMER", 0.670, False, "Hotfix B: customer confirmation, was false AGENT"),
    "Perfect. I have that down here.": ("AGENT", 0.700, False, "AGENT confirmation"),
}

for i, seg in enumerate(out_segments):
    text = seg.get("text", "").strip()
    role = seg.get("identified_speaker", "UNKNOWN")
    sim = seg.get("_best_sim", 0.0)
    emb_failed = seg.get("_emb_failed", False)

    for check_text, (gt_role, expected_sim, expected_emb_failed, reason) in key_checks.items():
        if check_text.lower() in text.lower():
            match = "PASS" if role == gt_role else "FAIL"
            print(f"{match:4s}  {gt_role:8s}  SYS={role:8s}  sim={sim:6.3f}  emb_failed={emb_failed}")
            print(f"      Text: {text[:60]:60s}")
            print(f"      Reason: {reason}")
            print()
            break

print("\n\nShort AGENT phrases with emb_failed flag (should now be protected by Hotfix A):")
print("-" * 100)
for seg in out_segments:
    if seg.get("_emb_failed"):
        role = seg.get("identified_speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        sim = seg.get("_best_sim", 0.0)
        dur = seg.get("end", 0) - seg.get("start", 0)
        print(f"  role={role:8s}  dur={dur:5.2f}s  sim={sim:6.3f}  text={text[:50]}")

# Full accuracy (simple text matching)
print("\n\nFull Ground Truth Comparison:")
print("-" * 100)

GT_TURNS = [
    ("CUSTOMER", "Hi, I'm"),
    ("AGENT", "Yeah speaking"),
    ("CUSTOMER", "Mini Hatch"),
    ("AGENT", "have a few"),
    ("CUSTOMER", "tomorrow morning"),
    ("AGENT", "Five o'clock"),
    ("CUSTOMER", "That's correct"),
    ("AGENT", "On this number"),
    ("CUSTOMER", "best way"),
    ("AGENT", "get your email"),
    ("CUSTOMER", "mark.j.stewart"),
    ("AGENT", "Perfect. I have"),
    ("CUSTOMER", "mileage"),
    ("AGENT", "45000 miles"),
    ("CUSTOMER", "asking price"),
    ("AGENT", "eighteen-five"),
    ("CUSTOMER", "bit more"),
    ("AGENT", "flexibility"),
    ("CUSTOMER", "view it first"),
    ("AGENT", "ready at five"),
    ("CUSTOMER", "Sounds good"),
    ("AGENT", "thank you"),
    ("CUSTOMER", "five PM"),
    ("AGENT", "thanks for calling"),
]

correct = 0
total = 0

for gt_role, gt_snippet in GT_TURNS:
    best_match = None
    for seg in out_segments:
        if gt_snippet.lower() in seg.get("text", "").lower():
            best_match = seg
            break

    if best_match:
        sys_role = best_match.get("identified_speaker", "UNKNOWN")
        if sys_role.startswith("AGENT"):
            sys_role = "AGENT"
        elif sys_role == "Customer 1":
            sys_role = "CUSTOMER"

        match = "OK" if sys_role == gt_role else "NG"
        if sys_role == gt_role:
            correct += 1
        total += 1

print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
print(f"\nExpected post-hotfix: >= 80% (was 38.6% before)")
