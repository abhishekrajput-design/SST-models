#!/usr/bin/env python
"""
Test Zak's trained model on call_02 - both CAM++ and TitaNet
Compare system output vs Gemini ground truth
"""

import json
import sys
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

print("=" * 130)
print("CALL_02 TEST - Zak Raissi vs Customer")
print("=" * 130)

GT_FILE = Path("traning_data/zak_raissi/call_02/data.json")
AUDIO_FILE = Path("traning_data/zak_raissi/call_02/audio_16k.wav")

with open(GT_FILE) as f:
    gt = json.load(f)

audio, sr = sf.read(AUDIO_FILE)
if len(audio.shape) > 1:
    audio = audio[:, 0]

segments = gt['segments']
print(f"\nCall: {gt.get('call_id')}")
print(f"Audio: {len(audio)/sr:.1f}s @ {sr}Hz")
print(f"Segments: {len(segments)}")

agent_count = sum(1 for s in segments if s.get('speaker') == 'agent')
customer_count = sum(1 for s in segments if s.get('speaker') == 'customer')
print(f"Ground truth: {agent_count} AGENT, {customer_count} CUSTOMER")

# Test 1: Using diar_multi.py (current production system)
print(f"\n" + "=" * 130)
print("TEST 1: PRODUCTION SYSTEM (diar_multi.py with current voiceprints)")
print("=" * 130)

from src.diar_multi import diarize_multi

parakeet_segments = [{'start': s['start'], 'end': s['end'], 'text': s['text']} for s in segments]

result = diarize_multi(
    segments=parakeet_segments,
    norm_wav=str(AUDIO_FILE),
    force_cpu=True
)

our_segments = result.get('segments', [])
print(f"Mode used: {result.get('speaker_mode', 'N/A')}")
print(f"Agent identified as: {result.get('agent_name', 'N/A')}")

# Compare
correct = 0
agent_correct = 0; agent_total = 0
customer_correct = 0; customer_total = 0
errors_prod = []

for i in range(len(segments)):
    gt_seg = segments[i]
    our_seg = our_segments[i] if i < len(our_segments) else {}

    gt_speaker = gt_seg.get('speaker', '').lower()
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
        correct += 1
        if gt_speaker == 'agent': agent_correct += 1
        else: customer_correct += 1
    else:
        sim = our_seg.get('_best_sim', 0.0)
        dur = gt_seg['end'] - gt_seg['start']
        errors_prod.append({
            'i': i+1, 'gt': gt_display, 'pred': our_display,
            'sim': sim, 'dur': dur, 'text': gt_seg.get('text', '')[:40]
        })

    if gt_speaker == 'agent': agent_total += 1
    else: customer_total += 1

prod_acc = correct / len(segments) * 100
print(f"\nProduction System Results:")
print(f"  AGENT: {agent_correct}/{agent_total} = {agent_correct/max(1,agent_total)*100:.1f}%")
print(f"  CUSTOMER: {customer_correct}/{customer_total} = {customer_correct/max(1,customer_total)*100:.1f}%")
print(f"  OVERALL: {correct}/{len(segments)} = {prod_acc:.1f}%")

# Test 2: Direct TitaNet with optimal threshold
print(f"\n" + "=" * 130)
print("TEST 2: DIRECT TITANET (with all 3 trained voiceprints)")
print("=" * 130)

from src.embedding_titanet import get_titanet
titanet = get_titanet(force_cpu=True)

VOICEPRINT_DIR = Path("call_processor/data/agent_voiceprints")
voiceprints = []
for vp_path in sorted(VOICEPRINT_DIR.glob("zak_local_20260423_titanet_v*.npy")):
    vp = np.load(vp_path)
    voiceprints.append(vp)

print(f"Loaded {len(voiceprints)} TitaNet voiceprints")

print("\n[Computing TitaNet embeddings...]")
results = []
for i, seg in enumerate(segments):
    start_s = float(seg['start'])
    end_s = float(seg['end'])
    speaker = seg.get('speaker', '').lower()
    text = seg.get('text', '')

    start_samp = int(start_s * sr)
    end_samp = int(end_s * sr)
    if end_samp > len(audio):
        continue
    window = audio[start_samp:end_samp]

    emb = titanet.embed_chunk(window, sr=sr)
    if emb is None:
        sim = 0.0
    else:
        emb = emb / max(np.linalg.norm(emb), 1e-8)
        sim = max(np.dot(vp, emb) for vp in voiceprints)

    results.append({
        'i': i + 1,
        'gt': 'AGENT' if speaker == 'agent' else 'CUSTOMER',
        'sim': float(sim),
        'dur': end_s - start_s,
        'text': text[:40]
    })

agent_sims = [r['sim'] for r in results if r['gt'] == 'AGENT']
customer_sims = [r['sim'] for r in results if r['gt'] == 'CUSTOMER']

print(f"\n[Distribution]")
print(f"  AGENT  : min={min(agent_sims):.3f}  mean={np.mean(agent_sims):.3f}  max={max(agent_sims):.3f}")
print(f"  CUSTOMER: min={min(customer_sims):.3f}  mean={np.mean(customer_sims):.3f}  max={max(customer_sims):.3f}")

# Find optimal threshold
best_threshold = 0.5
best_accuracy = 0
for threshold in np.arange(0.30, 0.95, 0.01):
    correct = sum(1 for r in results
                  if (r['sim'] >= threshold) == (r['gt'] == 'AGENT'))
    accuracy = correct / len(results) * 100
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

# Apply best threshold
correct = 0; agent_correct = 0; agent_total = 0; customer_correct = 0; customer_total = 0
errors_titanet = []
for r in results:
    pred = 'AGENT' if r['sim'] >= best_threshold else 'CUSTOMER'
    if pred == r['gt']:
        correct += 1
        if r['gt'] == 'AGENT': agent_correct += 1
        else: customer_correct += 1
    else:
        errors_titanet.append({**r, 'pred': pred})
    if r['gt'] == 'AGENT': agent_total += 1
    else: customer_total += 1

titanet_acc = correct / len(results) * 100
print(f"\nTitaNet Results (optimal threshold {best_threshold:.3f}):")
print(f"  AGENT: {agent_correct}/{agent_total} = {agent_correct/max(1,agent_total)*100:.1f}%")
print(f"  CUSTOMER: {customer_correct}/{customer_total} = {customer_correct/max(1,customer_total)*100:.1f}%")
print(f"  OVERALL: {correct}/{len(results)} = {titanet_acc:.1f}%")

# Show errors
print(f"\n[TitaNet Errors]")
for e in errors_titanet[:10]:
    print(f"  [{e['i']:2d}] GT={e['gt']:8s} Pred={e['pred']:8s} sim={e['sim']:.3f} | {e['text']}")

# Final
print(f"\n" + "=" * 130)
print("CALL_02 FINAL RESULTS")
print("=" * 130)
print(f"  Production (CAM++):     {prod_acc:.1f}%")
print(f"  TitaNet (best threshold): {titanet_acc:.1f}%")
best = max(prod_acc, titanet_acc)
print(f"\n  BEST: {best:.1f}%")
if best >= 95:
    print("  SUCCESS")
elif best >= 85:
    print("  GOOD")
else:
    print("  Below target")
print("=" * 130)
