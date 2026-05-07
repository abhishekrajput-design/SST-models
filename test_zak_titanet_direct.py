#!/usr/bin/env python
"""
Direct test: Use TitaNet embeddings and find optimal threshold for Zak vs Customer
Bypasses diar_multi.py completely - tests pure speaker verification capability
"""

import json
import sys
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.embedding_titanet import get_titanet

print("=" * 130)
print("DIRECT TITANET TEST - Optimal Threshold Search")
print("=" * 130)

GT_FILE = Path("traning_data/zak_raissi/call_01/data.json")
AUDIO_FILE = Path("traning_data/zak_raissi/call_01/audio_16k.wav")
VOICEPRINT_DIR = Path("call_processor/data/agent_voiceprints")

# Load voiceprints
print("\n[Loading TitaNet voiceprints...]")
voiceprints = []
for vp_path in sorted(VOICEPRINT_DIR.glob("zak_local_20260423_titanet_v*.npy")):
    vp = np.load(vp_path)
    voiceprints.append(vp)
    print(f"  Loaded: {vp_path.name} (shape={vp.shape})")

print(f"Total voiceprints: {len(voiceprints)}")

# Load TitaNet
print("\n[Loading TitaNet model...]")
titanet = get_titanet(force_cpu=True)

# Load ground truth and audio
with open(GT_FILE) as f:
    gt = json.load(f)

audio, sr = sf.read(AUDIO_FILE)
if len(audio.shape) > 1:
    audio = audio[:, 0]
print(f"Audio: {len(audio)/sr:.1f}s @ {sr}Hz")

segments = gt['segments']
print(f"Segments: {len(segments)}")

# Compute embeddings for all segments
print("\n[Computing TitaNet embeddings for all segments...]")
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
        # Max similarity across all voiceprints
        sim = max(np.dot(vp, emb) for vp in voiceprints)

    results.append({
        'i': i + 1,
        'gt_speaker': 'AGENT' if speaker == 'agent' else 'CUSTOMER',
        'sim': float(sim),
        'dur': end_s - start_s,
        'text': text[:35]
    })

print(f"Computed {len(results)} segment embeddings")

# Show the distribution
agent_sims = [r['sim'] for r in results if r['gt_speaker'] == 'AGENT']
customer_sims = [r['sim'] for r in results if r['gt_speaker'] == 'CUSTOMER']

print(f"\n[Similarity Distribution]")
print(f"  AGENT  : min={min(agent_sims):.3f}  mean={np.mean(agent_sims):.3f}  max={max(agent_sims):.3f}  std={np.std(agent_sims):.3f}")
print(f"  CUSTOMER: min={min(customer_sims):.3f}  mean={np.mean(customer_sims):.3f}  max={max(customer_sims):.3f}  std={np.std(customer_sims):.3f}")

# Find optimal threshold
print(f"\n[Searching for optimal threshold...]")
best_threshold = 0.5
best_accuracy = 0
for threshold in np.arange(0.30, 0.95, 0.01):
    correct = 0
    for r in results:
        pred = 'AGENT' if r['sim'] >= threshold else 'CUSTOMER'
        if pred == r['gt_speaker']:
            correct += 1
    accuracy = correct / len(results) * 100
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"\n[BEST RESULT]")
print(f"  Optimal threshold: {best_threshold:.3f}")
print(f"  Best accuracy: {best_accuracy:.1f}%")

# Apply best threshold and analyze
correct = 0
agent_correct = 0
agent_total = 0
customer_correct = 0
customer_total = 0
errors = []

for r in results:
    pred = 'AGENT' if r['sim'] >= best_threshold else 'CUSTOMER'
    if pred == r['gt_speaker']:
        correct += 1
        if r['gt_speaker'] == 'AGENT':
            agent_correct += 1
        else:
            customer_correct += 1
    else:
        errors.append({**r, 'pred': pred})

    if r['gt_speaker'] == 'AGENT':
        agent_total += 1
    else:
        customer_total += 1

print(f"\n[Per-Role Accuracy at threshold {best_threshold:.3f}]")
print(f"  AGENT   : {agent_correct}/{agent_total} = {agent_correct/max(1,agent_total)*100:.1f}%")
print(f"  CUSTOMER: {customer_correct}/{customer_total} = {customer_correct/max(1,customer_total)*100:.1f}%")
print(f"  OVERALL : {correct}/{len(results)} = {correct/len(results)*100:.1f}%")

# Show errors
print(f"\n[Errors at optimal threshold]")
for e in errors[:15]:
    print(f"  [{e['i']:2d}] GT={e['gt_speaker']:8s} Pred={e['pred']:8s} sim={e['sim']:.3f} dur={e['dur']:.1f}s | {e['text']}")

print(f"\n" + "=" * 130)
if best_accuracy >= 95:
    print(f"SUCCESS! TitaNet achieves {best_accuracy:.1f}% accuracy")
elif best_accuracy >= 85:
    print(f"GOOD! TitaNet achieves {best_accuracy:.1f}% (significant improvement)")
elif best_accuracy >= 75:
    print(f"BETTER. TitaNet at {best_accuracy:.1f}% (improvement)")
else:
    print(f"INSUFFICIENT. TitaNet at {best_accuracy:.1f}% - need different approach")
print("=" * 130)
