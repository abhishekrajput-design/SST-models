#!/usr/bin/env python
"""
Two-speaker clustering approach:
1. Embed all segments with TitaNet
2. Cluster into 2 groups (2 speakers in call)
3. Assign agent role to cluster CLOSEST to Zak's voiceprint
4. Other cluster = customer

This bypasses per-segment matching - relies on cluster separation
"""

import json
import sys
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.embedding_titanet import get_titanet

print("=" * 130)
print("TWO-SPEAKER CLUSTERING TEST")
print("=" * 130)

GT_FILE = Path("traning_data/zak_raissi/call_01/data.json")
AUDIO_FILE = Path("traning_data/zak_raissi/call_01/audio_16k.wav")
VOICEPRINT_DIR = Path("call_processor/data/agent_voiceprints")

# Load voiceprints
voiceprints = []
for vp_path in sorted(VOICEPRINT_DIR.glob("zak_local_20260423_titanet_v*.npy")):
    vp = np.load(vp_path)
    voiceprints.append(vp)
print(f"Loaded {len(voiceprints)} TitaNet voiceprints")

# Load TitaNet
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
print("\n[Computing TitaNet embeddings...]")
embeddings = []
seg_data = []
for i, seg in enumerate(segments):
    start_s = float(seg['start'])
    end_s = float(seg['end'])
    speaker = seg.get('speaker', '').lower()
    text = seg.get('text', '')
    dur = end_s - start_s

    start_samp = int(start_s * sr)
    end_samp = int(end_s * sr)
    if end_samp > len(audio):
        continue
    window = audio[start_samp:end_samp]

    emb = titanet.embed_chunk(window, sr=sr)
    if emb is None:
        emb = np.zeros(192, dtype=np.float32)
    else:
        emb = emb / max(np.linalg.norm(emb), 1e-8)

    embeddings.append(emb)
    seg_data.append({
        'i': i + 1,
        'gt': 'AGENT' if speaker == 'agent' else 'CUSTOMER',
        'dur': dur,
        'text': text[:35]
    })

embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# Try multiple clustering approaches
def evaluate_clustering(labels, seg_data, embeddings, voiceprints, method_name):
    # Compute cluster centroids
    n_clusters = len(set(labels))
    centroids = []
    for k in range(n_clusters):
        cluster_embs = embeddings[labels == k]
        if len(cluster_embs) > 0:
            c = np.mean(cluster_embs, axis=0)
            c = c / max(np.linalg.norm(c), 1e-8)
            centroids.append(c)

    # Find which cluster is closest to Zak's voiceprint (max sim across all VPs)
    cluster_to_zak_sim = []
    for c in centroids:
        max_sim = max(np.dot(vp, c) for vp in voiceprints)
        cluster_to_zak_sim.append(max_sim)

    agent_cluster = int(np.argmax(cluster_to_zak_sim))

    # Predict labels
    correct = 0
    agent_correct = 0
    agent_total = 0
    customer_correct = 0
    customer_total = 0

    for idx, seg in enumerate(seg_data):
        pred = 'AGENT' if labels[idx] == agent_cluster else 'CUSTOMER'
        if pred == seg['gt']:
            correct += 1
            if seg['gt'] == 'AGENT':
                agent_correct += 1
            else:
                customer_correct += 1
        if seg['gt'] == 'AGENT':
            agent_total += 1
        else:
            customer_total += 1

    accuracy = correct / len(seg_data) * 100
    print(f"\n[{method_name}]")
    print(f"  Cluster sim to Zak: {[f'{s:.3f}' for s in cluster_to_zak_sim]}")
    print(f"  Agent cluster: {agent_cluster}")
    print(f"  AGENT accuracy: {agent_correct}/{agent_total} = {agent_correct/max(1,agent_total)*100:.1f}%")
    print(f"  CUSTOMER accuracy: {customer_correct}/{customer_total} = {customer_correct/max(1,customer_total)*100:.1f}%")
    print(f"  OVERALL: {correct}/{len(seg_data)} = {accuracy:.1f}%")
    return accuracy

# Method 1: KMeans 2-speaker
print("\n" + "=" * 130)
print("METHOD COMPARISON")
print("=" * 130)

km2 = KMeans(n_clusters=2, random_state=42, n_init=20)
labels_km2 = km2.fit_predict(embeddings)
acc_km2 = evaluate_clustering(labels_km2, seg_data, embeddings, voiceprints, "KMeans (2 clusters)")

# Method 2: Agglomerative clustering
agg2 = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels_agg2 = agg2.fit_predict(embeddings)
acc_agg2 = evaluate_clustering(labels_agg2, seg_data, embeddings, voiceprints, "Agglomerative Ward (2 clusters)")

# Method 3: KMeans 3 clusters (then merge non-agent)
km3 = KMeans(n_clusters=3, random_state=42, n_init=20)
labels_km3 = km3.fit_predict(embeddings)
# Find which cluster has highest similarity to Zak
n3 = 3
centroids3 = []
for k in range(n3):
    c = np.mean(embeddings[labels_km3 == k], axis=0)
    c = c / max(np.linalg.norm(c), 1e-8)
    centroids3.append(c)
cluster_sims3 = [max(np.dot(vp, c) for vp in voiceprints) for c in centroids3]
agent_cluster3 = int(np.argmax(cluster_sims3))
# Merge other clusters into customer
labels_km3_binary = np.array([0 if l == agent_cluster3 else 1 for l in labels_km3])
print(f"\n[KMeans (3 clusters → binary, agent was cluster {agent_cluster3})]")
print(f"  Cluster sims: {[f'{s:.3f}' for s in cluster_sims3]}")

correct = 0; agent_correct = 0; agent_total = 0; customer_correct = 0; customer_total = 0
for idx, seg in enumerate(seg_data):
    pred = 'AGENT' if labels_km3_binary[idx] == 0 else 'CUSTOMER'
    if pred == seg['gt']:
        correct += 1
        if seg['gt'] == 'AGENT': agent_correct += 1
        else: customer_correct += 1
    if seg['gt'] == 'AGENT': agent_total += 1
    else: customer_total += 1

acc_km3 = correct / len(seg_data) * 100
print(f"  AGENT accuracy: {agent_correct}/{agent_total} = {agent_correct/max(1,agent_total)*100:.1f}%")
print(f"  CUSTOMER accuracy: {customer_correct}/{customer_total} = {customer_correct/max(1,customer_total)*100:.1f}%")
print(f"  OVERALL: {correct}/{len(seg_data)} = {acc_km3:.1f}%")

# Best result
print(f"\n" + "=" * 130)
print("BEST RESULTS SUMMARY")
print("=" * 130)
print(f"  KMeans (2 clusters):       {acc_km2:.1f}%")
print(f"  Agglomerative Ward:        {acc_agg2:.1f}%")
print(f"  KMeans (3→binary):         {acc_km3:.1f}%")
best = max(acc_km2, acc_agg2, acc_km3)
print(f"\n  BEST: {best:.1f}%")
if best >= 95:
    print("  SUCCESS! Achieved 95%+ accuracy")
elif best >= 85:
    print("  GOOD! Significant improvement over voiceprint-only")
elif best >= 75:
    print("  BETTER. Improvement made")
else:
    print("  Still below target - voices too similar acoustically")
print("=" * 130)
