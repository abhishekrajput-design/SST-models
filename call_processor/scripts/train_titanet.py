#!/usr/bin/env python
"""
Train Voiceprint using NVIDIA TitaNet-Large
State-of-the-art speaker verification model (better than CAM++)

EER on VoxCeleb1: 0.66% (vs CAM++ 0.91%)
Trained on Fisher + Switchboard + VoxCeleb + LibriSpeech (includes phone audio!)

Usage:
  python train_titanet.py "Agent Name" "agent_db_key"
"""

import json
import sys
import time
import warnings
import argparse
from pathlib import Path
import glob
import shutil

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.embedding_titanet import get_titanet

print("=" * 130)
print("TITANET-LARGE TRAINING - State-of-the-art Speaker Verification")
print("=" * 130)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_name', help='Display name (e.g., "Zak Raissi")')
    parser.add_argument('agent_key', help='Database key (e.g., "zak_local_20260423")')
    parser.add_argument('--min-dur', type=float, default=2.0, help='Min segment duration')
    parser.add_argument('--n-clusters', type=int, default=3, help='Number of voiceprint clusters')
    return parser.parse_args()


def load_audio(audio_path: str, target_sr: int = 16000):
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        except Exception:
            pass
    return audio.astype(np.float32), target_sr


def main():
    args = parse_args()
    AGENT_NAME = args.agent_name
    AGENT_KEY = args.agent_key

    print(f"\nAgent: {AGENT_NAME}")
    print(f"Database key: {AGENT_KEY}")
    print(f"Min duration: {args.min_dur}s, Clusters: {args.n_clusters}")

    # Step 1: Load TitaNet
    print(f"\n[STEP 1] Loading TitaNet-Large model...")
    print("-" * 130)
    titanet = get_titanet(force_cpu=True)
    print(f"Loaded: {titanet.model_name} (dim={titanet.dim})")

    # Step 2: Find Gemini training data
    print(f"\n[STEP 2] Finding Gemini training data...")
    print("-" * 130)

    training_dir = Path("call_processor/data/training")
    label_files = []
    for label_file in sorted(training_dir.glob("gemini_labels_*.json")):
        try:
            with open(label_file) as f:
                data = json.load(f)
            file_agent = (data.get('agent_name') or '').lower().replace(' ', '_')
            if file_agent == AGENT_KEY.lower() or AGENT_KEY.lower() in label_file.name.lower():
                label_files.append({'file': label_file, 'data': data})
                print(f"  Found: {label_file.name} ({len(data.get('segments', []))} segments)")
        except Exception:
            pass

    if not label_files:
        print(f"ERROR: No Gemini labels found")
        sys.exit(1)

    # Step 3: Load audio
    print(f"\n[STEP 3] Loading audio files...")
    print("-" * 130)

    audio_data = {}
    for entry in label_files:
        call_id = entry['data'].get('call_id', '')
        candidates = (
            glob.glob(f"call_processor/data/processed/**/*{call_id}*/*.wav", recursive=True) +
            glob.glob(f"call_processor/data/processed/{call_id}*/*.wav", recursive=True) +
            glob.glob(f"traning_data/**/*{call_id}*.mp3", recursive=True) +
            glob.glob(f"traning_data/**/*{call_id}*.wav", recursive=True)
        )
        if candidates:
            audio_path = candidates[0]
            try:
                audio, sr = load_audio(audio_path)
                audio_data[call_id] = audio
                print(f"  [{call_id}] Loaded {len(audio)/sr:.1f}s from {Path(audio_path).name}")
            except Exception as e:
                print(f"  [{call_id}] ERROR: {e}")
        else:
            print(f"  [{call_id}] AUDIO NOT FOUND")

    # Step 4: Extract embeddings
    print(f"\n[STEP 4] Extracting TitaNet embeddings...")
    print("-" * 130)

    agent_embs = []
    customer_embs = []

    for entry in label_files:
        call_id = entry['data'].get('call_id', '')
        if call_id not in audio_data:
            continue

        audio = audio_data[call_id]
        sr = 16000
        segments = entry['data'].get('segments', [])

        agent_count_call = 0
        customer_count_call = 0

        for seg in segments:
            speaker = seg.get('speaker', '').lower()
            start_s = float(seg.get('start', 0))
            end_s = float(seg.get('end', 0))
            dur = end_s - start_s

            if dur < args.min_dur:
                continue

            start_samp = int(start_s * sr)
            end_samp = int(end_s * sr)
            if end_samp > len(audio):
                continue

            window = audio[start_samp:end_samp]
            try:
                emb = titanet.embed_chunk(window, sr=sr)
                if emb is not None and not np.isnan(emb).any():
                    if speaker == 'agent':
                        agent_embs.append(emb)
                        agent_count_call += 1
                    elif speaker == 'customer':
                        customer_embs.append(emb)
                        customer_count_call += 1
            except Exception:
                pass

        print(f"  [{call_id}] Extracted: {agent_count_call} agent, {customer_count_call} customer")

    agent_embs = np.array(agent_embs) if agent_embs else np.array([])
    customer_embs = np.array(customer_embs) if customer_embs else np.array([])

    print(f"\nTOTAL: {len(agent_embs)} agent embeddings, {len(customer_embs)} customer embeddings")

    if len(agent_embs) < 3:
        print("ERROR: Need at least 3 agent embeddings")
        sys.exit(1)

    # Step 5: Cluster into multiple voiceprints
    print(f"\n[STEP 5] Clustering into {args.n_clusters} voiceprints...")
    print("-" * 130)

    if len(agent_embs) >= args.n_clusters:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(agent_embs)

        centroids = []
        for k in range(args.n_clusters):
            cluster_embs = agent_embs[labels == k]
            if len(cluster_embs) >= 2:
                centroid = np.mean(cluster_embs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                centroids.append(centroid)
                print(f"  Cluster {k+1}: {len(cluster_embs)} members")
    else:
        # Single centroid
        centroid = np.mean(agent_embs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids = [centroid]
        print(f"  Single centroid from {len(agent_embs)} embeddings")

    # Step 6: Compute statistics
    print(f"\n[STEP 6] Computing voiceprint statistics...")
    print("-" * 130)

    # Agent similarities (max across all centroids)
    agent_sims = []
    for emb in agent_embs:
        emb_norm = emb / max(np.linalg.norm(emb), 1e-8)
        best = max(np.dot(c, emb_norm) for c in centroids)
        agent_sims.append(best)

    customer_sims = []
    for emb in customer_embs:
        emb_norm = emb / max(np.linalg.norm(emb), 1e-8)
        best = max(np.dot(c, emb_norm) for c in centroids)
        customer_sims.append(best)

    mean_inside = float(np.mean(agent_sims)) if agent_sims else 0.0
    max_outside = float(np.percentile(customer_sims, 95)) if customer_sims else 0.30
    gap = mean_inside - max_outside

    print(f"  Mean agent similarity: {mean_inside:.4f}")
    print(f"  Max customer similarity (95th): {max_outside:.4f}")
    print(f"  Separation gap: {gap:+.4f}")

    if gap > 0.10:
        print(f"  EXCELLENT: Strong separation - high accuracy expected")
    elif gap > 0:
        print(f"  GOOD: Positive separation")
    else:
        print(f"  WARNING: Negative gap - voices too similar")

    # Step 7: Save voiceprints as .npy
    print(f"\n[STEP 7] Saving voiceprints...")
    print("-" * 130)

    voiceprint_dir = Path("call_processor/data/agent_voiceprints")
    voiceprint_dir.mkdir(parents=True, exist_ok=True)

    npy_paths = []
    for i, c in enumerate(centroids):
        npy_path = voiceprint_dir / f"{AGENT_KEY}_titanet_v{i+1}.npy"
        np.save(npy_path, c.astype(np.float32))
        npy_paths.append(str(npy_path.absolute()))
        print(f"  Saved cluster {i+1} to: {npy_path.name}")

    # Step 8: Update agents.json
    print(f"\n[STEP 8] Updating agents.json...")
    print("-" * 130)

    agents_json = "call_processor/data/agent_voiceprints/agents.json"
    with open(agents_json) as f:
        agents_db = json.load(f)

    backup_file = agents_json.replace(".json", f".titanet_backup.{int(time.time())}.json")
    shutil.copy(agents_json, backup_file)

    agents_db[AGENT_KEY] = {
        "agent_name": AGENT_NAME,
        "voiceprint_path": npy_paths[0],
        "voiceprints": npy_paths,
        "n_voiceprints": len(npy_paths),
        "mean_inside_sim": mean_inside,
        "max_outside_sim": min(max(max_outside, 0.30), 0.50),
        "embedding_model": "titanet_large",
        "embedding_dim": 192,
        "source": f"titanet_large_{len(agent_embs)}segs_{len(centroids)}clusters",
        "n_training_segments": len(agent_embs),
    }

    with open(agents_json, 'w') as f:
        json.dump(agents_db, f, indent=2)

    print(f"  Updated {AGENT_KEY} with {len(centroids)} TitaNet voiceprints")

    print(f"\n" + "=" * 130)
    print("TITANET TRAINING COMPLETE")
    print("=" * 130)
    print(f"""
Summary:
  Agent: {AGENT_NAME} ({AGENT_KEY})
  Embedding model: TitaNet-Large (192-dim)
  Training segments: {len(agent_embs)} agent
  Clusters: {len(centroids)}
  Mean inside: {mean_inside:.4f}
  Max outside: {max_outside:.4f}
  Gap: {gap:+.4f}

Next: Test the model
""")


if __name__ == "__main__":
    main()
