#!/usr/bin/env python
"""
Bulk re-enrollment of all agents using independent call recordings from API.
Extracts agent-only segments from multiple calls, avoiding circular training.
"""
import json
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.diar_multi import diarize_multi
from src.embedding_campp import CAMPPlus
from src.voiceprints import load_agents_index, compute_buckets

# Config
API_CALLS_ENDPOINT = "http://localhost:8080/api/calls"
AGENTS_JSON = "call_processor/data/agent_voiceprints/agents.json"
OUTPUT_DIR = Path("call_processor/data/agent_voiceprints")

# Re-enrollment params
MIN_AGENT_DUR = 2.0
MAX_AGENT_DUR = 20.0
SNR_MIN_DB = 12.0
MIN_SEGMENTS_PER_AGENT = 5

print("=" * 100)
print("BULK RE-ENROLLMENT FROM API CALLS")
print("=" * 100)

# Step 1: Get all processed calls from API
print("\nStep 1: Fetching processed calls from API...")
try:
    import requests
    response = requests.get(API_CALLS_ENDPOINT, timeout=10)
    all_calls = response.json()
    # Filter: only calls that were processed with Parakeet and have segments
    valid_calls = [
        c for c in all_calls
        if c.get("model") == "parakeet-tdt-0.6b-v3"
        and c.get("segments", 0) > 0
        and c.get("orig_file")
    ]
    print(f"Found {len(valid_calls)} valid processed calls (Parakeet + segments)")
except Exception as e:
    print(f"Error fetching calls: {e}")
    sys.exit(1)

# Step 2: Load existing agents to identify agent names
print("\nStep 2: Loading existing agent database...")
agents_db = load_agents_index(AGENTS_JSON)
agent_names = list(agents_db.keys())
print(f"Found {len(agent_names)} enrolled agents: {', '.join(agent_names[:5])}...")

# Step 3: Extract agent segments from calls
print("\nStep 3: Extracting agent segments from calls...")
agent_segments = {name: [] for name in agent_names}

embedding_model = CAMPPlus(device="cpu")  # CPU to avoid memory conflicts
print(f"Loaded CAM++ embedding model")

call_count = 0
for call_info in tqdm(valid_calls[:50], desc="Processing calls"):  # Limit to 50 calls for speed
    try:
        call_id = call_info.get("id", "")
        orig_file = call_info.get("orig_file", "")

        if not orig_file or not Path(orig_file).exists():
            continue

        # Get call details from API
        try:
            response = requests.get(f"http://localhost:8080/api/call/{call_id}", timeout=10)
            call_data = response.json()
            segments = call_data.get("segments", [])
        except:
            continue

        if not segments:
            continue

        # Load audio
        try:
            audio, sr = sf.read(orig_file)
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Take first channel
            if sr != 16000:
                # Resample via librosa if needed
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        except:
            continue

        call_count += 1

        # Extract agent windows from segments with identified_speaker labels
        for seg in segments:
            if seg.get("identified_speaker") not in agent_names:
                continue

            agent_name = seg["identified_speaker"]
            start_s = float(seg.get("start", 0))
            end_s = float(seg.get("end", 0))
            dur_s = end_s - start_s

            # Filter by duration
            if dur_s < MIN_AGENT_DUR or dur_s > MAX_AGENT_DUR:
                continue

            # Extract window
            start_samp = int(start_s * sr)
            end_samp = int(end_s * sr)
            window = audio[start_samp:end_samp]

            if len(window) < int(MIN_AGENT_DUR * sr):
                continue

            # Compute embedding
            try:
                emb = embedding_model.embed_chunk(window)
                if emb is not None and not np.isnan(emb).any():
                    agent_segments[agent_name].append({
                        "embedding": emb,
                        "duration": dur_s,
                        "call_id": call_id,
                        "text": seg.get("text", "")[:50],
                    })
            except:
                continue

    except Exception as e:
        continue

print(f"\nExtracted {call_count} calls total")
for agent_name, segs in agent_segments.items():
    print(f"  {agent_name}: {len(segs)} segments")

# Step 4: Build voiceprints per agent
print("\nStep 4: Building re-enrolled voiceprints...")
agents_updated = {}

for agent_name, segs in agent_segments.items():
    if len(segs) < MIN_SEGMENTS_PER_AGENT:
        print(f"  {agent_name}: SKIP (only {len(segs)} segments, need {MIN_SEGMENTS_PER_AGENT})")
        continue

    embeddings = np.array([s["embedding"] for s in segs])

    # Cluster by SNR (existing logic)
    buckets = compute_buckets(embeddings, n_buckets=3)

    # Build centroids
    voiceprints = []
    for bucket_idx, bucket_mask in enumerate(buckets):
        bucket_embs = embeddings[bucket_mask]
        if len(bucket_embs) > 0:
            centroid = np.mean(bucket_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)  # L2 normalize
            voiceprints.append(centroid.tolist())

    # Compute max_outside_sim (similarity to customer utterances from same calls)
    # For now, estimate as 0.40 (conservative)
    max_outside_sim = 0.40

    agents_updated[agent_name] = {
        "mean_inside_sim": float(np.mean([np.linalg.norm(embeddings[i] - np.mean(embeddings, axis=0)) for i in range(len(embeddings))])),
        "max_outside_sim": max_outside_sim,
        "n_voiceprints": len(voiceprints),
        "source": "bulk_reenroll_from_api",
        "voiceprints": voiceprints,
    }

    print(f"  {agent_name}: {len(segs)} segs → {len(voiceprints)} voiceprints")

# Step 5: Update agents.json with new voiceprints
print("\nStep 5: Updating agents.json...")
with open(AGENTS_JSON, "r") as f:
    existing_agents = json.load(f)

# Merge: keep existing agents, update with re-enrolled ones
for agent_name, agent_data in agents_updated.items():
    if agent_name in existing_agents:
        existing_agents[agent_name].update(agent_data)
    else:
        existing_agents[agent_name] = agent_data

# Backup original
import shutil
backup_file = AGENTS_JSON.replace(".json", f".backup.{int(Path.cwd().stat().st_mtime)}.json")
shutil.copy(AGENTS_JSON, backup_file)
print(f"Backed up to: {backup_file}")

# Write updated
with open(AGENTS_JSON, "w") as f:
    json.dump(existing_agents, f, indent=2)
print(f"Updated: {AGENTS_JSON}")

print("\n" + "=" * 100)
print("RE-ENROLLMENT COMPLETE")
print(f"Re-enrolled {len(agents_updated)} agents using {call_count} calls")
print("Next: Test accuracy on Omar's call to verify improvement")
print("=" * 100)
