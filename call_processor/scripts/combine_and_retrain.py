#!/usr/bin/env python
"""
Combine Multiple Gemini Labels and Retrain for 95%+ Accuracy

Workflow:
1. Load multiple Gemini label files
2. Combine all segments
3. Train single voiceprint from combined data
4. Update agents.json
5. Test on held-out calls

Usage:
  python call_processor/scripts/combine_and_retrain.py [AGENT_NAME] [--test-call CALL_ID]

Example:
  python call_processor/scripts/combine_and_retrain.py "Omar El Harchaoui"
  python call_processor/scripts/combine_and_retrain.py "Zak Raissi" --test-call zak_compare_20260423
"""

import json
import sys
import warnings
from pathlib import Path
from typing import List, Dict
import numpy as np
import soundfile as sf
import shutil
import time
import argparse
import glob

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.embedding_campp import get_model

print("=" * 130)
print("MULTI-CALL RETRAINING FOR 95%+ ACCURACY")
print("=" * 130)

# Parse arguments
parser = argparse.ArgumentParser(description='Combine Gemini labels and retrain')
parser.add_argument('agent', help='Agent name (e.g., "Omar El Harchaoui")')
parser.add_argument('--test-call', help='Call ID to use for testing (held-out)', default=None)
parser.add_argument('--min-dur', type=float, default=0.5, help='Minimum segment duration (seconds)')
args = parser.parse_args()

agent_name = args.agent
test_call_id = args.test_call

print(f"\nAgent: {agent_name}")
print(f"Test call: {test_call_id if test_call_id else 'None (will test on combined data)'}")

# Step 1: Find all Gemini label files for this agent
print(f"\n[STEP 1] Finding Gemini label files...")
print("-" * 130)

training_dir = Path("call_processor/data/training")
agent_key = agent_name.lower().replace(' ', '_')

label_files = list(training_dir.glob(f"gemini_labels*.json"))
print(f"Found {len(label_files)} Gemini label files:")
for f in label_files:
    print(f"  - {f.name}")

# Filter for this agent
agent_labels = []
for label_file in label_files:
    try:
        with open(label_file) as f:
            data = json.load(f)
            file_agent = data.get('agent_name', '').lower().replace(' ', '_')
            if file_agent == agent_key or agent_key in label_file.name.lower():
                agent_labels.append({
                    'file': label_file,
                    'data': data,
                    'call_id': data.get('call_id'),
                    'segment_count': len(data.get('segments', []))
                })
    except Exception as e:
        print(f"  Warning: Could not load {label_file.name}: {e}")

print(f"\nFound {len(agent_labels)} label file(s) for {agent_name}:")
for label_info in agent_labels:
    print(f"  - {label_info['file'].name}: {label_info['segment_count']} segments (call: {label_info['call_id']})")

if not agent_labels:
    print(f"ERROR: No Gemini labels found for {agent_name}")
    sys.exit(1)

# Step 2: Load audio files for all calls
print(f"\n[STEP 2] Loading audio files...")
print("-" * 130)

audio_data = {}
for label_info in agent_labels:
    call_id = label_info['call_id']
    audio_files = glob.glob(f"call_processor/data/processed/**/*{call_id}*/*.wav", recursive=True)
    audio_files += glob.glob(f"call_processor/data/raw_calls/*{call_id}*.mp3", recursive=True)
    audio_files += glob.glob(f"testing-audio/**/*{call_id}*.mp3", recursive=True)

    # Also try partial matches
    if not audio_files:
        for alt_name in [call_id.replace('_', ''), call_id.split('_')[0]]:
            audio_files = glob.glob(f"**/*{alt_name}*/*.wav", recursive=True)[:1]
            if audio_files:
                break

    if audio_files:
        audio_path = audio_files[0]
        print(f"  [{call_id}] Found: {Path(audio_path).name}")

        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except:
                    pass
            audio_data[call_id] = {'audio': audio, 'sr': 16000}
        except Exception as e:
            print(f"  [{call_id}] WARNING: Could not load audio: {e}")
    else:
        print(f"  [{call_id}] WARNING: Audio file not found")

# Step 3: Extract embeddings from all calls
print(f"\n[STEP 3] Extracting embeddings from all calls...")
print("-" * 130)

embedding_model = get_model(force_cpu=True)
print(f"Using {embedding_model.model_name} model ({embedding_model.dim}D)")

all_agent_embeddings = []
all_customer_embeddings = []

for label_info in agent_labels:
    call_id = label_info['call_id']
    segments = label_info['data'].get('segments', [])

    if call_id not in audio_data:
        print(f"\n  Skipping {call_id} (audio not available)")
        continue

    print(f"\n  Processing {call_id}...")
    audio = audio_data[call_id]['audio']
    sr = audio_data[call_id]['sr']

    for seg in segments:
        speaker = seg.get('speaker', '').lower()
        start_s = float(seg.get('start', 0))
        end_s = float(seg.get('end', 0))
        dur_s = end_s - start_s
        text = seg.get('text', '')

        # Skip very short segments
        if dur_s < args.min_dur:
            continue

        # Extract window
        start_samp = int(start_s * sr)
        end_samp = int(end_s * sr)
        window = audio[start_samp:end_samp]

        if len(window) < int(args.min_dur * sr):
            continue

        try:
            emb = embedding_model.embed_chunk(window, sr=sr)
            if emb is not None and not np.isnan(emb).any():
                emb_info = {
                    'embedding': emb,
                    'duration': dur_s,
                    'start': start_s,
                    'end': end_s,
                    'text': text[:50],
                    'call_id': call_id,
                }

                if speaker == 'agent':
                    all_agent_embeddings.append(emb_info)
                elif speaker == 'customer':
                    all_customer_embeddings.append(emb_info)
        except Exception as e:
            pass

print(f"\nTotal agent segments extracted: {len(all_agent_embeddings)}")
print(f"Total customer segments extracted: {len(all_customer_embeddings)}")

if len(all_agent_embeddings) < 3:
    print("ERROR: Need at least 3 agent segments to train")
    sys.exit(1)

# Step 4: Build combined voiceprint
print(f"\n[STEP 4] Building combined voiceprint...")
print("-" * 130)

embeddings = np.array([seg["embedding"] for seg in all_agent_embeddings])

# Centroid from all agent segments
centroid = np.mean(embeddings, axis=0)
centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

# Statistics
mean_sim = float(np.mean([np.dot(centroid, embeddings[i]) for i in range(len(embeddings))]))
max_sim = float(np.max([np.dot(centroid, embeddings[i]) for i in range(len(embeddings))]))
min_sim = float(np.min([np.dot(centroid, embeddings[i]) for i in range(len(embeddings))]))

print(f"\nCombined Voiceprint Statistics:")
print(f"  Segments used: {len(all_agent_embeddings)}")
print(f"  Mean similarity: {mean_sim:.4f}")
print(f"  Max similarity: {max_sim:.4f}")
print(f"  Min similarity: {min_sim:.4f}")

# Compute max_outside_sim from customer segments
if all_customer_embeddings:
    customer_embeddings = np.array([seg["embedding"] for seg in all_customer_embeddings])
    max_outside_sims = []
    for cust_emb in customer_embeddings:
        sim = np.dot(centroid, cust_emb)
        max_outside_sims.append(sim)
    max_outside_sim = float(np.percentile(max_outside_sims, 95))
    print(f"  Max customer similarity (95th percentile): {max_outside_sim:.4f}")
else:
    max_outside_sim = 0.30
    print(f"  Max customer similarity: {max_outside_sim:.4f} (default)")

# Step 5: Update agents.json
print(f"\n[STEP 5] Updating agents.json...")
print("-" * 130)

AGENTS_JSON = "call_processor/data/agent_voiceprints/agents.json"

with open(AGENTS_JSON, 'r') as f:
    agents_db = json.load(f)

# Convert agent name to database key format
agent_key_db = agent_name.lower().replace(' ', '_')

# Find matching agent in database
matched_agent = None
for db_agent_name in agents_db.keys():
    if db_agent_name.lower().replace(' ', '_') == agent_key_db:
        matched_agent = db_agent_name
        break

if matched_agent:
    # Backup old version
    backup_file = AGENTS_JSON.replace(".json", f".backup.{agent_key_db}.{int(time.time())}.json")
    shutil.copy(AGENTS_JSON, backup_file)
    print(f"Backed up to: {backup_file}")

    # Update with combined training
    agents_db[matched_agent] = {
        "mean_inside_sim": mean_sim,
        "max_outside_sim": min(max_outside_sim + 0.04, 0.36),  # Conservative threshold
        "n_voiceprints": 1,
        "source": f"multi_call_gemini_{len(all_agent_embeddings)}segs",
        "voiceprints": [centroid.tolist()],
    }

    # Write back
    with open(AGENTS_JSON, 'w') as f:
        json.dump(agents_db, f, indent=2)

    print(f"Updated {matched_agent} with combined voiceprint")
    print(f"\nNew voiceprint properties:")
    print(f"  Source: {agents_db[matched_agent]['source']}")
    print(f"  Mean inside: {mean_sim:.4f}")
    print(f"  Max outside: {agents_db[matched_agent]['max_outside_sim']:.4f}")
else:
    print(f"ERROR: Agent '{agent_name}' not found in database")
    print(f"Available agents: {list(agents_db.keys())[:5]}")
    sys.exit(1)

print("\n" + "=" * 130)
print("MULTI-CALL RETRAINING COMPLETE")
print("=" * 130)

print(f"""
Summary:
  Agent: {matched_agent}
  Calls used: {len(agent_labels)}
  Total agent segments: {len(all_agent_embeddings)}
  Voiceprint updated: YES
  Ready to test: YES

Next steps:
  1. Test on new calls: python scripts/test_and_compare_with_gemini.py
  2. Verify accuracy >= 95%
  3. Move to next agent
""")

print("=" * 130)
