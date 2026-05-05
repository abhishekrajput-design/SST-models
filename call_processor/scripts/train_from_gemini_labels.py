#!/usr/bin/env python
"""
Train our model using correct labels from Gemini.

Workflow:
1. Get correct transcription + labels from Gemini
2. Save as JSON file (see format below)
3. Run this script
4. Compare our accuracy with Gemini's

Expected JSON format from Gemini:
{
  "call_id": "omar_test",
  "agent_name": "Omar El Harchaoui",
  "source": "gemini",
  "segments": [
    {
      "start": 0.0,
      "end": 1.32,
      "speaker": "customer",
      "text": "Hi, I'm interested..."
    },
    {
      "start": 1.32,
      "end": 1.64,
      "speaker": "agent",
      "text": "Yeah speaking."
    }
  ]
}
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

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.embedding_campp import get_model
from src.voiceprints import load_agents_index

print("=" * 110)
print("TRAIN FROM GEMINI LABELS - PERFECT ACCURACY BASELINE")
print("=" * 110)

print("""
This script uses CORRECT labels from Gemini to build perfect voiceprints.
This is the gold standard - what our system SHOULD achieve with proper data.

Steps:
1. Save Gemini's transcription as JSON
2. Point this script to the file
3. Run training
4. Test accuracy (should be near 100%)
""")

# Configuration
GEMINI_LABELS_FILE = Path("call_processor/data/training/gemini_labels.json")
AUDIO_FILE = Path("c:/Users/abhis/Downloads/20260505T073055769_385036.mp3")
AGENTS_JSON = "call_processor/data/agent_voiceprints/agents.json"

# Check if labels file exists
if not GEMINI_LABELS_FILE.exists():
    print(f"""
ERROR: Gemini labels file not found at:
  {GEMINI_LABELS_FILE}

SETUP INSTRUCTIONS:
==================

1. Open: https://gemini.google.com
2. Upload the Omar audio file
3. Ask Gemini:
   "Transcribe this call and label speaker (agent/customer) with timestamps"
4. Copy Gemini's JSON response
5. Save to: {GEMINI_LABELS_FILE}

Example JSON format:
{{
  "call_id": "omar_gemini",
  "agent_name": "Omar El Harchaoui",
  "source": "gemini",
  "segments": [
    {{"start": 0.0, "end": 1.32, "speaker": "customer", "text": "Hi..."}},
    {{"start": 1.32, "end": 1.64, "speaker": "agent", "text": "Yeah..."}}
  ]
}}

Then run this script again.
""")
    sys.exit(1)

print(f"\nLoading Gemini labels from: {GEMINI_LABELS_FILE}")
with open(GEMINI_LABELS_FILE, 'r') as f:
    gemini_data = json.load(f)

call_id = gemini_data.get('call_id', 'unknown')
agent_name = gemini_data.get('agent_name', 'Unknown Agent')
segments = gemini_data.get('segments', [])

print(f"Call: {call_id}")
print(f"Agent: {agent_name}")
print(f"Segments: {len(segments)}")

if not AUDIO_FILE.exists():
    print(f"ERROR: Audio file not found: {AUDIO_FILE}")
    sys.exit(1)

print(f"\nLoading audio: {AUDIO_FILE.name}")
audio, sr = sf.read(str(AUDIO_FILE))
if len(audio.shape) > 1:
    audio = audio[:, 0]
if sr != 16000:
    try:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    except:
        print("Warning: Could not resample audio")
        pass
sr = 16000

print(f"Audio: {len(audio)/sr:.1f} seconds @ {sr} Hz")

# Extract agent segments from Gemini labels
print(f"\nExtracting agent segments from Gemini labels...")

agent_embeddings = []
agent_segments_found = 0

embedding_model = get_model(force_cpu=True)
print(f"Using {embedding_model.model_name} model ({embedding_model.dim}D)")

for seg in segments:
    if seg.get('speaker', '').lower() == 'agent':
        start_s = float(seg.get('start', 0))
        end_s = float(seg.get('end', 0))
        text = seg.get('text', '')
        dur_s = end_s - start_s

        # Skip very short segments
        if dur_s < 0.5:
            continue

        # Extract window
        start_samp = int(start_s * sr)
        end_samp = int(end_s * sr)
        window = audio[start_samp:end_samp]

        if len(window) < int(0.5 * sr):  # At least 0.5s
            continue

        try:
            emb = embedding_model.embed_chunk(window, sr=sr)
            if emb is not None and not np.isnan(emb).any():
                agent_embeddings.append({
                    "embedding": emb,
                    "duration": dur_s,
                    "start": start_s,
                    "end": end_s,
                    "text": text[:50],
                })
                agent_segments_found += 1
                print(f"  [OK] Segment {start_s:.2f}-{end_s:.2f}s ({dur_s:.2f}s): {text[:50]}")
        except Exception as e:
            print(f"  [FAIL] Segment {start_s:.2f}-{end_s:.2f}s: Failed to embed")

print(f"\nExtracted {agent_segments_found} agent segments")

if agent_segments_found < 3:
    print("ERROR: Need at least 3 agent segments to train. Not enough data.")
    sys.exit(1)

# Build voiceprint
print(f"\nBuilding voiceprint from {agent_segments_found} segments...")

embeddings = np.array([seg["embedding"] for seg in agent_embeddings])

# Centroid (single voiceprint for clarity)
centroid = np.mean(embeddings, axis=0)
centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

# Statistics
mean_sim = float(np.mean([np.dot(centroid, embeddings[i]) for i in range(len(embeddings))]))
max_sim = float(np.max([np.dot(centroid, embeddings[i]) for i in range(len(embeddings))]))
min_sim = float(np.min([np.dot(centroid, embeddings[i]) for i in range(len(embeddings))]))

print(f"Voiceprint statistics:")
print(f"  Mean similarity: {mean_sim:.4f}")
print(f"  Max similarity:  {max_sim:.4f}")
print(f"  Min similarity:  {min_sim:.4f}")

# Update agents.json
print(f"\nUpdating agents.json...")

with open(AGENTS_JSON, 'r') as f:
    agents_db = json.load(f)

# Backup
backup_file = AGENTS_JSON.replace(".json", f".backup.gemini.{int(time.time())}.json")
shutil.copy(AGENTS_JSON, backup_file)
print(f"Backed up to: {backup_file}")

# Convert agent name to database key format (lowercase with underscores)
agent_key = agent_name.lower().replace(' ', '_')

# Find matching agent in database
matched_agent = None
for db_agent_name in agents_db.keys():
    if db_agent_name.lower().replace(' ', '_') == agent_key:
        matched_agent = db_agent_name
        break

if matched_agent:
    agents_db[matched_agent] = {
        "mean_inside_sim": mean_sim,
        "max_outside_sim": 0.30,  # Conservative floor
        "n_voiceprints": 1,
        "source": f"gemini_labels_{agent_segments_found}segs",
        "voiceprints": [centroid.tolist()],
    }

    # Write back
    with open(AGENTS_JSON, 'w') as f:
        json.dump(agents_db, f, indent=2)

    print(f"Updated {matched_agent} with perfect voiceprint")
    print(f"\nResult:")
    print(f"  Segments used: {agent_segments_found}")
    print(f"  Mean inside similarity: {mean_sim:.4f}")
    print(f"  Voiceprints: 1 (centroid from all segments)")
else:
    print(f"ERROR: Agent '{agent_name}' not found in database")
    print(f"Available agents: {list(agents_db.keys())[:10]}")

print("\n" + "=" * 110)
print("TRAINING COMPLETE")
print("=" * 110)
print(f"""
Next steps:
1. Run accuracy test: python scripts/COMPREHENSIVE_TEST.py
2. Expected accuracy: 95-100% (since trained on correct data)
3. Compare with Gemini's accuracy

This shows what our system can achieve with proper training data!
""")
