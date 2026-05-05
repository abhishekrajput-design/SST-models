#!/usr/bin/env python
"""
Gemini API-based training - Fully automated version

Requires: GEMINI_API_KEY environment variable

Usage:
  export GEMINI_API_KEY="your-api-key-here"
  python gemini_api_train.py "Agent Name" --calls call_id1 call_id2
"""

import json
import sys
import os
from pathlib import Path
import glob
import argparse

# Check for API key
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("ERROR: GEMINI_API_KEY not set")
    print("Set it with: export GEMINI_API_KEY=your-key-here")
    sys.exit(1)

print(f"Using Gemini API key: {api_key[:20]}...")

try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "google-generativeai", "-q"], check=True)
    import google.generativeai as genai

# Configure API
genai.configure(api_key=api_key)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('agent', help='Agent name')
parser.add_argument('--calls', nargs='+', help='Call IDs', required=True)
args = parser.parse_args()

AGENT_NAME = args.agent
CALL_IDS = args.calls

print(f"\nAgent: {AGENT_NAME}")
print(f"Calls: {len(CALL_IDS)}")

# Find audio files
audio_files = {}
for call_id in CALL_IDS:
    candidates = (
        glob.glob(f"call_processor/data/processed/**/*{call_id}*/*.wav", recursive=True) +
        glob.glob(f"call_processor/data/processed/{call_id}*/*.wav", recursive=True)
    )
    if candidates:
        audio_files[call_id] = candidates[0]
        print(f"  [{call_id}] Found: {Path(candidates[0]).name}")

if not audio_files:
    print("ERROR: No audio files found")
    sys.exit(1)

# Process each call
print(f"\n[Processing {len(audio_files)} calls with Gemini API...]")

training_dir = Path("call_processor/data/training")
training_dir.mkdir(parents=True, exist_ok=True)

model = genai.GenerativeModel('gemini-2.0-flash')

for idx, (call_id, audio_path) in enumerate(audio_files.items(), 1):
    print(f"\n[{idx}/{len(audio_files)}] Processing {call_id}...")

    # Upload audio file
    print(f"  Uploading audio...")
    audio_file = genai.upload_file(audio_path)
    print(f"  Uploaded: {audio_file.name}")

    # Create prompt
    prompt = f"""Transcribe this call and identify the speaker (agent or customer) for each segment with precise timestamps.
The agent's name is {AGENT_NAME}.
Return ONLY valid JSON with no markdown, exactly this format:

{{
  "call_id": "{call_id}",
  "agent_name": "{AGENT_NAME}",
  "source": "gemini",
  "segments": [
    {{"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."}},
    {{"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}}
  ]
}}

Focus on accuracy of speaker identification. Mark each segment as either "agent" or "customer"."""

    # Send to Gemini
    print(f"  Sending to Gemini...")
    response = model.generate_content([audio_file, prompt])

    # Extract JSON from response
    response_text = response.text
    print(f"  Received response ({len(response_text)} chars)")

    # Parse JSON
    try:
        # Find JSON in response
        json_match = None
        start_idx = response_text.find('{')
        if start_idx >= 0:
            # Try to find matching closing brace
            brace_count = 0
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = response_text[start_idx:i+1]
                        break

            labels = json.loads(json_text)

            # Save to file
            output_file = training_dir / f"gemini_labels_{call_id}_call{idx}.json"
            with open(output_file, 'w') as f:
                json.dump(labels, f, indent=2)

            print(f"  Saved to: {output_file}")
            print(f"  Segments: {len(labels.get('segments', []))}")

    except Exception as e:
        print(f"  ERROR parsing JSON: {e}")
        print(f"  Response: {response_text[:200]}")

print(f"\n[DONE] Gemini API training complete")
print(f"Labels saved to: call_processor/data/training/gemini_labels_*.json")
print(f"\nNext step: python call_processor/scripts/combine_and_retrain.py \"{AGENT_NAME}\"")
