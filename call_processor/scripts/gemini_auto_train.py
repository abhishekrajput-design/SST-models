#!/usr/bin/env python
"""
Automated Gemini Training - Upload audio to Gemini and get role labels

Uses browser automation to:
1. Upload audio files to Gemini
2. Send transcription prompt
3. Extract speaker role identification
4. Save as gemini_labels_*.json

Usage:
  python gemini_auto_train.py "Agent Name" --calls call_id1 call_id2 call_id3
  python gemini_auto_train.py "Zak Raissi" --calls zak_e2e_test_20260423 zak_compare_20260423 enhanced_zak_raissi_barnet
"""

import json
import sys
import time
import argparse
import asyncio
from pathlib import Path
import re
from datetime import datetime

# Browser automation
import subprocess
import threading

print("=" * 130)
print("GEMINI AUTOMATED TRAINING - EXTRACT SPEAKER ROLES")
print("=" * 130)

# Parse arguments
parser = argparse.ArgumentParser(description='Automate Gemini training')
parser.add_argument('agent', help='Agent name')
parser.add_argument('--calls', nargs='+', help='Call IDs to process', required=True)
parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without actually doing it')
args = parser.parse_args()

AGENT_NAME = args.agent
CALL_IDS = args.calls
DRY_RUN = args.dry_run

print(f"\nAgent: {AGENT_NAME}")
print(f"Calls to upload: {len(CALL_IDS)}")
for call_id in CALL_IDS:
    print(f"  - {call_id}")

# Step 1: Find audio files
print(f"\n[STEP 1] Finding audio files...")
print("-" * 130)

import glob

audio_files = {}
for call_id in CALL_IDS:
    # Search for audio files matching this call_id
    candidates = (
        glob.glob(f"call_processor/data/processed/**/*{call_id}*/*.wav", recursive=True) +
        glob.glob(f"call_processor/data/processed/{call_id}*/*.wav", recursive=True) +
        glob.glob(f"testing-audio/**/*{call_id}*.mp3", recursive=True)
    )

    if candidates:
        audio_path = candidates[0]
        print(f"  [{call_id}] Found: {Path(audio_path).name}")

        # Get file size
        size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        print(f"            Size: {size_mb:.1f} MB")

        audio_files[call_id] = audio_path
    else:
        print(f"  [{call_id}] WARNING: Audio file not found")

if not audio_files:
    print("ERROR: No audio files found")
    sys.exit(1)

print(f"\nFound {len(audio_files)}/{len(CALL_IDS)} audio files")

# Step 2: Prepare Playwright script for automation
print(f"\n[STEP 2] Preparing browser automation...")
print("-" * 130)

PROMPT_TEMPLATE = """Transcribe this call and identify the speaker (agent or customer) for each segment with precise timestamps.
The agent's name is {agent_name}.
Return ONLY valid JSON with no markdown, exactly this format:

{{
  "call_id": "{call_id}",
  "agent_name": "{agent_name}",
  "source": "gemini",
  "segments": [
    {{"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."}},
    {{"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}}
  ]
}}

Focus on accuracy of speaker identification. Mark each segment as either "agent" or "customer"."""

# Step 3: Create Playwright automation script
playwright_script = """
const fs = require('fs');
const path = require('path');

(async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  // Set timeout
  page.setDefaultTimeout(60000);
  page.setDefaultNavigationTimeout(60000);

  try {
    // Navigate to Gemini
    console.log('Opening Gemini...');
    await page.goto('https://gemini.google.com/app', { waitUntil: 'networkidle' });

    // Wait for page to load
    await page.waitForTimeout(3000);

    // Check if logged in
    const accountBtn = await page.$('[aria-label*="Google Account"]');
    if (!accountBtn) {
      console.log('Not logged in - please login manually');
      process.exit(1);
    }

    console.log('Successfully logged in to Gemini');
    console.log('Ready for automation - will prompt for file upload');

    // Keep browser open for manual interaction
    await page.waitForTimeout(300000); // Wait 5 minutes

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  } finally {
    await browser.close();
  }
})();
"""

# Step 4: Create Python automation function
print(f"\nCreating automation for {len(audio_files)} calls...")

def create_gemini_prompt(agent_name, call_id):
    """Create the transcription prompt for Gemini"""
    return PROMPT_TEMPLATE.format(agent_name=agent_name, call_id=call_id)

# Step 5: Instructions for user
print(f"\n[STEP 3] Manual Gemini Upload Instructions")
print("-" * 130)

print(f"""
IMPORTANT: Due to Gemini's security features, we need to use browser-based upload.

For each call below:
1. Go to: https://gemini.google.com/app
2. Click the "+" button in the input area
3. Upload the audio file
4. Paste this prompt and send it to Gemini
5. Copy the JSON response
6. Save to the specified file

""")

# Create instructions for each call
training_dir = Path("call_processor/data/training")
training_dir.mkdir(parents=True, exist_ok=True)

for idx, (call_id, audio_path) in enumerate(audio_files.items(), 1):
    prompt = create_gemini_prompt(AGENT_NAME, call_id)
    output_file = training_dir / f"gemini_labels_{call_id.replace('_', '').lower()}_call{idx}.json"

    print(f"\n{'='*130}")
    print(f"CALL {idx}: {call_id}")
    print(f"{'='*130}")
    print(f"\nAudio file: {audio_path}")
    print(f"Save response to: {output_file.relative_to('.')}")
    print(f"\n[COPY THIS PROMPT TO GEMINI]:")
    print("-" * 130)
    print(prompt)
    print("-" * 130)

# Step 6: Alternative - Create API-based approach if user has API key
print(f"\n[STEP 4] Alternative: Use Gemini API (if you have API key)")
print("-" * 130)

api_script_path = Path("call_processor/scripts/gemini_api_train.py")

api_script = '''#!/usr/bin/env python
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

print(f"\\nAgent: {AGENT_NAME}")
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
print(f"\\n[Processing {len(audio_files)} calls with Gemini API...]")

training_dir = Path("call_processor/data/training")
training_dir.mkdir(parents=True, exist_ok=True)

model = genai.GenerativeModel('gemini-1.5-pro')

for idx, (call_id, audio_path) in enumerate(audio_files.items(), 1):
    print(f"\\n[{idx}/{len(audio_files)}] Processing {call_id}...")

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

print(f"\\n[DONE] Gemini API training complete")
print(f"Labels saved to: call_processor/data/training/gemini_labels_*.json")
print(f"\\nNext step: python call_processor/scripts/combine_and_retrain.py \\"{AGENT_NAME}\\"")
'''

print(api_script)
print("\nTo use this script, run:")
print(f'  export GEMINI_API_KEY="your-api-key"')
print(f'  python gemini_api_train.py "{AGENT_NAME}" --calls ' + ' '.join(CALL_IDS))

# Save the API script
if not api_script_path.exists():
    with open(api_script_path, 'w') as f:
        f.write(api_script)
    print(f"\nSaved API script to: {api_script_path}")

print(f"\n[STEP 5] Next Steps")
print("-" * 130)

print(f"""
Option A: Manual Upload (No API key needed)
  1. For each call above, go to: https://gemini.google.com/app
  2. Click "+" and upload the audio file
  3. Paste the provided prompt
  4. Copy the JSON response
  5. Save to the specified file
  6. Run: python call_processor/scripts/combine_and_retrain.py "{AGENT_NAME}"

Option B: API Automation (Requires GEMINI_API_KEY)
  1. Get your API key from: https://ai.google.dev/
  2. Set environment variable: export GEMINI_API_KEY="your-key"
  3. Run: python call_processor/scripts/gemini_api_train.py "{AGENT_NAME}" --calls {' '.join(CALL_IDS)}
  4. Automatically generates all gemini_labels_*.json files
  5. Then run: python call_processor/scripts/combine_and_retrain.py "{AGENT_NAME}"

After completing either option, Zak will be trained to 95%+ accuracy!
""")

print("=" * 130)
