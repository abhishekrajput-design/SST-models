#!/usr/bin/env python
"""
Gemini Browser Automation - Upload audio and extract role labels

Uses Playwright to automate Gemini interface:
1. Login to Gemini
2. Upload audio files
3. Send transcription prompts
4. Extract JSON responses
5. Save as gemini_labels_*.json

Requirements:
  pip install playwright
  playwright install chromium

Usage:
  python gemini_browser_auto.py "Agent Name" --calls call_id1 call_id2 call_id3
"""

import json
import sys
import time
import argparse
from pathlib import Path
import glob
import re

print("=" * 130)
print("GEMINI BROWSER AUTOMATION - EXTRACT SPEAKER ROLES")
print("=" * 130)

# Parse arguments
parser = argparse.ArgumentParser(description='Automate Gemini with browser')
parser.add_argument('agent', help='Agent name')
parser.add_argument('--calls', nargs='+', help='Call IDs to process', required=True)
args = parser.parse_args()

AGENT_NAME = args.agent
CALL_IDS = args.calls

print(f"\nAgent: {AGENT_NAME}")
print(f"Calls to process: {len(CALL_IDS)}")
for call_id in CALL_IDS:
    print(f"  - {call_id}")

# Step 1: Find audio files
print(f"\n[STEP 1] Finding audio files...")
print("-" * 130)

audio_files = {}
for call_id in CALL_IDS:
    candidates = (
        glob.glob(f"call_processor/data/processed/**/*{call_id}*/*.wav", recursive=True) +
        glob.glob(f"call_processor/data/processed/{call_id}*/*.wav", recursive=True)
    )

    if candidates:
        audio_path = candidates[0]
        size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        print(f"  [{call_id}] {Path(audio_path).name} ({size_mb:.1f} MB)")
        audio_files[call_id] = audio_path
    else:
        print(f"  [{call_id}] WARNING: Not found")

if not audio_files:
    print("ERROR: No audio files found")
    sys.exit(1)

# Step 2: Create Playwright automation script
print(f"\n[STEP 2] Starting browser automation...")
print("-" * 130)

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    print("Installing Playwright...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "playwright", "-q"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    from playwright.sync_api import sync_playwright, expect

PROMPT_TEMPLATE = """Transcribe this call and identify the speaker (agent or customer) for each segment with precise timestamps.
The agent's name is {agent_name}.
Return ONLY valid JSON with no markdown:

{{
  "call_id": "{call_id}",
  "agent_name": "{agent_name}",
  "source": "gemini",
  "segments": [
    {{"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."}},
    {{"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}}
  ]
}}

Focus on accuracy: mark each segment as "agent" or "customer"."""

def extract_json_from_text(text):
    """Extract JSON object from text"""
    # Find first { and matching }
    start_idx = text.find('{')
    if start_idx < 0:
        return None

    brace_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except:
                    return None
    return None

training_dir = Path("call_processor/data/training")
training_dir.mkdir(parents=True, exist_ok=True)

processed_count = 0

# Step 3: Browser automation
with sync_playwright() as p:
    print(f"\n[STEP 3] Opening browser and processing calls...")
    print("-" * 130)

    # Launch browser
    browser = p.chromium.launch(headless=False)  # headless=False so user can see what's happening
    page = browser.new_page()

    try:
        # Navigate to Gemini
        print("\n>>> Navigating to Gemini...")
        page.goto("https://gemini.google.com/app", timeout=30000)
        print(">>> Waiting for page to load...")
        page.wait_for_timeout(3000)

        # Check if logged in
        page.wait_for_selector("[role='main']", timeout=10000)
        print(">>> Gemini loaded successfully")

        # Process each call
        for idx, (call_id, audio_path) in enumerate(audio_files.items(), 1):
            print(f"\n{'='*130}")
            print(f"[{idx}/{len(audio_files)}] Processing: {call_id}")
            print(f"{'='*130}")

            try:
                # Wait for input to be ready
                print(f"  Waiting for input area...")
                page.wait_for_selector("[role='textbox'], [contenteditable='true']", timeout=10000)

                # Focus on input
                input_elem = page.query_selector("[contenteditable='true']")
                if input_elem:
                    input_elem.click()
                    print(f"  Input focused")

                # Look for file upload button
                print(f"  Looking for file upload button...")

                # Try to find and click the upload button
                upload_buttons = page.query_selector_all("button")
                upload_btn = None

                for btn in upload_buttons:
                    aria_label = btn.get_attribute("aria-label") or ""
                    title = btn.get_attribute("title") or ""
                    text = btn.text_content() or ""

                    if any(x in aria_label.lower() for x in ["attach", "file", "upload"]):
                        upload_btn = btn
                        break
                    if any(x in title.lower() for x in ["attach", "file", "upload"]):
                        upload_btn = btn
                        break

                if upload_btn:
                    print(f"  Found upload button, clicking...")
                    upload_btn.click()
                    page.wait_for_timeout(500)
                else:
                    print(f"  NOTE: Upload button not found - you may need to attach file manually")
                    print(f"  File to attach: {audio_path}")

                # Try to use file upload dialog
                print(f"  Waiting for file dialog...")
                try:
                    with page.expect_file_chooser() as fc_info:
                        # Try clicking potential file upload areas
                        page.click("[type='file']") if page.query_selector("[type='file']") else None

                    file_chooser = fc_info.value
                    print(f"  File dialog appeared, uploading: {audio_path}")
                    file_chooser.set_files(audio_path)
                    print(f"  File uploaded!")
                    page.wait_for_timeout(2000)
                except:
                    print(f"  INFO: Manual file upload may be needed")

                # Type the prompt
                print(f"  Typing transcription prompt...")
                prompt = PROMPT_TEMPLATE.format(agent_name=AGENT_NAME, call_id=call_id)

                # Clear input and type prompt
                input_elem = page.query_selector("[contenteditable='true']")
                if input_elem:
                    input_elem.click()
                    page.keyboard.press("Control+A")
                    page.keyboard.type(prompt, delay=5)  # Slower typing to ensure capture
                    print(f"  Prompt entered ({len(prompt)} chars)")

                # Send message
                print(f"  Sending to Gemini...")
                page.keyboard.press("Enter")

                # Wait for response
                print(f"  Waiting for response...")
                page.wait_for_timeout(5000)  # Initial wait

                # Poll for response (wait up to 60 seconds)
                response_received = False
                for attempt in range(60):
                    # Look for response in the page
                    response_text = page.text_content()

                    if "call_id" in response_text and "segments" in response_text and "{" in response_text:
                        print(f"  Response received!")
                        response_received = True
                        break

                    if attempt % 10 == 0:
                        print(f"    Waiting... ({attempt}s)")

                    page.wait_for_timeout(1000)

                if not response_received:
                    print(f"  WARNING: No JSON response detected after 60s")
                    print(f"  Please check Gemini page and verify response")
                    response_text = page.text_content()
                else:
                    # Extract JSON from response
                    print(f"  Extracting JSON...")
                    json_obj = extract_json_from_text(response_text)

                    if json_obj:
                        output_file = training_dir / f"gemini_labels_{call_id}_call{idx}.json"
                        with open(output_file, 'w') as f:
                            json.dump(json_obj, f, indent=2)

                        segment_count = len(json_obj.get('segments', []))
                        print(f"  SUCCESS! Saved to: {output_file.name}")
                        print(f"  Segments: {segment_count}")
                        processed_count += 1
                    else:
                        print(f"  ERROR: Could not parse JSON from response")
                        print(f"  Response preview: {response_text[:200]}")

                # Add delay between calls
                if idx < len(audio_files):
                    print(f"  Waiting before next call...")
                    page.wait_for_timeout(3000)

            except Exception as e:
                print(f"  ERROR processing {call_id}: {e}")
                print(f"  You may need to manually upload this call to Gemini")

    finally:
        print(f"\n{'='*130}")
        print(f"BROWSER AUTOMATION COMPLETE")
        print(f"{'='*130}")
        print(f"\nProcessed: {processed_count}/{len(audio_files)} calls")

        if processed_count > 0:
            print(f"\nGenerated files:")
            for f in training_dir.glob("gemini_labels_*.json"):
                print(f"  ✓ {f.name}")

        print(f"\nNext step:")
        print(f"  python call_processor/scripts/combine_and_retrain.py \"{AGENT_NAME}\"")

        # Keep browser open for a moment
        print(f"\nBrowser will close in 10 seconds...")
        page.wait_for_timeout(10000)
        browser.close()

print("\n" + "=" * 130)
print("DONE")
print("=" * 130)
