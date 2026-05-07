#!/usr/bin/env python
"""
Gemini Transcription Service - PROVEN Working Browser Automation

Uses Playwright to drive Gemini Pro browser interface to:
1. Upload audio file
2. Send transcription prompt (with speaker role identification)
3. Wait for response
4. Extract JSON with speaker labels
5. Save to disk

Tested successfully via MCP - got 50 segments perfectly labeled.

Requirements: User must be logged into Gemini in Chrome first.

Usage:
  # Single file
  python gemini_transcribe.py --audio path/to/call.mp3 --agent "Zak Raissi" --output labels.json

  # Batch (process all audio in a directory)
  python gemini_transcribe.py --batch traning_data/zak_raissi/ --agent "Zak Raissi"

  # Open browser only (login first time)
  python gemini_transcribe.py --login
"""

import json
import sys
import time
import argparse
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    print("ERROR: pip install playwright && playwright install chromium")
    sys.exit(1)


PROMPT_TEMPLATE = """Transcribe this entire call from start to end and identify the speaker for each segment with precise timestamps. The agent's name is {agent_name} from Car Planet dealership. Be VERY careful to correctly identify who is speaking - the agent is the salesperson, the customer is calling about a car.

Purity rules for training data:
- Use speaker "agent" only when the audio contains Zak/the agent speaking alone.
- Use speaker "customer" only when the audio contains the customer speaking alone.
- Use speaker "overlap" for simultaneous speech, interruption, crosstalk, background voice, uncertain speaker, or any segment that may contain both agent and customer audio.
- Split segments tightly at speaker changes. Do not include customer audio inside agent segments.

Return ONLY valid JSON with no markdown:

{{
  "call_id": "{call_id}",
  "agent_name": "{agent_name}",
  "source": "gemini",
  "segments": [
    {{"start": 0.0, "end": 1.0, "speaker": "customer", "text": "..."}},
    {{"start": 1.0, "end": 3.0, "speaker": "agent", "text": "..."}},
    {{"start": 3.0, "end": 3.5, "speaker": "overlap", "text": "..."}}
  ]
}}

Focus on accuracy of speaker identification and pure non-overlapping boundaries. Cover the full audio duration."""


# Persistent browser context dir (saves login session)
USER_DATA_DIR = Path.home() / ".gemini-claude-session"


def extract_json_from_page(page):
    """Extract longest JSON containing call_id+segments from page."""
    return page.evaluate("""() => {
        let best = null;
        let bestLen = 0;
        document.querySelectorAll('div, span, code, pre, p').forEach(el => {
            const text = el.textContent || '';
            if (text.includes('"call_id"') && text.includes('"segments"') && text.includes('"speaker"')) {
                const startIdx = text.indexOf('{');
                if (startIdx >= 0) {
                    let braceCount = 0;
                    for (let i = startIdx; i < text.length; i++) {
                        if (text[i] === '{') braceCount++;
                        if (text[i] === '}') {
                            braceCount--;
                            if (braceCount === 0) {
                                const json = text.substring(startIdx, i + 1);
                                if (json.length > bestLen && json.length < 200000) {
                                    best = json;
                                    bestLen = json.length;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        });
        return best;
    }""")


def wait_for_response(page, max_wait_sec=180):
    """Poll until Gemini stops generating."""
    print(f"  Waiting for Gemini response (max {max_wait_sec}s)...")
    start = time.time()
    last_check = 0
    while time.time() - start < max_wait_sec:
        elapsed = int(time.time() - start)
        is_generating = page.evaluate("""() => {
            const btn = document.querySelector('button[aria-label*="Stop"]');
            return btn !== null && btn.offsetParent !== null;
        }""")

        if not is_generating:
            # Verify we have a response
            has_resp = page.evaluate("""() => {
                const t = document.body.textContent;
                return t.includes('"call_id"') && t.includes('"segments"') && t.includes('"speaker"');
            }""")
            if has_resp:
                print(f"  Response complete in {elapsed}s")
                return True

        if elapsed - last_check >= 15:
            print(f"    Still generating... {elapsed}s")
            last_check = elapsed
        page.wait_for_timeout(2000)
    return False


def transcribe_call(page, audio_path: str, agent_name: str, call_id: str = None):
    """Upload one audio file to Gemini and return JSON labels."""
    audio_path = Path(audio_path).absolute()
    if call_id is None:
        call_id = audio_path.stem

    print(f"\n[Transcribing] {audio_path.name}")
    print(f"  Agent: {agent_name}, Call ID: {call_id}")

    # Navigate to fresh chat
    page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(3000)

    # Click upload menu (+)
    upload_menu = page.locator('button[aria-label*="upload file" i]').first
    upload_menu.wait_for(state="visible", timeout=15000)
    upload_menu.click()
    page.wait_for_timeout(500)

    # Click "Upload files" option and handle file chooser
    print(f"  Uploading {audio_path}...")
    upload_button = page.locator('[data-test-id="local-images-files-uploader-button"]')
    with page.expect_file_chooser() as fc_info:
        upload_button.click()
    file_chooser = fc_info.value
    file_chooser.set_files(str(audio_path))
    page.wait_for_timeout(4000)

    # Type prompt
    prompt = PROMPT_TEMPLATE.format(agent_name=agent_name, call_id=call_id)
    print(f"  Sending prompt ({len(prompt)} chars)...")
    textbox = page.locator('[role="textbox"]').first
    textbox.click()
    textbox.fill(prompt)
    textbox.press("Enter")

    # Wait for response
    if not wait_for_response(page, max_wait_sec=240):
        print(f"  WARNING: Response didn't complete in time")
        return None

    # Extract JSON
    json_str = extract_json_from_page(page)
    if not json_str:
        print(f"  ERROR: Could not find JSON in response")
        return None

    try:
        labels = json.loads(json_str)
        seg_count = len(labels.get('segments', []))
        agent_segs = sum(1 for s in labels.get('segments', []) if s.get('speaker') == 'agent')
        customer_segs = sum(1 for s in labels.get('segments', []) if s.get('speaker') == 'customer')
        print(f"  SUCCESS: {seg_count} segments ({agent_segs} agent, {customer_segs} customer)")
        return labels
    except json.JSONDecodeError as e:
        print(f"  ERROR parsing JSON: {e}")
        print(f"  First 300 chars: {json_str[:300]}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', help='Single audio file path')
    parser.add_argument('--batch', help='Directory with audio files (processes all)')
    parser.add_argument('--agent', default='Agent', help='Agent display name')
    parser.add_argument('--output', help='Output JSON path (single file mode)')
    parser.add_argument('--login', action='store_true', help='Just open browser to login')
    parser.add_argument('--headless', action='store_true', help='Run headless (after login)')
    args = parser.parse_args()

    if not args.audio and not args.batch and not args.login:
        parser.print_help()
        sys.exit(1)

    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Browser session dir: {USER_DATA_DIR}")

    with sync_playwright() as p:
        # Use persistent context to keep login session
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=args.headless,
            args=['--disable-blink-features=AutomationControlled'],
        )
        page = browser.new_page() if not browser.pages else browser.pages[0]

        if args.login:
            print("Opening Gemini for login. Login then close browser when done.")
            page.goto("https://gemini.google.com/app")
            print("Press Ctrl+C in this terminal when logged in...")
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                pass
            browser.close()
            return

        if args.audio:
            labels = transcribe_call(page, args.audio, args.agent)
            if labels:
                output = Path(args.output) if args.output else Path(args.audio).with_suffix('.gemini.json')
                with open(output, 'w') as f:
                    json.dump(labels, f, indent=2)
                print(f"\nSaved to: {output}")

        elif args.batch:
            batch_dir = Path(args.batch)
            audio_files = sorted(
                list(batch_dir.rglob("*.mp3")) +
                list(batch_dir.rglob("*.wav"))
            )
            # Skip already-processed
            audio_files = [a for a in audio_files
                          if not a.with_suffix('.gemini.json').exists()
                          and 'audio_16k' not in a.stem]

            print(f"Found {len(audio_files)} unprocessed audio files in {batch_dir}")

            for i, audio in enumerate(audio_files, 1):
                print(f"\n[{i}/{len(audio_files)}] {audio.name}")
                try:
                    labels = transcribe_call(page, str(audio), args.agent)
                    if labels:
                        output = audio.with_suffix('.gemini.json')
                        with open(output, 'w') as f:
                            json.dump(labels, f, indent=2)
                        print(f"  Saved: {output.name}")
                except Exception as e:
                    print(f"  ERROR: {e}")

        browser.close()
        print("\nDone")


if __name__ == "__main__":
    main()
