#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def _sanitize_cookies(raw: list[dict]) -> list[dict]:
    out: list[dict] = []
    for cookie in raw:
        item = {
            "name": cookie["name"],
            "value": cookie["value"],
            "domain": cookie["domain"],
            "path": cookie["path"],
        }
        if "secure" in cookie:
            item["secure"] = cookie["secure"]
        if "httpOnly" in cookie:
            item["httpOnly"] = cookie["httpOnly"]
        same_site = cookie.get("sameSite")
        if same_site in {"Lax", "Strict", "None"}:
            item["sameSite"] = same_site
        out.append(item)
    return out


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _find_audio(call_dir: Path) -> Path:
    preferred = ["audio.mp3", "audio.wav", "call.mp3", "audio_16k.wav"]
    for name in preferred:
        path = call_dir / name
        if path.exists():
            return path
    for path in sorted(call_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}:
            return path
    raise FileNotFoundError(f"no audio file found in {call_dir}")


def _build_prompt(agent_name: str, call_id: str) -> str:
    return f"""You are a professional call transcription and speaker identification assistant.
Analyze the uploaded audio file carefully and do the following:
1. Transcribe the entire call from start to end. Do not omit or summarize any parts.
2. Segment the call chronologically by speaker.
3. Identify whether each segment is spoken by the "agent" or the "customer".
4. The agent's exact name is "{agent_name}".
5. Return only valid JSON with no markdown and no explanation.
6. Use this exact structure:

{{
  "call_id": "{call_id}",
  "agent_name": "{agent_name}",
  "source": "gemini",
  "segments": [
    {{
      "start": 0.0,
      "end": 1.5,
      "speaker": "customer",
      "text": "example"
    }},
    {{
      "start": 1.5,
      "end": 3.2,
      "speaker": "agent",
      "text": "example"
    }}
  ]
}}

Return the actual full call content, not placeholders."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--call-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cookies-file", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    args = parser.parse_args()

    call_dir = Path(args.call_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    cookies_file = Path(args.cookies_file).resolve()
    if not call_dir.is_dir():
        raise SystemExit(f"call dir missing: {call_dir}")
    if not cookies_file.is_file():
        raise SystemExit(f"cookies file missing: {cookies_file}")

    audio_path = _find_audio(call_dir)
    original_data = {}
    data_path = call_dir / "data.json"
    if data_path.exists():
        try:
            original_data = json.loads(data_path.read_text(encoding="utf-8-sig"))
        except Exception:
            original_data = {}
    call_id = str(original_data.get("call_id") or call_dir.name).strip()
    prompt = _build_prompt(args.agent_name, call_id)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("playwright is required")

    result_obj: dict | None = None
    raw_response = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(viewport={"width": 1400, "height": 1000})
        cookies = json.loads(cookies_file.read_text(encoding="utf-8"))
        context.add_cookies(_sanitize_cookies(cookies))
        page = context.new_page()
        page.set_default_timeout(60000)
        try:
            page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("[role='textbox'], [contenteditable='true']", timeout=30000)

            upload_btn = page.query_selector("button[aria-label='Upload and tools']")
            if not upload_btn:
                raise RuntimeError("upload button not found")
            upload_btn.click()
            page.wait_for_timeout(1000)

            with page.expect_file_chooser() as fc_info:
                page.click("text=Upload files")
            file_chooser = fc_info.value
            file_chooser.set_files(str(audio_path))
            page.wait_for_timeout(35000)

            input_box = page.query_selector("[role='textbox'], [contenteditable='true']")
            if not input_box:
                raise RuntimeError("input box not found")
            input_box.click()
            page.keyboard.press("Control+A")
            page.keyboard.type(prompt, delay=1)

            existing_messages = page.query_selector_all("message-content, .message-content, .model-response")
            initial_count = len(existing_messages)

            send_btn = page.query_selector("button[aria-label='Send message']")
            if send_btn:
                send_btn.click()
            else:
                page.keyboard.press("Enter")

            previous_len = 0
            stable_ticks = 0
            for _ in range(args.timeout_seconds):
                page.wait_for_timeout(1000)
                messages = page.query_selector_all("message-content, .message-content, .model-response")
                if len(messages) <= initial_count:
                    continue
                last_message = messages[-1]
                raw_response = (last_message.text_content() or "").strip()
                if len(raw_response) > 40 and raw_response == raw_response[:]:
                    parsed = _extract_json(raw_response)
                    if parsed:
                        result_obj = parsed
                if len(raw_response) == previous_len:
                    stable_ticks += 1
                else:
                    stable_ticks = 0
                    previous_len = len(raw_response)
                if result_obj and stable_ticks >= 3:
                    break

            if not result_obj:
                parsed = _extract_json(raw_response)
                if parsed:
                    result_obj = parsed
            if not result_obj:
                raise RuntimeError("could not parse Gemini JSON response")
        finally:
            browser.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    target_audio = out_dir / audio_path.name
    if not target_audio.exists():
        shutil.copy2(audio_path, target_audio)
    result_obj["agent_name"] = args.agent_name
    result_obj["call_id"] = call_id
    result_obj["source"] = "gemini"
    (out_dir / "data.json").write_text(
        json.dumps(result_obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "gemini_response.txt").write_text(raw_response, encoding="utf-8")
    print(f"[ok] wrote {out_dir / 'data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
