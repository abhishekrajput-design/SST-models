#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import requests


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


def _extract_json_block(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = cleaned[start : idx + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


def _normalize_speaker(raw: str, agent_name: str) -> str:
    value = str(raw or "").strip().lower()
    agent_norm = agent_name.strip().lower()
    if value == "agent" or value == agent_norm or agent_norm in value:
        return "agent"
    return "customer"


def _normalize_segments(raw_segments: list[dict], agent_name: str) -> list[dict]:
    out: list[dict] = []
    for seg in raw_segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except Exception:
            continue
        if end <= start:
            continue
        text = seg.get("text")
        if text is None:
            text = seg.get("transcript")
        text = str(text or "").strip()
        if not text:
            continue
        out.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": _normalize_speaker(seg.get("speaker", ""), agent_name),
                "text": text,
            }
        )
    return out


def _build_prompt(agent_name: str, call_id: str) -> str:
    return f"""Transcribe this call and identify the speaker for each segment.
The agent's exact name is "{agent_name}".
Return only valid JSON with no markdown, no commentary, and no code fences.
Use exactly this schema:
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
Every segment speaker must be either exactly "agent" or exactly "customer".
Use "text" as the field name, not "transcript"."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--call-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--mime-type", default="")
    args = parser.parse_args()

    call_dir = Path(args.call_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not call_dir.is_dir():
        raise SystemExit(f"call dir missing: {call_dir}")

    input_data = {}
    input_json = call_dir / "data.json"
    if input_json.exists():
        try:
            input_data = json.loads(input_json.read_text(encoding="utf-8-sig"))
        except Exception:
            input_data = {}
    call_id = str(input_data.get("call_id") or call_dir.name).strip()
    audio_path = _find_audio(call_dir)

    mime_type = args.mime_type.strip()
    if not mime_type:
        mime_type = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
        }.get(audio_path.suffix.lower(), "application/octet-stream")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _build_prompt(args.agent_name, call_id)},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                        }
                    },
                ]
            }
        ]
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{args.model}:generateContent"
    )
    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": args.api_key,
        },
        json=payload,
        timeout=900,
    )
    response.raise_for_status()
    response_json = response.json()
    parts = (
        response_json.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
    parsed = _extract_json_block(text)
    if not parsed:
        raise SystemExit("Gemini response did not contain parseable JSON")

    normalized = {
        "call_id": call_id,
        "agent_name": args.agent_name,
        "source": "gemini",
        "segments": _normalize_segments(list(parsed.get("segments") or []), args.agent_name),
    }
    if not normalized["segments"]:
        raise SystemExit("Gemini response contained no usable segments")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_audio = output_dir / audio_path.name
    if not target_audio.exists():
        shutil.copy2(audio_path, target_audio)
    (output_dir / "data.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "gemini_raw_response.json").write_text(
        json.dumps(response_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ok] wrote {output_dir / 'data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
