"""Deepgram STT API — cloud transcription with speaker diarization.

Models (pass via `model` param on registry key):
  nova-3            — Latest flagship, best accuracy, smart formatting
  nova-2-phonecall  — Optimised for low-bandwidth desk/phone calls
  nova-2-meeting    — Optimised for multi-speaker meetings

API key read from env var DEEPGRAM_API_KEY (set in .env).
No GPU required — pure HTTP.  Returns segments with speaker labels.
"""
from __future__ import annotations
import os
import time
from typing import List, Dict, Any
from .base import BaseTranscriber

API_URL = "https://api.deepgram.com/v1/listen"


class DeepgramTranscriber(BaseTranscriber):

    def __init__(self, model: str = "nova-3", device: str = "cuda",
                 model_dir: str | None = None):
        super().__init__(device=device, model_dir=model_dir)
        self.dg_model = model
        self.name = f"deepgram-{model}"

    def load(self) -> None:
        # No weights to load — verify API key is present
        self.api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY not set. Add it to .env:\n"
                "  DEEPGRAM_API_KEY=your_key_here"
            )
        self.model = True  # sentinel so load() is skipped on repeat calls

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        import urllib.request, urllib.parse, json

        params = urllib.parse.urlencode({
            "model":        self.dg_model,
            "language":     language,
            "diarize":      "true",
            "utterances":   "true",
            "smart_format": "true",
            "punctuate":    "true",
        })
        url = f"{API_URL}?{params}"

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Determine Content-Type from extension
        ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
        mime = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
                "ogg": "audio/ogg", "flac": "audio/flac"}.get(ext, "audio/mpeg")

        req = urllib.request.Request(
            url, data=audio_bytes,
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": mime},
            method="POST",
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
        elapsed = time.time() - t0

        utterances = (result.get("results") or {}).get("utterances") or []
        out: List[Dict[str, Any]] = []
        for u in utterances:
            text = u.get("transcript", "").strip()
            if not text:
                continue
            spk_idx = u.get("speaker", 0)
            spk_id  = f"SPEAKER_{spk_idx:02d}"
            out.append({
                "start":              round(float(u.get("start", 0)), 2),
                "end":                round(float(u.get("end",   0)), 2),
                "text":               text,
                "speaker":            spk_id,
                "identified_speaker": spk_id,
                "confidence":         round(float(u.get("confidence", 0)), 3),
            })

        print(f"  [Deepgram/{self.dg_model}] {len(out)} utterances in {elapsed:.1f}s")
        return out

    def unload(self) -> None:
        self.model = None
        self.api_key = ""
