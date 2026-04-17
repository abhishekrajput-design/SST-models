"""AssemblyAI Universal-2 — cloud ASR with native speaker diarization.

Best for call-center audio: 30% fewer hallucinations than Whisper,
21% better alphanumeric accuracy (IDs, phone numbers, prices).

API key: set ASSEMBLYAI_API_KEY in .env
Flow: upload → submit → poll → parse utterances (async REST API).
"""
from __future__ import annotations
import os
import time
from typing import List, Dict, Any
from .base import BaseTranscriber

UPLOAD_URL     = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
CHUNK = 5 * 1024 * 1024  # 5 MB upload chunks


class AssemblyAITranscriber(BaseTranscriber):
    name = "assemblyai-universal-2"

    def load(self) -> None:
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY not set. Add it to .env:\n"
                "  ASSEMBLYAI_API_KEY=your_key_here\n"
                "  Get a key at https://www.assemblyai.com"
            )
        self.model = True

    def _headers(self) -> dict:
        return {"authorization": self.api_key, "content-type": "application/json"}

    def _upload(self, audio_path: str) -> str:
        import urllib.request
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        req = urllib.request.Request(
            UPLOAD_URL, data=audio_bytes,
            headers={"authorization": self.api_key, "content-type": "application/octet-stream"},
            method="POST",
        )
        import json
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["upload_url"]

    def _submit(self, upload_url: str, language: str) -> str:
        import urllib.request, json
        body = json.dumps({
            "audio_url":       upload_url,
            "speech_model":    "universal",   # Universal-2
            "speaker_labels":  True,
            "punctuate":       True,
            "format_text":     True,
            "language_code":   language,
        }).encode()
        req = urllib.request.Request(
            TRANSCRIPT_URL, data=body,
            headers=self._headers(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["id"]

    def _poll(self, transcript_id: str, timeout_s: int = 600) -> dict:
        import urllib.request, json
        url = f"{TRANSCRIPT_URL}/{transcript_id}"
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise RuntimeError(f"AssemblyAI error: {data.get('error')}")
            time.sleep(3)
        raise TimeoutError(f"AssemblyAI transcript timed out after {timeout_s}s")

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()

        print(f"  [AssemblyAI] Uploading {os.path.basename(audio_path)}...")
        upload_url = self._upload(audio_path)

        print(f"  [AssemblyAI] Submitting transcript job...")
        tid = self._submit(upload_url, language)

        print(f"  [AssemblyAI] Polling transcript {tid}...")
        data = self._poll(tid)

        out: List[Dict[str, Any]] = []
        utterances = data.get("utterances") or []

        if utterances:
            for u in utterances:
                text = u.get("text", "").strip()
                if not text:
                    continue
                spk = f"SPEAKER_{u.get('speaker', 'A')}"
                out.append({
                    "start":              round(u.get("start", 0) / 1000, 2),  # ms → s
                    "end":                round(u.get("end",   0) / 1000, 2),
                    "text":               text,
                    "speaker":            spk,
                    "identified_speaker": spk,
                    "confidence":         round(float(u.get("confidence", 0)), 3),
                })
        else:
            # Fallback: whole transcript as one segment
            transcript = (data.get("text") or "").strip()
            if transcript:
                dur = data.get("audio_duration") or 0
                out.append({
                    "start": 0.0, "end": round(float(dur), 2),
                    "text": transcript,
                    "speaker": "SPEAKER_00", "identified_speaker": "SPEAKER_00",
                    "confidence": round(float(data.get("confidence", 0)), 3),
                })

        elapsed = time.time() - t0
        print(f"  [AssemblyAI] {len(out)} segments in {elapsed:.1f}s")
        return out

    def unload(self) -> None:
        self.model = None
        self.api_key = ""
