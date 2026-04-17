"""Groq Whisper — cloud Whisper inference via Groq API.

Same Whisper large-v3 / turbo models as local but runs on Groq's LPU
hardware: ~5-10s for a 30-min audio vs 52s local.

API key: set GROQ_API_KEY in .env
Max file size: 25 MB (Groq limit). Larger files are chunked with ffmpeg.
"""
from __future__ import annotations
import os
import time
import tempfile
import subprocess
from typing import List, Dict, Any
from .base import BaseTranscriber

GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_BYTES = 24 * 1024 * 1024  # 24 MB — stay under Groq's 25 MB limit
CHUNK_S   = 600               # 10-min chunks to stay under size limit


class GroqWhisperTranscriber(BaseTranscriber):
    name = "groq-whisper-large-v3-turbo"

    def __init__(self, model: str = "whisper-large-v3-turbo",
                 device: str = "cuda", model_dir: str | None = None):
        super().__init__(device=device, model_dir=model_dir)
        self.groq_model = model
        self.name = f"groq-{model}"

    def load(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env:\n  GROQ_API_KEY=your_key_here"
            )
        self.model = True

    def _audio_duration(self, path: str) -> float:
        import json
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True
        )
        return float(json.loads(r.stdout)["format"]["duration"])

    def _wav_chunk(self, audio_path: str, start_s: float, duration_s: float) -> str:
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start_s), "-t", str(duration_s),
             "-ac", "1", "-ar", "16000", "-b:a", "32k", tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return tmp

    def _transcribe_chunk(self, chunk_path: str, offset_s: float,
                          language: str) -> List[Dict[str, Any]]:
        import urllib.request, urllib.error, json

        with open(chunk_path, "rb") as f:
            audio_bytes = f.read()

        # multipart/form-data encoding
        boundary = b"----GroqBoundary7a2f"
        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="audio.mp3"\r\n'
            b"Content-Type: audio/mpeg\r\n\r\n"
            + audio_bytes + b"\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            + self.groq_model.encode() + b"\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="response_format"\r\n\r\n'
            b"verbose_json\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="timestamp_granularities[]"\r\n\r\n'
            b"segment\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="language"\r\n\r\n'
            + language.encode() + b"\r\n"
            b"--" + boundary + b"--\r\n"
        )

        req = urllib.request.Request(
            GROQ_API_URL, data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary.decode()}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"Groq API {e.code}: {body[:300]}")

        out = []
        for seg in result.get("segments") or []:
            text = seg.get("text", "").strip()
            if not text:
                continue
            out.append({
                "start":              round(float(seg.get("start", 0)) + offset_s, 2),
                "end":                round(float(seg.get("end",   0)) + offset_s, 2),
                "text":               text,
                "speaker":            "SPEAKER_00",
                "identified_speaker": "SPEAKER_00",
                "confidence":         round(float(seg.get("avg_logprob", 0)), 3),
            })

        # Fallback: whole transcript if no segments returned
        if not out:
            transcript = result.get("text", "").strip()
            if transcript:
                dur = result.get("duration", CHUNK_S)
                out.append({
                    "start": round(offset_s, 2),
                    "end":   round(offset_s + float(dur), 2),
                    "text":  transcript,
                    "speaker": "SPEAKER_00", "identified_speaker": "SPEAKER_00",
                    "confidence": 0.0,
                })
        return out

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()
        total_dur = self._audio_duration(audio_path)
        out: List[Dict[str, Any]] = []

        start = 0.0
        while start < total_dur:
            dur = min(CHUNK_S, total_dur - start)
            if dur < 0.5:
                break
            tmp = self._wav_chunk(audio_path, start, dur)
            try:
                segs = self._transcribe_chunk(tmp, start, language)
                out.extend(segs)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            start += dur

        print(f"  [Groq/{self.groq_model}] {len(out)} segments in {time.time()-t0:.1f}s")
        return out

    def unload(self) -> None:
        self.model = None
        self.api_key = ""
