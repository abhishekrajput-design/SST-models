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

    def _call_api(self, audio_bytes: bytes, mime: str, extra_params: dict) -> dict:
        import urllib.request, urllib.parse, json
        params = urllib.parse.urlencode(extra_params)
        req = urllib.request.Request(
            f"{API_URL}?{params}", data=audio_bytes,
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": mime},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
        mime = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
                "ogg": "audio/ogg", "flac": "audio/flac"}.get(ext, "audio/mpeg")

        base_params = {"model": self.dg_model, "language": language,
                       "smart_format": "true", "punctuate": "true"}

        t0 = time.time()
        # First attempt: full diarization + utterances
        result = self._call_api(audio_bytes, mime, {
            **base_params, "diarize": "true", "utterances": "true"
        })
        res = result.get("results") or {}
        utterances = res.get("utterances") or []
        alt0 = ((res.get("channels") or [{}])[0].get("alternatives") or [{}])[0]
        words = alt0.get("words") or []

        # Fallback: some nova-2 variants return empty with diarize=true on
        # heavily processed audio. Retry without diarization.
        if not utterances and not words:
            print(f"  [Deepgram/{self.dg_model}] 0 utterances/words — retrying without diarize")
            result = self._call_api(audio_bytes, mime, base_params)
            res = result.get("results") or {}
            utterances = res.get("utterances") or []
            alt0 = ((res.get("channels") or [{}])[0].get("alternatives") or [{}])[0]
            words = alt0.get("words") or []

        # Log response shape
        utt_count  = len(utterances)
        word_count = len(words)
        print(f"  [Deepgram/{self.dg_model}] utterances={utt_count} words={word_count} "
              f"transcript_len={len(alt0.get('transcript',''))}")

        elapsed = time.time() - t0
        out: List[Dict[str, Any]] = []

        if utterances:
            # Preferred path: utterances with speaker labels
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
        else:
            # Fallback: use words from channels, group by consecutive speaker
            channels = res.get("channels") or []
            alt = (channels[0].get("alternatives") or [{}])[0] if channels else {}
            words = alt.get("words") or []
            if not words:
                # Last resort: whole transcript as one segment
                transcript = alt.get("transcript", "").strip()
                if transcript:
                    out.append({
                        "start":              0.0,
                        "end":                0.0,
                        "text":               transcript,
                        "speaker":            "SPEAKER_00",
                        "identified_speaker": "SPEAKER_00",
                        "confidence":         round(float(alt.get("confidence", 0)), 3),
                    })
            else:
                # Group consecutive words by speaker into utterance-like chunks
                chunks: List[Dict[str, Any]] = []
                cur: Dict[str, Any] = {}
                for w in words:
                    spk = w.get("speaker", 0)
                    if not cur or cur["speaker"] != spk:
                        if cur:
                            chunks.append(cur)
                        cur = {"speaker": spk, "start": w.get("start", 0),
                               "end": w.get("end", 0), "words": [w.get("punctuated_word", w.get("word", ""))]}
                    else:
                        cur["end"] = w.get("end", cur["end"])
                        cur["words"].append(w.get("punctuated_word", w.get("word", "")))
                if cur:
                    chunks.append(cur)
                for c in chunks:
                    text = " ".join(c["words"]).strip()
                    if not text:
                        continue
                    spk_id = f"SPEAKER_{c['speaker']:02d}"
                    out.append({
                        "start":              round(float(c["start"]), 2),
                        "end":                round(float(c["end"]),   2),
                        "text":               text,
                        "speaker":            spk_id,
                        "identified_speaker": spk_id,
                        "confidence":         0.0,
                    })

        print(f"  [Deepgram/{self.dg_model}] {len(out)} segments in {elapsed:.1f}s "
              f"(utterances={len(utterances)}, words={len((((res.get('channels') or [{}])[0]).get('alternatives') or [{}])[0].get('words') or [])})")
        return out

    def unload(self) -> None:
        self.model = None
        self.api_key = ""
