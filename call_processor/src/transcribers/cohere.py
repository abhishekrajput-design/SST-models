"""Cohere Transcribe 03-2026 — best WER (5.42%) on Open ASR Leaderboard, March 2026.

Uses the model's own transcribe() method (bypasses generate() entirely) to avoid
generate() API incompatibilities between the local custom class and transformers 5.6.x.

Load via AutoModelForSpeechSeq2Seq + trust_remote_code=True → loads local
CohereAsrForConditionalGeneration which has transcribe(processor, language, audio_arrays, ...).

For timestamps: manually chunk the audio into 30s pieces, pass all as a batch,
map returned strings back to chunk start/end offsets.
"""
from __future__ import annotations
import os
import time
from typing import List, Dict, Any
from .base import BaseTranscriber

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
CHUNK_S = 30  # must be ≤ config.max_audio_clip_s (35) so the model doesn't re-chunk


class CohereTranscriber(BaseTranscriber):
    name = "cohere-transcribe-03-2026"

    def __init__(self, device: str = "cuda", model_dir: str | None = None):
        super().__init__(device=device, model_dir=model_dir)
        self.processor = None

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        cache_dir = self.model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "hf"
        )
        os.makedirs(cache_dir, exist_ok=True)

        import torch
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, cache_dir=cache_dir, trust_remote_code=True
        )
        # float16 halves VRAM (~3 GB vs ~6 GB float32) so the full model fits
        # on a 6 GB GPU without CPU offloading.  device_map="cuda" forces all
        # layers onto GPU; "auto" splits to CPU when VRAM is tight, making
        # inference 50-100× slower.
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        # bfloat16: same memory as float16 (~3 GB) but float32 exponent range
        # so no overflow on audio feature values (float16 max ~65504 is too low).
        dtype = torch.bfloat16 if use_cuda else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            device_map="cuda" if use_cuda else "cpu",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        import numpy as np
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_dur = len(audio) / sr
        chunk_size = int(CHUNK_S * sr)

        # Split into 30 s chunks — each ≤ max_audio_clip_s so no internal re-splitting
        # Skip chunks with < 10% non-silent frames to avoid multilingual hallucinations
        # on background-noise-only segments (common in desk recordings).
        chunks: List[np.ndarray] = []
        offsets: List[float] = []
        for start_sample in range(0, len(audio), chunk_size):
            chunk = audio[start_sample: start_sample + chunk_size]
            if len(chunk) < int(sr * 0.5):
                continue
            non_silent = librosa.effects.split(chunk, top_db=35)
            speech_samples = sum(end - start for start, end in non_silent)
            if speech_samples / len(chunk) < 0.08:
                continue  # skip mostly-silent chunk
            chunks.append(chunk)
            offsets.append(start_sample / sr)

        if not chunks:
            return []

        t0 = time.time()
        # transcribe() handles batching internally; returns list[str], one per chunk
        texts: List[str] = self.model.transcribe(
            self.processor,
            language=language,
            audio_arrays=chunks,
            sample_rates=[sr] * len(chunks),
            punctuation=True,
        )
        elapsed = time.time() - t0
        print(f"  [Cohere] {len(chunks)} chunks in {elapsed:.1f}s")

        out: List[Dict[str, Any]] = []
        for i, (text, start_s) in enumerate(zip(texts, offsets)):
            text = text.strip()
            if not text:
                continue
            end_s = min(start_s + CHUNK_S, total_dur)
            out.append({
                "start": round(start_s, 2),
                "end":   round(end_s, 2),
                "text":  text,
                "speaker": "SPEAKER_00",
                "identified_speaker": "SPEAKER_00",
                "confidence": 0.0,
            })

        return out

    def unload(self) -> None:
        import gc
        try:
            import torch
            if self.model is not None and torch.cuda.is_available():
                try:
                    self.model = self.model.cpu()
                except Exception:
                    pass
        except ImportError:
            pass
        del self.model
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
