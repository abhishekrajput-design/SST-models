"""Microsoft VibeVoice-ASR (9B) — long-form (60 min) + native speaker diarization."""
from __future__ import annotations
import os
import time
from typing import List, Dict, Any
from .base import BaseTranscriber

MODEL_ID = "microsoft/VibeVoice-ASR"


class VibeVoiceAsrTranscriber(BaseTranscriber):
    name = "vibevoice-asr"
    supports_diarization = True
    supports_word_timestamps = True

    def __init__(self, device: str = "cuda", model_dir: str | None = None,
                 load_in_4bit: bool = False):
        super().__init__(device=device, model_dir=model_dir)
        self.processor = None
        self.load_in_4bit = load_in_4bit

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModel
        cache_dir = self.model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "hf"
        )
        os.makedirs(cache_dir, exist_ok=True)
        kwargs = {"cache_dir": cache_dir, "trust_remote_code": True}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32
        # Try AutoProcessor first; fall back to WhisperProcessor if custom class unrecognised
        processor_loaded = False
        for proc_fn in [
            lambda: AutoProcessor.from_pretrained(
                MODEL_ID, cache_dir=cache_dir, trust_remote_code=True),
            lambda: __import__('transformers').WhisperProcessor.from_pretrained(
                MODEL_ID, cache_dir=cache_dir, trust_remote_code=True),
            lambda: __import__('transformers').AutoFeatureExtractor.from_pretrained(
                MODEL_ID, cache_dir=cache_dir, trust_remote_code=True),
        ]:
            try:
                self.processor = proc_fn()
                processor_loaded = True
                break
            except Exception as pe:
                print(f"  [VibeVoice] processor attempt failed: {pe}")
        if not processor_loaded:
            raise RuntimeError("Could not load any processor for VibeVoice-ASR")
        self.model = AutoModel.from_pretrained(MODEL_ID, **kwargs)
        if not self.load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        import torch
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        t0 = time.time()
        inputs = self.processor(audio=audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=4096)
        # VibeVoice returns "Who/When/What" structured output — parse if present
        raw_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        out = self._parse_vibevoice_output(raw_text)
        print(f"  [VibeVoice] {len(out)} segments in {time.time()-t0:.1f}s")
        return out

    @staticmethod
    def _parse_vibevoice_output(raw: str) -> List[Dict[str, Any]]:
        """VibeVoice format: '[SPEAKER_X] [00:01.5-00:04.2] text...' on each line."""
        import re
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        pat = re.compile(
            r"\[?(SPEAKER[_\s]*\d+|[A-Z]+)\]?\s*\[?(\d+:?\d*\.?\d*)\s*[-–]\s*(\d+:?\d*\.?\d*)\]?\s*(.*)"
        )
        out: List[Dict[str, Any]] = []
        for ln in lines:
            m = pat.match(ln)
            if not m:
                # Fallback: append as single segment with no speaker
                if ln:
                    out.append({"start": 0.0, "end": 0.0, "text": ln,
                                "speaker": "SPEAKER_00", "identified_speaker": "SPEAKER_00",
                                "confidence": 0.0})
                continue
            spk, st, en, text = m.groups()
            def t2s(t):
                if ":" in t:
                    a, b = t.split(":")
                    return float(a) * 60 + float(b)
                return float(t)
            out.append({
                "start": round(t2s(st), 2),
                "end":   round(t2s(en), 2),
                "text":  text.strip(),
                "speaker": spk.replace(" ", "_"),
                "identified_speaker": spk.replace(" ", "_"),
                "confidence": 0.0,
            })
        return out
