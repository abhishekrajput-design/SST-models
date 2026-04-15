"""Whisper Large-v3-Turbo via faster-whisper (already-cached locally)."""
from __future__ import annotations
import os
import time
from typing import List, Dict, Any
from .base import BaseTranscriber


class WhisperTurboTranscriber(BaseTranscriber):
    name = "whisper-large-v3-turbo"
    supports_word_timestamps = True

    # Map dropdown names → actual faster-whisper model strings
    _MODEL_MAP = {
        "whisper-large-v3-turbo": "large-v3-turbo",
        "whisper-large-v3":       "large-v3",
        "distil-large-v3":        "distil-large-v3",
        "distil-large-v3.5":      "distil-large-v3.5",
    }

    def __init__(self, device: str = "cuda", model_dir: str | None = None,
                 model_size: str = "large-v3-turbo"):
        super().__init__(device=device, model_dir=model_dir)
        self.model_size = self._MODEL_MAP.get(model_size, model_size)

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from faster_whisper import WhisperModel
        compute = "float16" if (self.device == "cuda" and torch.cuda.is_available()) else "float32"
        download_root = self.model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "faster-whisper"
        )
        self.model = WhisperModel(self.model_size, device=self.device,
                                  compute_type=compute, download_root=download_root)

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        self.load()
        t0 = time.time()
        segs_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            word_timestamps=False,
            temperature=0,
            no_speech_threshold=0.6,
        )
        out = []
        for s in segs_iter:
            out.append({
                "start": round(s.start, 2),
                "end":   round(s.end, 2),
                "text":  s.text.strip(),
                "speaker": "SPEAKER_00",
                "identified_speaker": "SPEAKER_00",
                "confidence": round(getattr(s, "avg_logprob", 0.0), 3),
            })
        print(f"  [WhisperTurbo] {len(out)} segments in {time.time()-t0:.1f}s")
        return out

    def unload(self) -> None:
        import gc
        # CTranslate2 has its own CUDA allocator — must explicitly delete
        # the model object so CTranslate2's __del__ fires and frees its pool.
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        self.model = None
        gc.collect()
        gc.collect()   # two passes: first frees Python wrappers, second frees CT2 objects
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
