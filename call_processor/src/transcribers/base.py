"""Common interface for all ASR transcribers (Whisper, Cohere, Parakeet, Qwen3, VibeVoice)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTranscriber(ABC):
    """All concrete transcribers return the same segment schema the UI expects."""

    name: str = "base"
    supports_diarization: bool = False
    supports_word_timestamps: bool = False

    def __init__(self, device: str = "cuda", model_dir: str | None = None):
        # Auto-fallback: if CUDA requested but not available, use CPU
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print(f"  [{self.__class__.__name__}] CUDA not available — falling back to CPU")
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.model_dir = model_dir
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (lazy — only when first transcribe is called)."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Transcribe a 16 kHz mono WAV (or any FFmpeg-decodable audio).
        Returns a list of segment dicts with keys:
          - start (float seconds)
          - end (float seconds)
          - text (str)
          - speaker (str, optional — "SPEAKER_00" or model-provided label)
          - identified_speaker (str, optional — same as speaker by default)
          - confidence (float, optional)
        """

    @staticmethod
    def filter_hallucinations(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove hallucinated/empty segments:
        - Pure punctuation ('.' '..' '...' '?' etc.) — silence artefacts
        - Repetition loops ('down down down down...')
        """
        import re
        from collections import Counter
        clean = []
        for seg in segments:
            text = seg.get("text", "").strip()
            # Drop pure-punctuation segments (silence artefacts from Whisper)
            if not re.sub(r'[^\w]', '', text):
                continue
            words = text.lower().split()
            if len(words) < 6:
                clean.append(seg)
                continue
            top_word, top_count = Counter(words).most_common(1)[0]
            if top_count / len(words) > 0.45:
                continue  # one word dominates >45% — hallucination
            # Check for 4+ identical consecutive words
            run, skip = 1, False
            for i in range(1, len(words)):
                if words[i] == words[i-1]:
                    run += 1
                    if run >= 4:
                        skip = True
                        break
                else:
                    run = 1
            if not skip:
                clean.append(seg)
        return clean

    def unload(self) -> None:
        """Free GPU memory — move to CPU first, then delete and flush allocator."""
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
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
