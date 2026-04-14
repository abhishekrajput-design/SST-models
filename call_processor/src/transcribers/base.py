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

    def unload(self) -> None:
        """Free GPU memory."""
        import gc
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
