"""
Speech transcription using faster-whisper.
Optimized for 4GB VRAM with GPU int8 quantization.
"""

import gc
import logging
from typing import List, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribe audio segments using faster-whisper.

    Supported model_size values (passed directly to faster-whisper):
        Standard:  tiny, base, small, medium, large-v2, large-v3
        Turbo:     large-v3-turbo
        Distil:    distil-large-v3, distil-small.en
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        language: str = "en",
        download_root: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,
    ):
        """
        Args:
            model_size: Whisper model size. See class docstring for options.
            device: 'auto', 'cuda', or 'cpu'.
            compute_type: 'float16' for 6GB+ VRAM, 'int8' for 4GB, 'float32' for CPU.
            language: Language code for transcription.
            initial_prompt: Domain context fed to the decoder to reduce hallucination.
            word_timestamps: Return word-level timestamps.
        """
        self.model_size = model_size
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "float32"
        self.language = language
        self.download_root = download_root
        self.initial_prompt = initial_prompt
        self.word_timestamps = word_timestamps
        self.model = None

    def load_model(self):
        """Load Whisper model via faster-whisper."""
        if self.model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Whisper {self.model_size} "
            f"(device={self.device}, compute_type={self.compute_type})..."
        )

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
        )
        logger.info("Whisper model loaded")

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Whisper model unloaded, VRAM cleared")

    def transcribe_segment(self, audio_path: str) -> str:
        """
        Transcribe a single audio segment.

        Args:
            audio_path: Path to audio file.

        Returns:
            Transcribed text string.
        """
        self.load_model()

        segments_iter, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=1,          # greedy — diarized segments need speed, not beam search
            vad_filter=False,     # segments are already diarized; re-running VAD can drop valid speech
            condition_on_previous_text=False,
            # No initial_prompt — causes hallucination on short/noisy segments
            word_timestamps=False,  # diarization already provides boundaries; word-level DTW is slow
            temperature=0,
            no_speech_threshold=0.6,
        )

        text_parts = []
        for segment in segments_iter:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    def transcribe_segments(
        self,
        segment_paths: List[str],
        segments_metadata: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Transcribe multiple audio segments.

        Args:
            segment_paths: List of audio file paths.
            segments_metadata: Optional list of segment metadata dicts
                               to attach transcriptions to.

        Returns:
            List of dicts with 'text' field added.
        """
        self.load_model()

        results = []
        total = len(segment_paths)

        for i, path in enumerate(segment_paths):
            logger.info(f"Transcribing segment {i + 1}/{total}: {path}")
            try:
                text = self.transcribe_segment(path)
            except Exception as e:
                logger.warning(f"Transcription failed for {path}: {e}")
                text = "[transcription failed]"

            if segments_metadata and i < len(segments_metadata):
                entry = segments_metadata[i].copy()
                entry["text"] = text
            else:
                entry = {"text": text, "segment_file": path}

            results.append(entry)

        return results
