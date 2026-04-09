"""
Speaker diarization using pyannote.audio.
Detects who spoke when in a call recording.
"""

import os
import gc
import logging
from typing import List, Dict, Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


class Diarizer:
    """Handles speaker diarization using pyannote.audio pipeline."""

    def __init__(
        self,
        hf_token: str,
        device: str = "cuda",
        min_segment_duration: float = 1.0,
        merge_gap: float = 0.5,
    ):
        """
        Args:
            hf_token: HuggingFace token (required for pyannote models).
            device: 'cuda' or 'cpu'.
            min_segment_duration: Skip segments shorter than this (seconds).
            merge_gap: Merge same-speaker segments closer than this gap (seconds).
        """
        self.hf_token = hf_token
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.min_segment_duration = min_segment_duration
        self.merge_gap = merge_gap
        self.pipeline = None

    def load_model(self):
        """Load pyannote diarization pipeline."""
        if self.pipeline is not None:
            return

        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )
        self.pipeline.to(self.device)
        logger.info(f"Diarization pipeline loaded on {self.device}")

    def unload_model(self):
        """Free GPU memory by unloading the model."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Diarization model unloaded, VRAM cleared")

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Run diarization on an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of segments: [{"start": float, "end": float, "speaker": str}, ...]
        """
        self.load_model()

        logger.info(f"Diarizing: {audio_path}")
        diarization = self.pipeline(audio_path)

        # Extract raw segments
        raw_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            })

        logger.info(f"Raw segments: {len(raw_segments)}")

        # Merge consecutive same-speaker segments
        merged = self._merge_segments(raw_segments)
        logger.info(f"After merging: {len(merged)}")

        # Filter short segments
        filtered = [
            s for s in merged
            if (s["end"] - s["start"]) >= self.min_segment_duration
        ]
        logger.info(f"After filtering (<{self.min_segment_duration}s): {len(filtered)}")

        return filtered

    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge consecutive segments from the same speaker if gap < merge_gap."""
        if not segments:
            return []

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            if seg["speaker"] == prev["speaker"] and gap <= self.merge_gap:
                prev["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        return merged

    def save_segments(
        self,
        audio_path: str,
        segments: List[Dict],
        output_dir: str,
    ) -> List[str]:
        """
        Extract each diarized segment as a separate WAV file.

        Args:
            audio_path: Path to original audio.
            segments: Diarization segments.
            output_dir: Directory to save segment WAVs.

        Returns:
            List of saved WAV file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        saved_paths = []
        for i, seg in enumerate(segments):
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            chunk = waveform[:, start_sample:end_sample]

            filename = f"segment_{i:04d}_{seg['speaker']}_{seg['start']:.1f}_{seg['end']:.1f}.wav"
            filepath = os.path.join(output_dir, filename)

            sf.write(filepath, chunk.squeeze().numpy(), sample_rate)
            saved_paths.append(filepath)

        logger.info(f"Saved {len(saved_paths)} segment WAVs to {output_dir}")
        return saved_paths
