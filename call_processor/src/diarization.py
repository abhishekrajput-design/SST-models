"""
Speaker diarization using pyannote.audio.
Detects who spoke when in a call recording.
"""

import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class Diarizer:
    """Handles speaker diarization using pyannote.audio pipeline."""

    def __init__(
        self,
        hf_token: str,
        device: str = "cuda",
        min_segment_duration: float = 1.0,
        merge_gap: float = 0.5,
        cache_dir: Optional[str] = None,
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
        self.cache_dir = cache_dir
        self.pipeline = None

    @staticmethod
    def _hf_cache_root() -> Path:
        hf_home = os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        )
        return Path(hf_home) / "hub"

    @classmethod
    def _find_local_snapshot(cls, model_id: str) -> Optional[Path]:
        cache_name = f"models--{model_id.replace('/', '--')}"
        snapshot_root = cls._hf_cache_root() / cache_name / "snapshots"
        if not snapshot_root.exists():
            return None

        snapshots = [path for path in snapshot_root.iterdir() if path.is_dir()]
        if not snapshots:
            return None

        return max(snapshots, key=lambda path: path.stat().st_mtime)

    @classmethod
    def _build_local_checkpoint(cls, model_id: str):
        snapshot_dir = cls._find_local_snapshot(model_id)
        if snapshot_dir is None:
            return model_id

        config_path = snapshot_dir / "config.yaml"
        if not config_path.exists():
            return str(snapshot_dir)

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        pipeline_params = config.get("pipeline", {}).get("params", {})
        for key in ("embedding", "segmentation", "plda"):
            value = pipeline_params.get(key)
            if isinstance(value, str) and "/" in value:
                local_submodel = cls._find_local_snapshot(value)
                if local_submodel is not None:
                    pipeline_params[key] = str(local_submodel)

        if "plda" not in pipeline_params:
            community_snapshot = cls._find_local_snapshot(
                "pyannote/speaker-diarization-community-1"
            )
            if community_snapshot is not None:
                pipeline_params["plda"] = {
                    "checkpoint": str(community_snapshot),
                    "subfolder": "plda",
                }

        logger.info("Using local HF cache for %s", model_id)
        return config

    def load_model(self):
        """Load pyannote diarization pipeline."""
        if self.pipeline is not None:
            return

        from pyannote.audio import Pipeline

        checkpoint = self._build_local_checkpoint("pyannote/speaker-diarization-3.1")
        logger.info("Loading pyannote diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            checkpoint,
            token=self.hf_token,
            cache_dir=self.cache_dir,
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

    @staticmethod
    def _load_waveform(audio_path: str) -> tuple[torch.Tensor, int]:
        """Load audio with soundfile to avoid torchcodec issues on Windows."""
        waveform_np, sample_rate = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(waveform_np.T).float()
        return waveform, sample_rate

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None,
                min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> List[Dict]:
        """
        Run diarization on an audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: If known (e.g. 2 for typical call), force exact count.
            min_speakers / max_speakers: bounds if not exact.

        Returns:
            List of segments: [{"start": float, "end": float, "speaker": str}, ...]
        """
        self.load_model()

        logger.info(f"Diarizing: {audio_path}  num_speakers={num_speakers}")
        waveform, sample_rate = self._load_waveform(audio_path)
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None: kwargs["min_speakers"] = min_speakers
            if max_speakers is not None: kwargs["max_speakers"] = max_speakers
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **kwargs,
        )
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization

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

        waveform, sample_rate = self._load_waveform(audio_path)

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
