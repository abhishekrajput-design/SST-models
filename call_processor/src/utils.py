"""
Shared audio utilities for the call processing pipeline.
"""

import os
import gc
import logging
from typing import Optional, Tuple

import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def load_audio(
    audio_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and preprocess to standard format.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, etc.).
        target_sr: Target sample rate (default 16000 Hz).
        mono: Convert to mono if True.

    Returns:
        Tuple of (waveform tensor [channels, time], sample_rate).
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    return waveform, sample_rate


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds."""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def clear_gpu_memory():
    """Force clear GPU VRAM — call between model stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_device_info() -> dict:
    """Get GPU/CPU device information for diagnostics."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 2
        )
        info["vram_allocated_gb"] = round(
            torch.cuda.memory_allocated(0) / (1024 ** 3), 2
        )
        info["vram_reserved_gb"] = round(
            torch.cuda.memory_reserved(0) / (1024 ** 3), 2
        )

    return info


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def list_audio_files(directory: str, extensions: tuple = (".wav", ".mp3", ".flac", ".m4a", ".ogg")) -> list:
    """List all audio files in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ])
