"""
Centralized configuration for the call processing pipeline.
All default paths, thresholds, and model settings in one place.
"""

import os
from dataclasses import dataclass, field


# Project root — always relative to this file's parent directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    """Full pipeline configuration with sensible defaults for RTX 3050 (4GB VRAM)."""

    # ── Paths ──────────────────────────────────────────────────
    raw_calls_dir: str = os.path.join(PROJECT_ROOT, "data", "raw_calls")
    agent_samples_dir: str = os.path.join(PROJECT_ROOT, "data", "agent_samples")
    processed_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
    embeddings_path: str = os.path.join(PROJECT_ROOT, "embeddings", "agent_embeddings.pkl")
    model_cache_dir: str = os.path.join(PROJECT_ROOT, "models")

    # ── Device ─────────────────────────────────────────────────
    device: str = "cuda"  # "cuda" or "cpu"

    # ── Diarization (pyannote) ─────────────────────────────────
    hf_token: str = ""  # Set via HF_TOKEN env var or CLI
    min_segment_duration: float = 1.0  # Skip segments shorter than this (seconds)
    merge_gap: float = 0.5  # Merge same-speaker segments closer than this (seconds)

    # ── Speaker Embedding (SpeechBrain ECAPA-TDNN) ─────────────
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192

    # ── Speaker Matching ───────────────────────────────────────
    similarity_threshold: float = 0.25  # Below this → "Customer"
    unknown_label: str = "Customer"

    # ── Transcription (Whisper via faster-whisper) ─────────────
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8"  # int8 for 4GB VRAM, float16 for 8GB+
    language: str = "en"
    beam_size: int = 5

    # ── Audio Processing ───────────────────────────────────────
    target_sample_rate: int = 16000

    def __post_init__(self):
        """Resolve HF token from environment if not set."""
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN", "")


# Singleton config — import and use directly
default_config = PipelineConfig()
