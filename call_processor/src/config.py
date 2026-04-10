"""
Centralized configuration for the call processing pipeline.
"""

import os
from dataclasses import dataclass


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    """Defaults tuned for an RTX 3050 4GB system."""

    raw_calls_dir: str = os.path.join(PROJECT_ROOT, "data", "raw_calls")
    agent_samples_dir: str = os.path.join(PROJECT_ROOT, "data", "agent_samples")
    processed_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
    embeddings_path: str = os.path.join(PROJECT_ROOT, "embeddings", "agent_embeddings.pkl")
    model_cache_dir: str = os.path.join(PROJECT_ROOT, "models")

    device: str = "cpu"
    whisper_device: str = "auto"

    hf_token: str = ""
    min_segment_duration: float = 1.0
    merge_gap: float = 0.5

    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192

    similarity_threshold: float = 0.25
    unknown_label: str = "Customer"

    whisper_model: str = "medium"
    whisper_compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5

    target_sample_rate: int = 16000

    def __post_init__(self):
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN", "")


default_config = PipelineConfig()
