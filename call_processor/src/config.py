"""
Centralized configuration for the call processing pipeline.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    """Defaults tuned for a 6GB+ VRAM GPU system."""

    raw_calls_dir: str = os.path.join(PROJECT_ROOT, "data", "raw_calls")
    agent_samples_dir: str = os.path.join(PROJECT_ROOT, "data", "agent_samples")
    processed_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
    embeddings_path: str = os.path.join(PROJECT_ROOT, "embeddings", "agent_embeddings.pkl")
    model_cache_dir: str = os.path.join(PROJECT_ROOT, "models")

    device: str = "cuda"
    whisper_device: str = "auto"

    hf_token: str = ""
    min_segment_duration: float = 1.0
    merge_gap: float = 0.5

    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192

    similarity_threshold: float = 0.25
    unknown_label: str = "Customer"

    whisper_model: str = "large-v3"
    whisper_compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 5
    initial_prompt: str = "CarPlanet car dealership. Agent speaking with customer."
    word_timestamps: bool = True

    target_sample_rate: int = 16000

    def __post_init__(self):
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN", "")


default_config = PipelineConfig()
