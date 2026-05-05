"""
Speaker embedding extraction using SpeechBrain ECAPA-TDNN.
Generates 192-dimensional speaker voice embeddings.
"""

import gc
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
MAX_EMBEDDING_AUDIO_SECONDS = 30


class EmbeddingExtractor:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN."""

    def __init__(
        self,
        device: str = "cuda",
        model_dir: str = "./models/spkrec-ecapa",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir
        self.model = None

    def load_model(self):
        """Load the pretrained SpeechBrain speaker recognition model."""
        if self.model is not None:
            return

        from speechbrain.inference.speaker import SpeakerRecognition

        logger.info("Loading SpeechBrain ECAPA-TDNN model...")
        kwargs = dict(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self.model_dir,
            run_opts={"device": self.device},
        )
        # LocalStrategy.COPY available only in newer SpeechBrain versions
        try:
            from speechbrain.utils.fetching import FetchConfig, LocalStrategy
            kwargs["local_strategy"] = LocalStrategy.COPY
            kwargs["fetch_config"] = FetchConfig(allow_network=False)
        except ImportError:
            pass

        self.model = SpeakerRecognition.from_hparams(**kwargs)
        self.model.eval()
        logger.info("ECAPA-TDNN loaded on %s", self.device)

    def unload_model(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as exc:
                logger.warning("GPU cleanup skipped after CUDA error: %s", exc)
        logger.info("Embedding model unloaded, VRAM cleared")

    def _load_audio(self, audio_path: str, channel: Optional[int] = None) -> torch.Tensor:
        """Load an audio file as 16kHz waveform, optionally selecting one channel."""
        waveform_np, sample_rate = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(waveform_np.T).float()

        if channel is not None and waveform.shape[0] > 1:
            channel = max(0, min(channel, waveform.shape[0] - 1))
            waveform = waveform[channel:channel + 1, :]
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        max_samples = TARGET_SAMPLE_RATE * MAX_EMBEDDING_AUDIO_SECONDS
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        return waveform

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _candidate_score(self, embeddings: List[np.ndarray], expected_count: int) -> float:
        if len(embeddings) < 2 or expected_count <= 0:
            return 0.0

        scores = []
        for idx in range(len(embeddings)):
            for jdx in range(idx + 1, len(embeddings)):
                scores.append(self._cosine_similarity(embeddings[idx], embeddings[jdx]))

        coverage = len(embeddings) / expected_count
        return float(np.mean(scores)) * coverage if scores else 0.0

    def extract_embedding(self, audio_path: str, channel: Optional[int] = None) -> np.ndarray:
        """Extract a 192-dimensional speaker embedding from an audio file."""
        self.load_model()
        waveform = self._load_audio(audio_path, channel=channel)

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform.to(self.device))

        return embedding.squeeze().cpu().numpy()

    def extract_embeddings_batch(self, audio_paths: List[str]) -> List[Optional[np.ndarray]]:
        """Extract embeddings one file at a time to conserve VRAM."""
        self.load_model()

        embeddings: List[Optional[np.ndarray]] = []
        for path in audio_paths:
            try:
                embeddings.append(self.extract_embedding(path))
            except Exception as exc:
                logger.warning("Failed to extract embedding for %s: %s", path, exc)
                embeddings.append(None)

        return embeddings

    def enroll_agent(self, sample_dir: str) -> Optional[np.ndarray]:
        """Average multiple samples into a single normalized agent embedding."""
        sample_path = Path(sample_dir)
        audio_files = (
            list(sample_path.glob("*.wav"))
            + list(sample_path.glob("*.mp3"))
            + list(sample_path.glob("*.flac"))
        )

        if not audio_files:
            logger.warning("No audio files found in %s", sample_path)
            return None

        logger.info(
            "Enrolling agent from %s samples in %s",
            len(audio_files),
            sample_path.name,
        )

        candidate_channels: List[Optional[int]] = [None]
        if any(sf.info(str(audio_file)).channels > 1 for audio_file in audio_files):
            candidate_channels.extend([0, 1])

        best_channel: Optional[int] = None
        best_score = float("-inf")
        best_embeddings: List[np.ndarray] = []

        for candidate_channel in candidate_channels:
            channel_embeddings: List[np.ndarray] = []
            for audio_file in audio_files:
                try:
                    channel_embeddings.append(
                        self.extract_embedding(str(audio_file), channel=candidate_channel)
                    )
                except Exception as exc:
                    logger.warning("Skipping %s: %s", audio_file.name, exc)

            score = self._candidate_score(channel_embeddings, len(audio_files))
            if score > best_score:
                best_channel = candidate_channel
                best_score = score
                best_embeddings = channel_embeddings

        embeddings = best_embeddings

        if not embeddings:
            return None

        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        channel_label = "mono" if best_channel is None else f"channel {best_channel}"
        logger.info(
            "Agent %s: enrolled from %s samples using %s (score=%.4f)",
            sample_path.name,
            len(embeddings),
            channel_label,
            best_score,
        )
        return avg_embedding

    def enroll_all_agents(
        self,
        agents_dir: str,
        output_path: str = "embeddings/agent_embeddings.pkl",
    ) -> Dict[str, np.ndarray]:
        """Process all agent sample folders and save the embedding database."""
        self.load_model()

        agents_path = Path(agents_dir)
        agent_embeddings: Dict[str, np.ndarray] = {}

        for agent_folder in sorted(agents_path.iterdir()):
            if not agent_folder.is_dir():
                continue

            agent_name = agent_folder.name
            embedding = self.enroll_agent(str(agent_folder))

            if embedding is not None:
                agent_embeddings[agent_name] = embedding
                logger.info("Enrolled: %s", agent_name)
            else:
                logger.warning("Skipped: %s (no valid samples)", agent_name)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as handle:
            pickle.dump(agent_embeddings, handle)

        logger.info(
            "Saved %s agent embeddings to %s",
            len(agent_embeddings),
            output_path,
        )
        return agent_embeddings
