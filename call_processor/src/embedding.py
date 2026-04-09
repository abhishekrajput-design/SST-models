"""
Speaker embedding extraction using SpeechBrain ECAPA-TDNN.
Generates 192-dimensional speaker voice embeddings.
"""

import os
import gc
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


class EmbeddingExtractor:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN pretrained model."""

    def __init__(
        self,
        device: str = "cuda",
        model_dir: str = "./models/spkrec-ecapa",
    ):
        """
        Args:
            device: 'cuda' or 'cpu'.
            model_dir: Local directory to cache the pretrained model.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir
        self.model = None

    def load_model(self):
        """Load SpeechBrain ECAPA-TDNN speaker recognition model."""
        if self.model is not None:
            return

        from speechbrain.inference.speaker import SpeakerRecognition

        logger.info("Loading SpeechBrain ECAPA-TDNN model...")
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self.model_dir,
            run_opts={"device": self.device},
        )
        self.model.eval()
        logger.info(f"ECAPA-TDNN loaded on {self.device}")

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Embedding model unloaded, VRAM cleared")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio to 16kHz mono tensor.

        Uses SpeechBrain's native loader when available (handles format
        conversion automatically), with torchaudio as fallback.
        """
        try:
            # SpeechBrain's load_audio handles resampling + mono internally
            waveform = self.model.load_audio(audio_path)
            # Returns shape (time,) — add batch dim → (1, time)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            return waveform
        except Exception:
            # Fallback: manual loading via torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)

            return waveform

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract a 192-dim speaker embedding from an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            numpy array of shape (192,).
        """
        self.load_model()

        waveform = self._load_audio(audio_path)

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform.to(self.device))

        # Shape: (1, 1, 192) → (192,)
        return embedding.squeeze().cpu().numpy()

    def extract_embeddings_batch(self, audio_paths: List[str]) -> List[np.ndarray]:
        """
        Extract embeddings for multiple audio files.
        Processes one at a time to conserve VRAM.

        Args:
            audio_paths: List of audio file paths.

        Returns:
            List of numpy arrays, each shape (192,).
        """
        self.load_model()

        embeddings = []
        for path in audio_paths:
            try:
                emb = self.extract_embedding(path)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to extract embedding for {path}: {e}")
                embeddings.append(None)

        return embeddings

    def enroll_agent(self, sample_dir: str) -> Optional[np.ndarray]:
        """
        Create an average embedding for an agent from multiple voice samples.

        Args:
            sample_dir: Directory containing WAV files for one agent.

        Returns:
            Average embedding as numpy array (192,), or None if no valid samples.
        """
        sample_dir = Path(sample_dir)
        audio_files = list(sample_dir.glob("*.wav")) + list(sample_dir.glob("*.mp3")) + list(sample_dir.glob("*.flac"))

        if not audio_files:
            logger.warning(f"No audio files found in {sample_dir}")
            return None

        logger.info(f"Enrolling agent from {len(audio_files)} samples in {sample_dir.name}")

        embeddings = []
        for audio_file in audio_files:
            try:
                emb = self.extract_embedding(str(audio_file))
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Skipping {audio_file.name}: {e}")

        if not embeddings:
            return None

        # Average all embeddings to create a robust agent voiceprint
        avg_embedding = np.mean(embeddings, axis=0)

        # L2 normalize the average embedding
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        logger.info(f"Agent {sample_dir.name}: enrolled from {len(embeddings)} samples")
        return avg_embedding

    def enroll_all_agents(
        self,
        agents_dir: str,
        output_path: str = "embeddings/agent_embeddings.pkl",
    ) -> Dict[str, np.ndarray]:
        """
        Process all agent sample directories and save embeddings database.

        Args:
            agents_dir: Parent directory containing agent_NAME/ subdirectories.
            output_path: Path to save the embeddings pickle file.

        Returns:
            Dict mapping agent names to their average embeddings.
        """
        self.load_model()

        agents_dir = Path(agents_dir)
        agent_embeddings = {}

        for agent_folder in sorted(agents_dir.iterdir()):
            if not agent_folder.is_dir():
                continue

            agent_name = agent_folder.name  # e.g., "agent_abhishek"
            embedding = self.enroll_agent(str(agent_folder))

            if embedding is not None:
                agent_embeddings[agent_name] = embedding
                logger.info(f"Enrolled: {agent_name}")
            else:
                logger.warning(f"Skipped: {agent_name} (no valid samples)")

        # Save to pickle
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(agent_embeddings, f)

        logger.info(f"Saved {len(agent_embeddings)} agent embeddings to {output_path}")
        return agent_embeddings
