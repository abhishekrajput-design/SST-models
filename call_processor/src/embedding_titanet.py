"""
TitaNet-Large Speaker Embedding (NVIDIA NeMo)
State-of-the-art speaker verification model (192-dim)

Paper: TitaNet - Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions
EER on VoxCeleb1: 0.66% (vs CAM++ 0.91%)
"""

from __future__ import annotations

import gc
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import tempfile

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_TARGET_SR = 16000
_MIN_DURATION_S = 1.0
_MIN_SAMPLES = int(_TARGET_SR * _MIN_DURATION_S)


class TitanetEmbedder:
    """NVIDIA TitaNet-Large speaker embedding model.

    Returns 192-dim L2-normalized embeddings from raw audio.
    """

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self, force_cpu: bool = True) -> None:
        if self._loaded:
            return

        try:
            import torch
            import nemo.collections.asr as nemo_asr

            device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"

            logger.info("Loading TitaNet-Large from NeMo (this may take a moment)...")
            self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                "nvidia/speakerverification_en_titanet_large"
            )
            self._model = self._model.to(device)
            self._model.eval()
            self._device = device
            self._loaded = True
            logger.info(f"TitaNet-Large loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load TitaNet: {e}")
            raise

    def embed_chunk(self, audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """Extract 192-dim L2-normalized embedding from audio chunk."""
        if not self._loaded:
            self.load()

        if audio.ndim > 1:
            audio = audio.mean(axis=0) if audio.shape[0] < audio.shape[1] else audio[:, 0]

        # Resample if needed
        if sr != _TARGET_SR:
            try:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=_TARGET_SR)
            except Exception:
                return None

        # Pad short audio
        if len(audio) < _MIN_SAMPLES:
            n_repeats = int(np.ceil(_MIN_SAMPLES / max(len(audio), 1)))
            audio = np.tile(audio, n_repeats)[:_MIN_SAMPLES]

        # Save to temp file (TitaNet API requires file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_path = f.name
            sf.write(tmp_path, audio.astype(np.float32), _TARGET_SR)

        try:
            embedding = self._model.get_embedding(tmp_path)
            emb_np = embedding.cpu().numpy().squeeze()
            # L2 normalize
            n = np.linalg.norm(emb_np)
            if n > 0:
                emb_np = emb_np / n
            return emb_np.astype(np.float32)
        except Exception as e:
            logger.warning(f"TitaNet embedding failed: {e}")
            return None
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def dim(self) -> int:
        return 192

    @property
    def model_name(self) -> str:
        return "titanet_large"


_DEFAULT: Optional[TitanetEmbedder] = None


def get_titanet(force_cpu: bool = True) -> TitanetEmbedder:
    """Get singleton TitaNet embedder."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TitanetEmbedder()
    _DEFAULT.load(force_cpu=force_cpu)
    return _DEFAULT
