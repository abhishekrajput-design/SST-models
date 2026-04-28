"""
Speaker embedding extraction using WeSpeaker CAM++ (512-dim) with
SpeechBrain ECAPA-TDNN (192-dim) as fallback when wespeaker is not installed.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F_ta

logger = logging.getLogger(__name__)

_TARGET_SR = 16000
# CAM++ needs at least ~1.5s of audio to produce a stable embedding.
# Below this we repeat-pad the chunk up to _MIN_SAMPLES (better than rejecting,
# which leaves short segments unembedded and forces unreliable neighbour-vote
# smoothing in the diariser — a major source of mis-labelling).
_MIN_DURATION_S = 1.5
_MIN_SAMPLES = int(_TARGET_SR * _MIN_DURATION_S)
# Hard floor — clips shorter than this carry too little speech to embed at all.
_REJECT_DURATION_S = 0.3
_REJECT_SAMPLES = int(_TARGET_SR * _REJECT_DURATION_S)

_DEFAULT_MODEL: Optional["EmbeddingModel"] = None


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_norm(a), l2_norm(b)))


class EmbeddingModel:
    def __init__(self) -> None:
        self._wsp = None
        self._ecapa = None
        self._backend: str = ""
        self._device: str = ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, force_cpu: bool = False) -> None:
        if self._backend:
            return

        self._device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"

        try:
            import wespeaker
            campplus_dir = str(Path(__file__).parent.parent / "models" / "campplus_en")
            self._wsp = wespeaker.load_model(campplus_dir)
            # wespeaker.extract_embedding_from_pcm does not move feats to device
            # before calling self.model(feats), so GPU inference silently fails.
            # Always use CPU for CAM++ until wespeaker fixes this.
            self._wsp.set_device("cpu")
            self._backend = "cam++"
            logger.info("CAM++ loaded via wespeaker from %s (CPU)", campplus_dir)
        except Exception as exc:
            logger.info("wespeaker unavailable (%s), falling back to ECAPA-TDNN", exc)
            self._load_ecapa()
            self._backend = "ecapa"

    def unload(self) -> None:
        if self._wsp is not None:
            del self._wsp
            self._wsp = None
        if self._ecapa is not None:
            del self._ecapa
            self._ecapa = None
        self._backend = ""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        logger.info("EmbeddingModel unloaded")

    @property
    def dim(self) -> int:
        return 512 if self._backend == "cam++" else 192

    @property
    def model_name(self) -> str:
        return self._backend

    def embed_file(self, wav_path: str) -> Optional[np.ndarray]:
        self.load()
        try:
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        except Exception as exc:
            logger.warning("embed_file: cannot read %s: %s", wav_path, exc)
            return None

        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        if sr != _TARGET_SR:
            audio = F_ta.resample(
                torch.from_numpy(audio), sr, _TARGET_SR
            ).numpy()

        if len(audio) < _MIN_SAMPLES:
            return None

        if self._backend == "cam++":
            return self._embed_file_campp(wav_path, audio)
        return self._embed_chunk_ecapa(audio)

    def embed_chunk(self, audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        self.load()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        if sr != _TARGET_SR:
            audio = F_ta.resample(
                torch.from_numpy(audio.astype(np.float32)), sr, _TARGET_SR
            ).numpy()
        if len(audio) < _REJECT_SAMPLES:
            return None
        if len(audio) < _MIN_SAMPLES:
            # Repeat-pad short clips up to the model's minimum so they embed
            # instead of being dropped (which forces neighbour-vote smoothing
            # downstream and mis-labels back-channel responses).
            n_repeats = int(np.ceil(_MIN_SAMPLES / max(len(audio), 1)))
            audio = np.tile(audio, n_repeats)[:_MIN_SAMPLES]

        if self._backend == "cam++":
            return self._embed_campp_from_array(audio)
        return self._embed_chunk_ecapa(audio)

    def embed_batch(self, wav_paths: List[str]) -> List[np.ndarray]:
        self.load()
        results = []
        for path in wav_paths:
            emb = self.embed_file(path)
            if emb is not None:
                results.append(emb)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ecapa(self) -> None:
        from speechbrain.inference.speaker import SpeakerRecognition
        savedir = str(Path(__file__).parent.parent / "models" / "ecapa")
        kwargs = dict(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": self._device},
        )
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass
        self._ecapa = SpeakerRecognition.from_hparams(**kwargs)
        self._ecapa.eval()
        logger.info("ECAPA-TDNN loaded on %s", self._device)

    def _embed_chunk_ecapa(self, audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            t = torch.tensor(audio.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                emb = self._ecapa.encode_batch(
                    t.to(self._device)
                ).squeeze().cpu().numpy()
            return l2_norm(emb)
        except Exception as exc:
            logger.warning("ECAPA embed failed: %s", exc)
            return None

    def _embed_file_campp(self, wav_path: str, audio_16k: np.ndarray) -> Optional[np.ndarray]:
        # wespeaker.extract_embedding reads from disk; reuse the original path
        # only when it is already 16 kHz mono — otherwise write a temp file.
        try:
            info = sf.info(wav_path)
            need_tmp = info.samplerate != _TARGET_SR or info.channels != 1
        except Exception:
            need_tmp = True

        if not need_tmp:
            return self._call_campp(wav_path)
        return self._embed_campp_from_array(audio_16k)

    def _embed_campp_from_array(self, audio: np.ndarray) -> Optional[np.ndarray]:
        # wespeaker has no in-memory API, so write a temp WAV and immediately remove it.
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, audio.astype(np.float32), _TARGET_SR)
            return self._call_campp(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _call_campp(self, wav_path: str) -> Optional[np.ndarray]:
        try:
            emb = self._wsp.extract_embedding(wav_path)
            if emb is None:
                return None
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            return l2_norm(np.asarray(emb, dtype=np.float32))
        except Exception as exc:
            logger.warning("CAM++ embed failed: %s", exc)
            return None


def get_model(force_cpu: bool = False) -> EmbeddingModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = EmbeddingModel()
    _DEFAULT_MODEL.load(force_cpu=force_cpu)
    return _DEFAULT_MODEL
