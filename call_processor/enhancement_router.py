"""
Enhancement router — loads ClearVoice models once at startup and routes
each audio file to the correct tier pipeline based on its DNSMOS quality score.

Models (lazy-loaded singleton, stays in VRAM between calls):
  MossFormerGAN_SE_16K   — aggressive 16 kHz denoiser
  MossFormer2_SE_48K     — lighter 48 kHz denoiser (better quality output)
  MossFormer2_SS_16K     — speech separation → 2 streams
  MossFormer2_SR_48K     — super-resolution: 16 kHz → 48 kHz

Install:  pip install clearvoice
"""
from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_models_lock = threading.Lock()
_models: Optional["ClearVoiceModels"] = None


# --------------------------------------------------------------------------- #
#  Model singleton
# --------------------------------------------------------------------------- #

class ClearVoiceModels:
    """
    Holds all four ClearVoice models in memory simultaneously.
    Instantiate once via ClearVoiceModels.get(); all subsequent calls
    return the cached instance.
    """

    def __init__(self):
        from clearvoice import ClearVoice
        logger.info("Loading ClearVoice models into memory…")
        self.se_16k = ClearVoice(task="speech_enhancement",     model_names=["MossFormerGAN_SE_16K"])
        logger.info("  ✓ MossFormerGAN_SE_16K")
        self.se_48k = ClearVoice(task="speech_enhancement",     model_names=["MossFormer2_SE_48K"])
        logger.info("  ✓ MossFormer2_SE_48K")
        self.ss_16k = ClearVoice(task="speech_separation",      model_names=["MossFormer2_SS_16K"])
        logger.info("  ✓ MossFormer2_SS_16K")
        self.sr_48k = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
        logger.info("  ✓ MossFormer2_SR_48K")
        logger.info("ClearVoice models ready.")

    @classmethod
    def get(cls) -> "ClearVoiceModels":
        global _models
        with _models_lock:
            if _models is None:
                _models = cls()
        return _models


# --------------------------------------------------------------------------- #
#  Public router
# --------------------------------------------------------------------------- #

def route(
    input_path: str,
    output_path: str,
    quality: dict,
    status_cb=None,
) -> dict:
    """
    Route audio through the tier-appropriate enhancement pipeline.

    Args:
        input_path:  Path to raw/FFmpeg-normalized audio.
        output_path: Path to write the final enhanced WAV.
        quality:     Dict from quality_scorer.score_audio() (has 'tier' key).
        status_cb:   Optional callable(stage_name: str) for progress reporting.

    Returns:
        dict with:
            pipeline_used  (str)
            separated_streams (list[str], Tier 4 only — paths to separated WAVs)
    """
    tier = quality.get("tier", 2)

    def _status(msg: str):
        logger.info(f"[Enhancement] {msg}")
        if status_cb:
            status_cb(msg)

    if tier == 1:
        from enhancement_tiers.tier1_good import process
    elif tier == 2:
        from enhancement_tiers.tier2_medium import process
    elif tier == 3:
        from enhancement_tiers.tier3_bad import process
    else:
        from enhancement_tiers.tier4_worst import process

    try:
        models = ClearVoiceModels.get()
    except ImportError:
        logger.warning("clearvoice not installed — skipping ClearVoice enhancement.")
        # Copy input → output as-is so caller always has a valid output path
        import shutil
        shutil.copy2(input_path, output_path)
        return {"pipeline_used": "passthrough_no_clearvoice", "separated_streams": []}

    return process(input_path, output_path, models, _status)


# --------------------------------------------------------------------------- #
#  Audio helpers shared across tier modules
# --------------------------------------------------------------------------- #

def load_16k_mono(path: str) -> tuple[np.ndarray, int]:
    """Load audio as 16 kHz mono float32 numpy array."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import torch
        import torchaudio.functional as F_ta
        t = torch.from_numpy(data).unsqueeze(0)
        t = F_ta.resample(t, sr, 16000)
        data = t.squeeze(0).numpy()
    return data, 16000


def save_wav(data: np.ndarray, sr: int, path: str):
    """Save numpy array as WAV. Clips to [-1, 1] to prevent clipping artifacts."""
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, data, sr, subtype="PCM_16")


def apply_se_16k(models: ClearVoiceModels, audio_np: np.ndarray) -> np.ndarray:
    """Run MossFormerGAN_SE_16K on 16kHz numpy array → returns 16kHz numpy array."""
    out = models.se_16k(input_path=audio_np, online_write=False)
    return _extract_np(out)


def apply_se_48k(models: ClearVoiceModels, audio_np: np.ndarray) -> np.ndarray:
    """Run MossFormer2_SE_48K. Input 48kHz → output 48kHz."""
    out = models.se_48k(input_path=audio_np, online_write=False)
    return _extract_np(out)


def apply_ss_16k(models: ClearVoiceModels, audio_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run MossFormer2_SS_16K (speech separation). Returns (stream1, stream2) at 16kHz."""
    out = models.ss_16k(input_path=audio_np, online_write=False)
    # Output may be list/tuple of 2 arrays, or shape (2, T)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        s1, s2 = _extract_np(out[0]), _extract_np(out[1])
    else:
        arr = _extract_np(out)
        if arr.ndim == 2 and arr.shape[0] == 2:
            s1, s2 = arr[0], arr[1]
        else:
            # Couldn't separate — return same audio for both streams
            logger.warning("SS_16K output unexpected shape; using mono for both streams.")
            s1, s2 = arr, arr.copy()
    return s1, s2


def apply_sr_48k(models: ClearVoiceModels, audio_np: np.ndarray) -> np.ndarray:
    """Run MossFormer2_SR_48K (super-resolution). Input 16kHz → output 48kHz."""
    out = models.sr_48k(input_path=audio_np, online_write=False)
    return _extract_np(out)


def resample_np(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample numpy array using torchaudio."""
    if src_sr == dst_sr:
        return data
    import torch
    import torchaudio.functional as F_ta
    t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    t = F_ta.resample(t, src_sr, dst_sr)
    return t.squeeze(0).numpy()


def apply_metricgan(audio_np: np.ndarray, sr: int, models_dir: str = "models/metricgan") -> np.ndarray:
    """Run SpeechBrain MetricGAN+ on 16kHz mono numpy array → 16kHz output."""
    try:
        import torch
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        from speechbrain.utils.fetching import LocalStrategy

        # Resample to 16kHz if needed (MetricGAN+ expects 16kHz)
        audio_16k = resample_np(audio_np, sr, 16000) if sr != 16000 else audio_np

        enhancer = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir=models_dir,
            local_strategy=LocalStrategy.COPY,
        )
        noisy   = torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0)
        lengths = torch.tensor([1.0])
        with torch.no_grad():
            enhanced = enhancer.enhance_batch(noisy, lengths)

        out = enhanced.squeeze(0).cpu().numpy()
        del enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out
    except Exception as exc:
        logger.warning(f"MetricGAN+ failed: {exc} — returning input unchanged.")
        return audio_np


# --------------------------------------------------------------------------- #
#  Internal
# --------------------------------------------------------------------------- #

def _extract_np(out) -> np.ndarray:
    """Normalise ClearVoice output to a 1-D float32 numpy array."""
    import torch
    if isinstance(out, torch.Tensor):
        arr = out.cpu().numpy()
    elif isinstance(out, np.ndarray):
        arr = out
    else:
        arr = np.array(out, dtype=np.float32)

    arr = arr.squeeze()
    if arr.ndim > 1:
        arr = arr[0]  # take first channel
    return arr.astype(np.float32)
