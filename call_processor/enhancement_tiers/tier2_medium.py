"""
Tier 2 — Medium Audio (mos_ovr 2.0–3.0)

Pipeline:
  load 16kHz
  → MossFormerGAN_SE_16K  (aggressive denoising, 16kHz)
  → MossFormer2_SR_48K    (super-resolution: 16kHz → 48kHz)
  → MetricGAN+            (PESQ-optimised polish, 16kHz, then back to 48kHz)
  → save 48kHz WAV
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


def process(input_path: str, output_path: str, models, status_cb) -> dict:
    from enhancement_router import (
        load_16k_mono, save_wav, resample_np, _extract_np,
        apply_metricgan,
    )

    status_cb("Tier 2: loading audio")
    audio_16k, _ = load_16k_mono(input_path)

    # ── Stage 1: aggressive GAN denoising (16kHz) ────────────────────────────
    status_cb("Tier 2: MossFormerGAN_SE_16K — aggressive denoising")
    denoised_16k = models.se_16k(input_path=audio_16k, online_write=False)
    denoised_16k = _extract_np(denoised_16k)

    # ── Stage 2: super-resolution 16kHz → 48kHz ──────────────────────────────
    status_cb("Tier 2: MossFormer2_SR_48K — bandwidth recovery")
    hifi_48k = models.sr_48k(input_path=denoised_16k, online_write=False)
    hifi_48k = _extract_np(hifi_48k)

    # ── Stage 3: MetricGAN+ PESQ polish (needs 16kHz) ────────────────────────
    status_cb("Tier 2: MetricGAN+ — PESQ polish")
    hifi_16k_for_mg = resample_np(hifi_48k, 48000, 16000)
    polished_16k    = apply_metricgan(hifi_16k_for_mg, sr=16000)
    polished_48k    = resample_np(polished_16k, 16000, 48000)

    status_cb("Tier 2: saving output")
    save_wav(polished_48k, 48000, output_path)

    logger.info(f"Tier 2 done → {output_path}")
    return {
        "pipeline_used":     "tier2_medium",
        "separated_streams": [],
    }
