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
        load_16k_mono, save_wav, resample_np, _extract_np, _to_2d,
        apply_metricgan, process_in_chunks,
    )

    status_cb("Tier 2: loading audio")
    audio_16k, _ = load_16k_mono(input_path)

    # ── Stage 1: aggressive GAN denoising (16kHz) ────────────────────────────
    status_cb("Tier 2: MossFormerGAN_SE_16K — aggressive denoising (chunked)")
    denoised_16k = process_in_chunks(
        audio_np=audio_16k,
        sr=16000,
        fn=lambda chunk: models.se_16k(input_path=chunk, online_write=False),
    )

    # ── Stage 2: super-resolution 16kHz → 48kHz (chunked) ───────────────────
    import torch
    status_cb("Tier 2: MossFormer2_SR_48K — bandwidth recovery (chunked)")
    chunk_in = int(60.0 * 16000)   # 60s @ 16kHz = 960K samples

    def _run_sr_chunk(chunk):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            out = models.sr_48k(input_path=_to_2d(chunk), online_write=False)
            return _extract_np(out)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "torchscript" in msg or "cuda" in msg or "allocate" in msg:
                logger.warning("SR_48K OOM in Tier 2 — falling back to torchaudio resample for chunk.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return resample_np(chunk, 16000, 48000)
            raise

    n_16k = len(denoised_16k)
    if n_16k <= chunk_in:
        hifi_48k = _run_sr_chunk(denoised_16k)
    else:
        sr_chunks = []
        start = 0
        while start < n_16k:
            end = min(start + chunk_in, n_16k)
            sr_chunks.append(_run_sr_chunk(denoised_16k[start:end]))
            start = end
        hifi_48k = np.concatenate(sr_chunks, axis=0)

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
