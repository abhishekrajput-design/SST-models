"""
Tier 1 — Good Audio (mos_ovr >= 3.0)

Pipeline:
  load 16kHz → MossFormer2_SE_48K (upsample + light denoise) → save 48kHz WAV

Light touch only. No MetricGAN+ needed.
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


def process(input_path: str, output_path: str, models, status_cb) -> dict:
    """
    Args:
        input_path:  16kHz mono WAV (FFmpeg-normalized)
        output_path: Path to write enhanced 48kHz WAV
        models:      ClearVoiceModels singleton
        status_cb:   callable(stage_name)

    Returns:
        dict with pipeline_used and separated_streams
    """
    from enhancement_router import (
        load_16k_mono, save_wav, resample_np, _extract_np, _to_2d,
        process_in_chunks, remove_silence,
    )

    status_cb("Tier 1: loading audio")
    audio_16k, _ = load_16k_mono(input_path)
    audio_16k, orig_dur = remove_silence(audio_16k, 16000)
    trim_dur = len(audio_16k) / 16000
    if trim_dur < orig_dur - 1:
        status_cb(f"Tier 1: silence removed — {orig_dur:.0f}s → {trim_dur:.0f}s  ({trim_dur/max(orig_dur,1):.0%} retained)")

    # Upsample to 48kHz for SE_48K model
    status_cb("Tier 1: MossFormer2_SE_48K — light denoise + upsample (chunked)")
    audio_48k = resample_np(audio_16k, 16000, 48000)
    enhanced_48k = process_in_chunks(
        audio_np=audio_48k,
        sr=48000,
        fn=lambda chunk: models.se_48k(input_path=chunk, online_write=False),
    )

    status_cb("Tier 1: saving output")
    save_wav(enhanced_48k, 48000, output_path)

    logger.info(f"Tier 1 done → {output_path}")
    return {
        "pipeline_used":     "tier1_good",
        "separated_streams": [],
    }
