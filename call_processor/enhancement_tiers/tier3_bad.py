"""
Tier 3 — Bad Audio (mos_ovr 1.5–2.0)

Pipeline:
  load 16kHz
  → MossFormerGAN_SE_16K (first-pass aggressive denoise)
  → Re-score with DNSMOS
      If improved enough (mos_ovr >= 2.5):
          → SR_48K → MetricGAN+ → save
      Else:
          → MossFormer2_SE_48K (second pass, 48kHz)
          → back to 16kHz → MetricGAN+ → SR → save
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

# After first pass, if mos_ovr reaches this threshold we skip the second pass
_SUFFICIENT_MOS = 2.5


def process(input_path: str, output_path: str, models, status_cb) -> dict:
    import io
    from enhancement_router import (
        load_16k_mono, save_wav, resample_np, _extract_np,
        apply_metricgan,
    )
    import tempfile, os

    status_cb("Tier 3: loading audio")
    audio_16k, _ = load_16k_mono(input_path)

    # ── Pass 1: MossFormerGAN_SE_16K ─────────────────────────────────────────
    status_cb("Tier 3: pass 1 — MossFormerGAN_SE_16K")
    pass1_16k = models.se_16k(input_path=audio_16k, online_write=False)
    pass1_16k = _extract_np(pass1_16k)

    # ── Interim DNSMOS check ──────────────────────────────────────────────────
    status_cb("Tier 3: re-scoring after pass 1")
    interim_mos = _score_np(pass1_16k, 16000)
    logger.info(f"Tier 3 interim mos_ovr = {interim_mos:.3f} (threshold {_SUFFICIENT_MOS})")

    if interim_mos >= _SUFFICIENT_MOS:
        status_cb("Tier 3: pass 1 sufficient — SR + MetricGAN+")
        hifi_48k      = models.sr_48k(input_path=pass1_16k, online_write=False)
        hifi_48k      = _extract_np(hifi_48k)
        hifi_16k_mg   = resample_np(hifi_48k, 48000, 16000)
        polished_16k  = apply_metricgan(hifi_16k_mg, sr=16000)
        polished_48k  = resample_np(polished_16k, 16000, 48000)
        pipeline_tag  = "tier3_bad_1pass"
    else:
        # ── Pass 2: MossFormer2_SE_48K (different architecture, second shot) ─
        status_cb("Tier 3: pass 2 — MossFormer2_SE_48K")
        pass1_48k     = resample_np(pass1_16k, 16000, 48000)
        pass2_48k     = models.se_48k(input_path=pass1_48k, online_write=False)
        pass2_48k     = _extract_np(pass2_48k)

        status_cb("Tier 3: SR + MetricGAN+ after pass 2")
        pass2_16k     = resample_np(pass2_48k, 48000, 16000)
        hifi_48k      = models.sr_48k(input_path=pass2_16k, online_write=False)
        hifi_48k      = _extract_np(hifi_48k)
        hifi_16k_mg   = resample_np(hifi_48k, 48000, 16000)
        polished_16k  = apply_metricgan(hifi_16k_mg, sr=16000)
        polished_48k  = resample_np(polished_16k, 16000, 48000)
        pipeline_tag  = "tier3_bad_2pass"

    status_cb("Tier 3: saving output")
    save_wav(polished_48k, 48000, output_path)

    logger.info(f"Tier 3 done ({pipeline_tag}) → {output_path}")
    return {
        "pipeline_used":     pipeline_tag,
        "interim_mos":       round(interim_mos, 3),
        "separated_streams": [],
    }


# --------------------------------------------------------------------------- #
#  Internal helper: score a numpy array without writing a temp file
# --------------------------------------------------------------------------- #

def _score_np(audio_np: np.ndarray, sr: int) -> float:
    """Quick DNSMOS mos_ovr score from a numpy array. Returns 0.0 on failure."""
    try:
        import torch
        import torchaudio.functional as F_ta
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

        data = audio_np.astype(np.float32)
        if sr != 16000:
            t    = torch.from_numpy(data).unsqueeze(0)
            t    = F_ta.resample(t, sr, 16000)
            data = t.squeeze(0).numpy()

        tensor = torch.from_numpy(data).unsqueeze(0)
        dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
        dnsmos.update(tensor)
        raw = dnsmos.compute()

        if isinstance(raw, dict):
            return float(raw.get("mos_ovr", raw.get("ovr", 0)))
        else:
            s = raw.flatten()
            return float(s[3]) if len(s) >= 4 else float(s[0])
    except Exception:
        return 0.0
