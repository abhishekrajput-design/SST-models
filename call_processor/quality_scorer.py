"""
DNSMOS quality scoring and tier classification.

Scores audio using Microsoft's Deep Noise Suppression MOS model (via torchmetrics).
Runs entirely on CPU via ONNX Runtime — fast, no GPU needed.

Install:  pip install torchmetrics[audio]

Quality Tiers (based on mos_ovr):
  Tier 1 (Good):   mos_ovr >= 3.0  -> Light enhancement
  Tier 2 (Medium): mos_ovr 2.0-3.0 -> Standard pipeline
  Tier 3 (Bad):    mos_ovr 1.5-2.0 -> Aggressive multi-pass
  Tier 4 (Worst):  mos_ovr < 1.5   -> Generative reconstruction
"""
from __future__ import annotations

import logging
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Tier thresholds
# --------------------------------------------------------------------------- #
TIER_GOOD   = 3.0   # mos_ovr >= 3.0  → Tier 1
TIER_MEDIUM = 2.0   # mos_ovr 2.0-3.0 → Tier 2
TIER_BAD    = 1.5   # mos_ovr 1.5-2.0 → Tier 3
                    # mos_ovr < 1.5   → Tier 4

TIER_NAMES  = {1: "good",   2: "medium",        3: "bad",        4: "worst"}
TIER_LABELS = {1: "Tier 1 — Good", 2: "Tier 2 — Medium",
               3: "Tier 3 — Bad",  4: "Tier 4 — Worst"}
TIER_COLORS = {1: "green",  2: "yellow",         3: "orange",     4: "red"}


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def score_audio(audio_path: str) -> dict:
    """
    Score audio quality using DNSMOS.

    Args:
        audio_path: Path to any audio file (any format soundfile supports).

    Returns:
        dict with keys:
            p808_mos, mos_sig, mos_bak, mos_ovr  (float, 1.0–5.0 scale)
            tier         (int, 1–4)
            tier_name    (str, "good"/"medium"/"bad"/"worst")
            tier_label   (str, human-readable label)
            tier_color   (str, CSS color name)
            error        (str, optional — present only on failure)
    """
    try:
        import torch
        import torchaudio.functional as F_ta
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

        # ── Load + convert to 16 kHz mono float32 ────────────────────────────
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo → mono

        if sr != 16000:
            t = torch.from_numpy(data).unsqueeze(0)  # [1, T]
            t = F_ta.resample(t, sr, 16000)
            data = t.squeeze(0).numpy()

        # ── Run DNSMOS (CPU / ONNX) ───────────────────────────────────────────
        audio_tensor = torch.from_numpy(data).unsqueeze(0)  # [1, T]

        dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
        dnsmos.update(audio_tensor)
        raw = dnsmos.compute()

        # torchmetrics may return dict or tensor depending on version
        if isinstance(raw, dict):
            p808_mos = float(raw.get("p808_mos", raw.get("overall", 0)))
            mos_sig  = float(raw.get("mos_sig",  raw.get("sig", 0)))
            mos_bak  = float(raw.get("mos_bak",  raw.get("bak", 0)))
            mos_ovr  = float(raw.get("mos_ovr",  raw.get("ovr", 0)))
        else:
            # Tensor shape: [4] or [1, 4]
            s        = raw.flatten()
            p808_mos = float(s[0])
            mos_sig  = float(s[1])
            mos_bak  = float(s[2])
            mos_ovr  = float(s[3]) if len(s) >= 4 else float(s[0])

    except ImportError:
        logger.warning("torchmetrics[audio] not installed — DNSMOS unavailable. Defaulting to Tier 2.")
        return _default_result(2, "torchmetrics[audio] not installed")
    except Exception as exc:
        logger.warning(f"DNSMOS scoring failed: {exc} — defaulting to Tier 2.")
        return _default_result(2, str(exc))

    tier = _classify_tier(mos_ovr)
    logger.info(
        f"DNSMOS  p808={p808_mos:.2f}  sig={mos_sig:.2f}  "
        f"bak={mos_bak:.2f}  ovr={mos_ovr:.2f}  → {TIER_LABELS[tier]}"
    )
    return {
        "p808_mos":   round(p808_mos, 3),
        "mos_sig":    round(mos_sig,  3),
        "mos_bak":    round(mos_bak,  3),
        "mos_ovr":    round(mos_ovr,  3),
        "tier":       tier,
        "tier_name":  TIER_NAMES[tier],
        "tier_label": TIER_LABELS[tier],
        "tier_color": TIER_COLORS[tier],
    }


def compute_enhancement_gain(pre: dict, post: dict) -> float:
    """Return mos_ovr gain (post - pre), clamped to reasonable range."""
    return round(
        max(-5.0, min(5.0, post.get("mos_ovr", 0) - pre.get("mos_ovr", 0))),
        3,
    )


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #

def _classify_tier(mos_ovr: float) -> int:
    if mos_ovr >= TIER_GOOD:
        return 1
    elif mos_ovr >= TIER_MEDIUM:
        return 2
    elif mos_ovr >= TIER_BAD:
        return 3
    else:
        return 4


def _default_result(tier: int, error: str = "") -> dict:
    return {
        "p808_mos":   0.0,
        "mos_sig":    0.0,
        "mos_bak":    0.0,
        "mos_ovr":    0.0,
        "tier":       tier,
        "tier_name":  TIER_NAMES[tier],
        "tier_label": TIER_LABELS[tier],
        "tier_color": TIER_COLORS[tier],
        "error":      error,
    }
