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
    import torch
    import tempfile, os
    from enhancement_router import (
        load_16k_mono, save_wav, resample_np, _extract_np, _to_2d,
        apply_metricgan, process_in_chunks,
    )

    def _empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    status_cb("Tier 3: loading audio")
    audio_16k, _ = load_16k_mono(input_path)

    # ── Pass 1: MossFormerGAN_SE_16K ─────────────────────────────────────────
    n_chunks_p1 = max(1, int(len(audio_16k) / 16000 / 60.0))
    status_cb(f"Tier 3: pass 1 — MossFormerGAN_SE_16K (~{n_chunks_p1} chunks)")
    pass1_16k = process_in_chunks(
        audio_np=audio_16k,
        sr=16000,
        fn=lambda chunk: models.se_16k(input_path=chunk, online_write=False),
        chunk_sec=60.0,
        progress_cb=lambda i, n, s, e: status_cb(
            f"Tier 3: pass 1 — chunk {i}/{n}  ({s:.0f}s–{e:.0f}s)"
        ),
    )
    _empty_cache()

    # ── Interim DNSMOS check ──────────────────────────────────────────────────
    status_cb("Tier 3: re-scoring after pass 1")
    interim_mos = _score_np(pass1_16k, 16000)
    logger.info(f"Tier 3 interim mos_ovr = {interim_mos:.3f} (threshold {_SUFFICIENT_MOS})")
    status_cb(f"Tier 3: interim MOS = {interim_mos:.3f}  (need ≥ {_SUFFICIENT_MOS} for 1-pass)")

    if interim_mos >= _SUFFICIENT_MOS:
        status_cb("Tier 3: pass 1 sufficient — SR (chunked) + MetricGAN+")
        hifi_48k      = _sr_chunked(models, pass1_16k, status_cb=status_cb)
        hifi_16k_mg   = resample_np(hifi_48k, 48000, 16000)
        polished_16k  = apply_metricgan(hifi_16k_mg, sr=16000)
        polished_48k  = resample_np(polished_16k, 16000, 48000)
        pipeline_tag  = "tier3_bad_1pass"
    else:
        # ── Pass 2: MossFormer2_SE_48K (different architecture, second shot) ─
        n_chunks_p2 = max(1, int(len(pass1_16k) / 16000 / 20.0))
        status_cb(f"Tier 3: pass 2 — MossFormer2_SE_48K (~{n_chunks_p2} chunks @ 20s)")
        pass1_48k     = resample_np(pass1_16k, 16000, 48000)
        del pass1_16k  # free RAM before allocating 48kHz buffer
        _empty_cache()
        pass2_48k     = process_in_chunks(
            audio_np=pass1_48k,
            sr=48000,
            fn=lambda chunk: models.se_48k(input_path=chunk, online_write=False),
            chunk_sec=20.0,
            progress_cb=lambda i, n, s, e: status_cb(
                f"Tier 3: pass 2 — chunk {i}/{n}  ({s:.0f}s–{e:.0f}s)"
            ),
        )
        del pass1_48k
        _empty_cache()

        pass2_16k     = resample_np(pass2_48k, 48000, 16000)
        del pass2_48k
        hifi_48k      = _sr_chunked(models, pass2_16k, status_cb=status_cb)
        del pass2_16k
        hifi_16k_mg   = resample_np(hifi_48k, 48000, 16000)
        status_cb("Tier 3: MetricGAN+ polishing (chunked 30s)")
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
#  Internal helpers
# --------------------------------------------------------------------------- #

def _sr_chunked(
    models,
    audio_16k: np.ndarray,
    chunk_sec: float = 60.0,
    status_cb=None,
) -> np.ndarray:
    """
    Run MossFormer2_SR_48K in chunks to avoid CUDA OOM on long files.
    Input: 16 kHz mono array.  Output: 48 kHz mono array (3× length).
    Falls back to torchaudio resample if SR model OOMs (low-VRAM machines).
    """
    import torch
    from enhancement_router import _extract_np, _to_2d, resample_np

    chunk_in = int(chunk_sec * 16000)
    n        = len(audio_16k)
    n_chunks = max(1, int(np.ceil(n / chunk_in)))

    if status_cb:
        status_cb(f"Tier 3: SR_48K — ~{n_chunks} chunks")

    def _run_chunk(chunk):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            out = models.sr_48k(input_path=_to_2d(chunk), online_write=False)
            return _extract_np(out)
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "torchscript" in msg
                    or "cuda" in msg or "allocate" in msg):
                logger.warning(
                    "SR_48K CUDA OOM — falling back to torchaudio resample."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return resample_np(chunk, 16000, 48000)
            raise

    if n <= chunk_in:
        return _run_chunk(audio_16k)

    out_chunks = []
    idx   = 0
    start = 0
    while start < n:
        end = min(start + chunk_in, n)
        idx += 1
        if status_cb:
            status_cb(f"Tier 3: SR_48K — chunk {idx}/{n_chunks}  ({start//16000:.0f}s–{end//16000:.0f}s)")
        out_chunks.append(_run_chunk(audio_16k[start:end]))
        start = end

    return np.concatenate(out_chunks, axis=0)


# --------------------------------------------------------------------------- #
#  Internal helper: score a numpy array without writing a temp file
# --------------------------------------------------------------------------- #

def _score_np(audio_np: np.ndarray, sr: int) -> float:
    """Quick DNSMOS mos_ovr from a numpy array. Clips to 60s to stay fast."""
    try:
        import torch
        import torchaudio.functional as F_ta
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

        data = audio_np.astype(np.float32)
        if sr != 16000:
            t    = torch.from_numpy(data).unsqueeze(0)
            t    = F_ta.resample(t, sr, 16000)
            data = t.squeeze(0).numpy()

        # Clip to 60s max
        data   = data[: 16000 * 60]
        tensor = torch.from_numpy(data).unsqueeze(0)
        dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
        dnsmos.update(tensor)
        raw = dnsmos.compute()

        if isinstance(raw, dict):
            return float(raw.get("mos_ovr", raw.get("ovr", 0)))
        s = raw.flatten()
        return float(s[3]) if len(s) >= 4 else float(s[0])
    except Exception:
        return 0.0
