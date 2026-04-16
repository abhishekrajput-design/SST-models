"""
Tier 4 — Worst Audio (mos_ovr < 1.5)

"Nuclear" pipeline — generative reconstruction for severely degraded audio.

Pipeline:
  load 16kHz
  → FFmpeg normalize only (preserve signal, don't strip what little exists)
  → MossFormer2_SS_16K  (speech separation → stream1, stream2)
  → MossFormerGAN_SE_16K on each stream (individual denoising)
  → Resemble Enhance (denoiser + CFM generative reconstruction) on each stream
  → MossFormer2_SR_48K on each stream (super-resolution 16k→48k)
  → MetricGAN+ on each stream
  → Mix (sum + normalize) streams back → primary output
  → Re-score DNSMOS; flag needs_human_review if still < 2.0

Side effect:
  Writes two separated-stream WAVs alongside the main output for downstream
  diarization use (each stream is one speaker, much cleaner for pyannote).

Install:  pip install resemble-enhance clearvoice
"""
from __future__ import annotations

import gc
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Resemble Enhance CFM number of function evaluations
# 64 = production quality, 128 = maximum (slower)
_NFE = 64


def process(input_path: str, output_path: str, models, status_cb) -> dict:
    import torch
    from enhancement_router import (
        load_16k_mono, save_wav, resample_np, _extract_np, _to_2d,
        apply_metricgan, process_in_chunks,
    )

    status_cb("Tier 4: loading audio")
    audio_16k, _ = load_16k_mono(input_path)

    # ── Speech separation: mixed → 2 streams ─────────────────────────────────
    status_cb("Tier 4: MossFormer2_SS_16K — separating speakers")
    out_ss = models.ss_16k(input_path=_to_2d(audio_16k), online_write=False)
    if isinstance(out_ss, (list, tuple)) and len(out_ss) >= 2:
        stream1 = _extract_np(out_ss[0])
        stream2 = _extract_np(out_ss[1])
    else:
        arr = _extract_np(out_ss)
        if arr.ndim == 2 and arr.shape[0] == 2:
            stream1, stream2 = arr[0], arr[1]
        else:
            logger.warning("SS_16K returned unexpected shape; treating as single stream.")
            stream1 = arr
            stream2 = arr.copy()

    # ── Per-stream processing ─────────────────────────────────────────────────
    processed_streams = []
    stream_paths: list[str] = []
    base, ext = os.path.splitext(output_path)

    for idx, raw_stream in enumerate([stream1, stream2]):
        lbl = f"stream{idx + 1}"

        # GAN denoise (chunked — each stream can be up to 30 min)
        status_cb(f"Tier 4: MossFormerGAN_SE_16K — denoising {lbl} (chunked)")
        denoised = process_in_chunks(
            audio_np=raw_stream,
            sr=16000,
            fn=lambda chunk: models.se_16k(input_path=chunk, online_write=False),
        )

        # Resemble Enhance — generative reconstruction
        status_cb(f"Tier 4: Resemble Enhance — reconstructing {lbl}")
        denoised = _resemble_enhance_np(denoised, sr=16000)

        # Super-resolution (chunked — 60s @ 16kHz per chunk)
        import torch
        status_cb(f"Tier 4: MossFormer2_SR_48K — upsampling {lbl} (chunked)")
        chunk_in = int(60.0 * 16000)
        n_d = len(denoised)

        def _run_sr(chunk):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                out = models.sr_48k(input_path=_to_2d(chunk), online_write=False)
                return _extract_np(out)
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "torchscript" in msg or "cuda" in msg or "allocate" in msg:
                    logger.warning(f"SR_48K OOM ({lbl}) — falling back to torchaudio resample.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return resample_np(chunk, 16000, 48000)
                raise

        if n_d <= chunk_in:
            hifi_48k = _run_sr(denoised)
        else:
            sr_chunks = []
            s = 0
            while s < n_d:
                e = min(s + chunk_in, n_d)
                sr_chunks.append(_run_sr(denoised[s:e]))
                s = e
            hifi_48k = np.concatenate(sr_chunks, axis=0)

        # MetricGAN+ polish
        status_cb(f"Tier 4: MetricGAN+ — polishing {lbl}")
        hifi_16k = resample_np(hifi_48k, 48000, 16000)
        polished = apply_metricgan(hifi_16k, sr=16000)
        polished = resample_np(polished, 16000, 48000)

        # Save individual stream (for diarization)
        stream_path = f"{base}_sep{idx + 1}{ext}"
        save_wav(polished, 48000, stream_path)
        stream_paths.append(stream_path)
        processed_streams.append(polished)

    # ── Mix streams back together ─────────────────────────────────────────────
    status_cb("Tier 4: mixing streams → final output")
    min_len = min(len(s) for s in processed_streams)
    mixed   = sum(s[:min_len] for s in processed_streams)
    peak    = np.abs(mixed).max()
    if peak > 1e-6:
        mixed = mixed / peak * 0.9  # normalize to 90% FS

    save_wav(mixed, 48000, output_path)

    # ── Post-process quality check ─────────────────────────────────────────────
    status_cb("Tier 4: re-scoring post-enhancement quality")
    post_mos = _score_np_quick(mixed, 48000)
    needs_human_review = post_mos < 2.0
    status_cb(f"Tier 4: post-MOS = {post_mos:.3f}  {'⚠ needs human review' if needs_human_review else '✓ OK'}")
    if needs_human_review:
        logger.warning(
            f"Tier 4: post-enhancement mos_ovr={post_mos:.3f} still < 2.0 → "
            "flagging for human review."
        )

    logger.info(f"Tier 4 done → {output_path}  (post_mos={post_mos:.3f})")
    return {
        "pipeline_used":       "tier4_worst_generative",
        "post_enhancement_mos": round(post_mos, 3),
        "separated_streams":   stream_paths,
        "needs_human_review":  needs_human_review,
        "review_reason":       "post-enhancement MOS < 2.0" if needs_human_review else "",
    }


# --------------------------------------------------------------------------- #
#  Resemble Enhance wrapper
# --------------------------------------------------------------------------- #

def _resemble_enhance_np(audio_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Run Resemble Enhance (denoiser + CFM) on a numpy array.
    Returns numpy array at same sample rate.
    Falls back to input if not installed or fails.
    """
    try:
        import torch
        from resemble_enhance.enhancer.inference import enhance

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)  # [1, T]
        enhanced, out_sr = enhance(t, sr=sr, device=device, nfe=_NFE)
        result = enhanced.squeeze(0).cpu().numpy()
        # Resample back to input sr if resemble_enhance changed it
        if out_sr != sr:
            from enhancement_router import resample_np
            result = resample_np(result, out_sr, sr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except ImportError:
        logger.warning("resemble-enhance not installed — skipping generative reconstruction.")
        return audio_np
    except Exception as exc:
        logger.warning(f"Resemble Enhance failed: {exc} — skipping.")
        return audio_np


def _score_np_quick(audio_np: np.ndarray, sr: int) -> float:
    """Quick DNSMOS mos_ovr from a numpy array. Returns 0.0 on failure."""
    try:
        import torch
        import torchaudio.functional as F_ta
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

        data = audio_np.astype(np.float32)
        if sr != 16000:
            t    = torch.from_numpy(data).unsqueeze(0)
            t    = F_ta.resample(t, sr, 16000)
            data = t.squeeze(0).numpy()

        # Clip to 60s max — DNSMOS is not designed for long tensors
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
