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
    Lazy-loading ClearVoice model manager.

    Models are loaded into VRAM **on first access** (not at startup) and can
    be unloaded between pipeline stages so that only the model currently
    needed occupies GPU memory.  This prevents OOM when all 4 models would
    exceed available VRAM.

    Usage::

        models = ClearVoiceModels.get()
        models.ensure_only("se_16k")     # load SE_16K, unload everything else
        result = models.se_16k(input_path=chunk, online_write=False)
        models.ensure_only("sr_48k")     # swap to SR model
        result = models.sr_48k(input_path=chunk, online_write=False)
    """

    _ALL = ("se_16k", "se_48k", "ss_16k", "sr_48k")

    def __init__(self):
        from clearvoice import ClearVoice          # fail-fast if missing
        self._ClearVoice = ClearVoice
        self._se_16k = None
        self._se_48k = None
        self._ss_16k = None
        self._sr_48k = None
        logger.info("ClearVoice available — models will load on demand.")

    # ── lazy-loading properties ───────────────────────────────────────────────

    @property
    def se_16k(self):
        if self._se_16k is None:
            logger.info("  Loading MossFormerGAN_SE_16K …")
            self._se_16k = self._ClearVoice(
                task="speech_enhancement",
                model_names=["MossFormerGAN_SE_16K"],
            )
            logger.info("  ✓ MossFormerGAN_SE_16K ready")
        return self._se_16k

    @property
    def se_48k(self):
        if self._se_48k is None:
            logger.info("  Loading MossFormer2_SE_48K …")
            self._se_48k = self._ClearVoice(
                task="speech_enhancement",
                model_names=["MossFormer2_SE_48K"],
            )
            logger.info("  ✓ MossFormer2_SE_48K ready")
        return self._se_48k

    @property
    def ss_16k(self):
        if self._ss_16k is None:
            logger.info("  Loading MossFormer2_SS_16K …")
            self._ss_16k = self._ClearVoice(
                task="speech_separation",
                model_names=["MossFormer2_SS_16K"],
            )
            logger.info("  ✓ MossFormer2_SS_16K ready")
        return self._ss_16k

    @property
    def sr_48k(self):
        if self._sr_48k is None:
            logger.info("  Loading MossFormer2_SR_48K …")
            self._sr_48k = self._ClearVoice(
                task="speech_super_resolution",
                model_names=["MossFormer2_SR_48K"],
            )
            logger.info("  ✓ MossFormer2_SR_48K ready")
        return self._sr_48k

    # ── VRAM management ───────────────────────────────────────────────────────

    def unload(self, *names: str):
        """Unload specific models to free VRAM."""
        import torch
        for name in names:
            attr = f"_{name}"
            if getattr(self, attr, None) is not None:
                logger.info(f"  Unloading {name} to free VRAM")
                setattr(self, attr, None)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # empty_cache can raise after OOM — safe to ignore

    def ensure_only(self, *keep: str):
        """Unload every model except those listed in *keep*."""
        to_drop = [m for m in self._ALL if m not in keep]
        if to_drop:
            self.unload(*to_drop)

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
        import shutil
        shutil.copy2(input_path, output_path)
        return {"pipeline_used": "passthrough_no_clearvoice", "separated_streams": []}

    # ── VAD-based speech extraction (Silero VAD) ──────────────────────────────
    _status("Detecting speech regions (Silero VAD)…")
    audio_16k, _ = load_16k_mono(input_path)
    regions = get_speech_regions(audio_16k, 16000)

    speech_dur = sum(e - s for s, e in regions) / 16000
    total_dur  = len(audio_16k) / 16000
    ratio      = speech_dur / max(total_dur, 1)
    _status(f"Speech: {speech_dur:.0f}s / {total_dur:.0f}s ({ratio:.0%})")

    use_vad = ratio < 0.80  # only bother if >20% is silence

    if not use_vad:
        return process(input_path, output_path, models, _status)

    # Save speech-only to temp file for tier processing
    speech_audio = concatenate_regions(audio_16k, regions)
    _status(
        f"Processing {speech_dur:.0f}s of speech "
        f"(skipping {total_dur - speech_dur:.0f}s silence)"
    )

    tmp_input = input_path + ".vad_speech.wav"
    save_wav(speech_audio, 16000, tmp_input)

    try:
        result = process(tmp_input, output_path, models, _status)

        # Load enhanced speech (48 kHz) written by the tier pipeline
        enhanced_48k, _ = sf.read(output_path, dtype="float32")
        if enhanced_48k.ndim > 1:
            enhanced_48k = enhanced_48k.mean(axis=1)

        # Reconstruct full-length output preserving original timeline
        _status("Reconstructing full-length enhanced audio")
        full_output = reconstruct_enhanced(audio_16k, regions, enhanced_48k)
        save_wav(full_output, 48000, output_path)

        # Tier 4 also writes separated streams — reconstruct those too
        for sp in result.get("separated_streams", []):
            if os.path.isfile(sp):
                sp_data, _ = sf.read(sp, dtype="float32")
                if sp_data.ndim > 1:
                    sp_data = sp_data.mean(axis=1)
                sp_full = reconstruct_enhanced(audio_16k, regions, sp_data)
                save_wav(sp_full, 48000, sp)

        return result
    finally:
        try:
            if os.path.isfile(tmp_input):
                os.unlink(tmp_input)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
#  Silero VAD — speech region detection
# --------------------------------------------------------------------------- #

_vad_model = None
_vad_get_ts = None


def _load_vad():
    """Load Silero VAD once and cache globally."""
    global _vad_model, _vad_get_ts
    if _vad_model is None:
        import torch
        _vad_model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )
        _vad_get_ts = utils[0]  # get_speech_timestamps
    return _vad_model, _vad_get_ts


def get_speech_regions(
    audio_np: np.ndarray,
    sr: int,
    min_speech_ms: int = 250,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 200,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Detect speech regions using Silero VAD.

    Returns a list of ``(start_sample, end_sample)`` tuples.
    Falls back to returning the full audio as one region if VAD fails.
    """
    try:
        import torch
        model, get_ts = _load_vad()

        wav = torch.from_numpy(audio_np.astype(np.float32))

        timestamps = get_ts(
            wav,
            model,
            sampling_rate=sr,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            threshold=threshold,
        )

        if not timestamps:
            logger.warning("Silero VAD: no speech detected — using full audio")
            return [(0, len(audio_np))]

        regions = [
            (max(0, ts["start"]), min(len(audio_np), ts["end"]))
            for ts in timestamps
        ]
        return regions

    except Exception as exc:
        logger.warning(f"Silero VAD unavailable ({exc}) — using full audio")
        return [(0, len(audio_np))]


def concatenate_regions(
    audio_np: np.ndarray, regions: list[tuple[int, int]],
) -> np.ndarray:
    """Extract and concatenate speech regions into a single array."""
    return np.concatenate([audio_np[s:e] for s, e in regions])


def reconstruct_enhanced(
    original_16k: np.ndarray,
    regions_16k: list[tuple[int, int]],
    enhanced_48k: np.ndarray,
) -> np.ndarray:
    """
    Place enhanced speech (48 kHz) back into the original timeline.

    Non-speech regions keep the original audio upsampled to 48 kHz so
    timestamps are preserved for downstream transcription / A-B playback.
    """
    SR_RATIO = 3  # 48000 / 16000

    # Upsample original to 48 kHz as the background canvas
    output = resample_np(original_16k, 16000, 48000)

    offset = 0
    for start_16k, end_16k in regions_16k:
        seg_48k = (end_16k - start_16k) * SR_RATIO
        s48     = start_16k * SR_RATIO
        e48     = min(s48 + seg_48k, len(output))
        avail   = min(e48 - s48, len(enhanced_48k) - offset)
        if avail > 0:
            output[s48 : s48 + avail] = enhanced_48k[offset : offset + avail]
        offset += seg_48k

    return output


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


def remove_silence(
    audio_np: np.ndarray,
    sr: int,
    top_db: float = 40.0,
    min_silence_ms: float = 800.0,
    keep_padding_ms: float = 200.0,
) -> tuple[np.ndarray, float]:
    """
    Remove silence regions longer than min_silence_ms from audio.

    Args:
        audio_np:        1-D float32 numpy array.
        sr:              Sample rate of audio_np.
        top_db:          Frames quieter than (peak_rms_db - top_db) are silent.
        min_silence_ms:  Only remove silence runs longer than this (ms).
        keep_padding_ms: Keep this many ms of audio on each side of speech
                         regions so onsets/offsets aren't cut off.

    Returns:
        (trimmed_audio, original_duration_s)
        If trimming would leave < 1 s of audio the original is returned.
    """
    n = len(audio_np)
    orig_dur = n / sr

    hop       = max(1, int(0.0125 * sr))   # 12.5 ms per frame
    frame_len = max(1, int(0.025  * sr))   # 25 ms analysis window

    n_frames = max(1, (n - frame_len) // hop + 1)

    # ── RMS in dB per frame (vectorised) ──────────────────────────────────────
    # Build a 2-D view: shape (n_frames, frame_len) via strided indexing
    indices = (np.arange(n_frames)[:, None] * hop +
               np.arange(frame_len)[None, :])
    indices = np.clip(indices, 0, n - 1)          # clamp last partial frame
    frames  = audio_np[indices]                    # (n_frames, frame_len)
    rms     = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db  = (20.0 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32)

    peak = float(rms_db.max())
    if peak < -70.0:
        return audio_np, orig_dur               # near-silent — nothing to trim

    # ── Voice-activity mask ────────────────────────────────────────────────────
    threshold = peak - top_db
    voiced    = (rms_db > threshold).astype(np.float32)

    if voiced.sum() == 0:
        return audio_np, orig_dur

    # ── Expand voiced regions by keep_padding_ms (dilation via convolution) ───
    pad_f  = max(1, int(round(keep_padding_ms / 1000.0 * sr / hop)))
    kernel = np.ones(2 * pad_f + 1, dtype=np.float32)
    voiced = (np.convolve(voiced, kernel, mode="same") > 0)

    # ── Fill silence gaps shorter than min_silence_ms ─────────────────────────
    sil_f = max(1, int(round(min_silence_ms / 1000.0 * sr / hop)))
    i = 0
    while i < n_frames:
        if not voiced[i]:
            j = i + 1
            while j < n_frames and not voiced[j]:
                j += 1
            if (j - i) < sil_f:           # short gap → keep it
                voiced[i:j] = True
            i = j
        else:
            i += 1

    # ── Upsample frame mask → per-sample mask ─────────────────────────────────
    sample_mask = np.repeat(voiced, hop)
    if len(sample_mask) < n:
        # Last partial frame: keep (don't cut the tail)
        sample_mask = np.concatenate(
            [sample_mask, np.ones(n - len(sample_mask), dtype=bool)]
        )
    sample_mask = sample_mask[:n]

    trimmed = audio_np[sample_mask]

    # Safety: never return less than 1 second of audio
    if len(trimmed) < sr:
        return audio_np, orig_dur

    return trimmed, orig_dur


def save_wav(data: np.ndarray, sr: int, path: str):
    """Save numpy array as WAV. Clips to [-1, 1] to prevent clipping artifacts."""
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, data, sr, subtype="PCM_16")


def apply_se_16k(models: ClearVoiceModels, audio_np: np.ndarray) -> np.ndarray:
    """Run MossFormerGAN_SE_16K on 16kHz numpy array → returns 16kHz numpy array."""
    out = models.se_16k(input_path=_to_2d(audio_np), online_write=False)
    return _extract_np(out)


def apply_se_48k(models: ClearVoiceModels, audio_np: np.ndarray) -> np.ndarray:
    """Run MossFormer2_SE_48K. Input 48kHz → output 48kHz."""
    out = models.se_48k(input_path=_to_2d(audio_np), online_write=False)
    return _extract_np(out)


def apply_ss_16k(models: ClearVoiceModels, audio_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run MossFormer2_SS_16K (speech separation). Returns (stream1, stream2) at 16kHz."""
    out = models.ss_16k(input_path=_to_2d(audio_np), online_write=False)
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
    out = models.sr_48k(input_path=_to_2d(audio_np), online_write=False)
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


def apply_metricgan(
    audio_np: np.ndarray,
    sr: int,
    models_dir: str = "models/metricgan",
    chunk_sec: float = 30.0,
) -> np.ndarray:
    """
    Run SpeechBrain MetricGAN+ on 16kHz mono numpy array → 16kHz output.

    Automatically chunks long audio (> chunk_sec) to avoid the BLSTM in
    MetricGAN+ hanging or OOMing on multi-minute inputs.  The model is loaded
    once and reused across chunks; boundaries are smoothed with a 2-second
    linear crossfade (overlap-add).
    """
    try:
        import torch
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        from speechbrain.utils.fetching import LocalStrategy

        # Resample to 16kHz if needed (MetricGAN+ expects 16kHz)
        audio_16k = resample_np(audio_np, sr, 16000) if sr != 16000 else audio_np
        n         = len(audio_16k)
        chunk_len = int(chunk_sec * 16000)

        # Load the model once
        enhancer = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir=models_dir,
            local_strategy=LocalStrategy.COPY,
        )

        def _run_chunk(chunk_1d: np.ndarray) -> np.ndarray:
            noisy   = torch.from_numpy(chunk_1d.astype(np.float32)).unsqueeze(0)
            lengths = torch.tensor([1.0])
            with torch.no_grad():
                enhanced = enhancer.enhance_batch(noisy, lengths)
            return enhanced.squeeze(0).cpu().numpy()

        if n <= chunk_len:
            out = _run_chunk(audio_16k)
        else:
            # ── Overlap-add chunked processing ─────────────────────────────────
            overlap = int(2.0 * 16000)   # 2-second crossfade at each boundary
            step    = chunk_len - overlap

            starts = list(range(0, n - overlap, step))
            if starts and starts[-1] + chunk_len < n:
                starts.append(n - chunk_len)

            n_chunks = len(starts)
            output   = np.zeros(n, dtype=np.float32)
            weight   = np.zeros(n, dtype=np.float32)

            for i, start in enumerate(starts):
                end     = min(start + chunk_len, n)
                seg_len = end - start
                chunk   = audio_16k[start:end]

                logger.debug(
                    f"MetricGAN+ chunk {i + 1}/{n_chunks} "
                    f"[{start / 16000:.1f}s – {end / 16000:.1f}s]"
                )
                processed = _run_chunk(chunk)
                if len(processed) > seg_len:
                    processed = processed[:seg_len]
                elif len(processed) < seg_len:
                    processed = np.pad(processed, (0, seg_len - len(processed)))

                # Linear crossfade window
                win = np.ones(seg_len, dtype=np.float32)
                if i > 0 and overlap > 0:
                    fade = min(overlap, seg_len)
                    win[:fade] = np.linspace(0.0, 1.0, fade)
                if i < n_chunks - 1 and overlap > 0:
                    fade = min(overlap, seg_len)
                    win[seg_len - fade:] = np.linspace(1.0, 0.0, fade)

                output[start:end] += processed * win
                weight[start:end] += win

            mask = weight > 1e-8
            output[mask] /= weight[mask]
            out = output

        del enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    except Exception as exc:
        logger.warning(f"MetricGAN+ failed: {exc} — returning input unchanged.")
        return audio_np


# --------------------------------------------------------------------------- #
#  Chunked processing helper
# --------------------------------------------------------------------------- #

def process_in_chunks(
    audio_np: np.ndarray,
    sr: int,
    fn,
    chunk_sec: float = 60.0,
    overlap_sec: float = 2.0,
    progress_cb=None,
) -> np.ndarray:
    """
    Split a long 1-D audio array into overlapping chunks, process each with
    ``fn``, then stitch back together using linear crossfade (overlap-add).

    Args:
        audio_np:    1-D float32 numpy array at sample rate ``sr``.
        sr:          Sample rate of ``audio_np``.
        fn:          Callable that accepts a 2-D ``(1, T)`` chunk (float32
                     numpy array) and returns a 1-D or 2-D numpy array of
                     the same length.
        chunk_sec:   Duration of each processing chunk in seconds.
        overlap_sec: Half-overlap (crossfade region) at each boundary.

    Returns:
        1-D float32 numpy array with the same total length as ``audio_np``.

    Notes:
        * Short audio (< chunk_sec) is passed through in one shot — no
          chunking or crossfade overhead.
        * Only SE models should be chunked.  SR and SS models are fast and
          stateful across frames; do not chunk them.
    """
    audio_np = np.asarray(audio_np, dtype=np.float32)
    n_samples = len(audio_np)
    chunk_len   = int(chunk_sec   * sr)
    overlap_len = int(overlap_sec * sr)
    step        = chunk_len - overlap_len  # non-overlapping stride

    # Short audio — process as a single chunk
    if n_samples <= chunk_len:
        logger.debug("process_in_chunks: audio shorter than chunk_sec — single pass")
        return _extract_np(fn(_to_2d(audio_np)))

    # Build chunk start positions
    starts = list(range(0, n_samples - overlap_len, step))

    # Ensure the final chunk always reaches the end of the signal
    if starts[-1] + chunk_len < n_samples:
        starts.append(n_samples - chunk_len)

    n_chunks = len(starts)
    output   = np.zeros(n_samples, dtype=np.float32)
    weight   = np.zeros(n_samples, dtype=np.float32)

    for i, start in enumerate(starts):
        end   = min(start + chunk_len, n_samples)
        chunk = audio_np[start:end]

        logger.debug(
            f"process_in_chunks: chunk {i + 1}/{n_chunks} "
            f"[{start / sr:.1f}s – {end / sr:.1f}s]"
        )
        if progress_cb:
            progress_cb(i + 1, n_chunks, start / sr, end / sr)

        processed = _extract_np(fn(_to_2d(chunk)))

        # Guard: processed length might differ by ±1 sample due to model internals
        proc_len = len(processed)
        seg_len  = end - start
        if proc_len != seg_len:
            if proc_len > seg_len:
                processed = processed[:seg_len]
            else:
                processed = np.pad(processed, (0, seg_len - proc_len))

        # Build a window for overlap-add crossfade
        #   - First chunk:  no fade-in at the leading edge
        #   - Last chunk:   no fade-out at the trailing edge
        #   - All chunks:   linear fade-in for the first overlap_len samples
        #                   and linear fade-out for the last overlap_len samples
        #                   (except at the very start/end of the file)
        win = np.ones(seg_len, dtype=np.float32)

        # Fade-in at the left boundary (skip for the very first chunk)
        if i > 0 and overlap_len > 0:
            fade_in_len = min(overlap_len, seg_len)
            win[:fade_in_len] = np.linspace(0.0, 1.0, fade_in_len)

        # Fade-out at the right boundary (skip for the very last chunk)
        if i < n_chunks - 1 and overlap_len > 0:
            fade_out_len = min(overlap_len, seg_len)
            win[seg_len - fade_out_len:] = np.linspace(1.0, 0.0, fade_out_len)

        output[start:end] += processed * win
        weight[start:end] += win

    # Normalise overlapping regions so they sum to 1 (avoid 2× amplitude in overlaps)
    mask = weight > 1e-8
    output[mask] /= weight[mask]

    return output.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Internal
# --------------------------------------------------------------------------- #

def _to_2d(audio_np: np.ndarray) -> np.ndarray:
    """ClearVoice expects (batch, time) shape. Wrap 1-D array → (1, T)."""
    arr = np.asarray(audio_np, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]   # (T,) → (1, T)
    return arr


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
