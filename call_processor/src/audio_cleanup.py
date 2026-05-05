import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf


logger = logging.getLogger("audio_cleanup_claude")

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------
TARGET_SR = 16000

# --- Noise reduction (two-pass spectral gating) ---
SPECTRAL_GATE_STRENGTH_STATIONARY = 0.98  # near-max stationary suppression
SPECTRAL_GATE_STRENGTH_DYNAMIC = 0.72     # aggressive dynamic pass — key for transient bg voices
SPECTRAL_GATE_N_FFT = 1024
SPECTRAL_GATE_WIN_LENGTH = 1024
SPECTRAL_GATE_HOP_LENGTH = 256
SPECTRAL_GATE_TIME_SMOOTH_MS = 36
SPECTRAL_GATE_FREQ_SMOOTH_HZ = 200        # tighter freq resolution for precise removal

# --- Band-pass filter for near-field speech isolation ---
BANDPASS_LOW_HZ = 85        # cut rumble / HVAC — agent fundamentals start ~85 Hz
BANDPASS_HIGH_HZ = 6500     # tighter HF cut — removes more bleed from neighbouring desks

# --- Voice Activity Detection ---
VAD_TOP_DB = 30             # more sensitive — capture all agent speech incl. softer parts
VAD_FRAME_LENGTH = 2048
VAD_HOP_LENGTH = 512
MIN_SPEECH_MS = 180         # catch shorter agent utterances ("yes", "ok", "hmm")
SPEECH_PAD_START_MS = 150   # wider pad to preserve word onsets
SPEECH_PAD_END_MS = 220     # wider pad to preserve word endings / trailing consonants
MIN_SILENCE_TO_REMOVE_MS = 1500  # only remove gaps >= this

# --- Near-field energy gate (~50 cm isolation) ---
# Segments whose RMS (dBFS) is below this threshold relative to the
# loudest segment are treated as far-field / background and dropped.
# At 50 cm the agent varies ~8 dB; background at 2 m is 12+ dB below
# the quietest agent speech.  11 dB keeps all agent speech with 3 dB margin.
NEAR_FIELD_RMS_DROP_DB = 11.0

# --- Spectral-flatness clarity gate ---
# Wiener entropy close to 1.0 means noise-like; close to 0.0 means tonal/voiced.
# Segments above this threshold are considered unclear / non-speech.
CLARITY_FLATNESS_THRESHOLD = 0.68  # stricter — only keep clearly voiced agent speech (was 0.85)

# --- Speech enhancement (background) ---
BACKGROUND_ATTENUATION_DB = -48.0    # near-silence non-speech regions
TARGET_PEAK_DBFS = -1.0

# --- Per-segment normalization ---
TARGET_SEGMENT_RMS_DBFS = -18.0      # each segment individually normalized to this
MAX_SEGMENT_BOOST_DB = 20.0          # max gain allowed for any single segment
MIN_SEGMENT_BOOST_DB = -6.0          # allow modest attenuation of loud segments
GAIN_RAMP_MS = 5.0                   # smooth transition at segment edges (bg↔speech)

# --- Intra-segment dynamic range compression ---
COMPRESSOR_THRESHOLD_DBFS = -22.0    # compress above this level (post per-seg norm)
COMPRESSOR_RATIO = 3.0               # 3:1 soft compression
COMPRESSOR_ATTACK_MS = 15.0          # fast attack to catch sudden loudness
COMPRESSOR_RELEASE_MS = 150.0        # slower release for natural decay
COMPRESSOR_WINDOW_MS = 30.0          # sliding RMS window for level estimation

# --- Crossfade ---
CROSSFADE_MS = 8.0                   # crossfade between concatenated segments


# =========================================================================
# Public API — same interface as deskStreamer_trae
# =========================================================================

def clean_audio(input_path: str, output_path: str) -> str:
    """Clean *input_path* and write the result to *output_path*.

    Returns the path to the saved cleaned file.  A companion
    ``<output_path>.segment_map_claude.json`` is written alongside it.
    """
    # 1. Load
    audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
    if audio.size == 0:
        raise ValueError(f"Unable to load audio or empty audio: {input_path}")
    logger.info("loaded %s  samples=%d  sr=%d  duration=%.2fs",
                input_path, len(audio), sr, len(audio) / sr)

    # 2. Two-pass noise reduction
    denoised = _two_pass_noise_reduction(audio, sr)

    # 3. Band-pass filter — isolate speech band, reject rumble + hiss
    denoised = _bandpass_filter(denoised, sr, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)

    # 4. Detect speech segments via VAD
    segments = _detect_speech_segments(denoised, sr)

    if not segments:
        logger.warning("no speech segments detected — returning normalised full audio")
        segment_map = _trivial_map(len(audio), sr)
        cleaned_audio = _apply_master_normalization(denoised)
    else:
        # 5. Energy gate — reject far-field (quiet) segments
        segments = _energy_gate(denoised, segments)

        # 6. Clarity gate — reject noisy / unintelligible segments
        segments = _clarity_gate(denoised, sr, segments)

        if not segments:
            logger.warning("all segments filtered by energy/clarity gates — returning normalised full audio")
            segment_map = _trivial_map(len(audio), sr)
            cleaned_audio = _apply_master_normalization(denoised)
        else:
            # 7. Per-segment gain normalization + background attenuation
            enhanced, per_seg_gains = _per_segment_normalize(denoised, segments, sr)

            # 8. Intra-segment dynamic range compression
            enhanced = _apply_intra_segment_compression(enhanced, segments, sr)

            # 9. Concatenate kept segments with crossfade
            cleaned_audio, clean_boundaries = _concat_segments_crossfade(enhanced, segments, sr)

            # 10. Master normalization
            cleaned_audio = _apply_master_normalization(cleaned_audio)

            # Build segment map
            segment_map = _build_segment_map(
                clean_boundaries, len(audio), len(cleaned_audio), sr, per_seg_gains,
            )

    # 10. Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cleaned_path = _save_audio(cleaned_audio, sr, output_path)

    map_path = get_segment_map_path(cleaned_path)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(segment_map, f)

    logger.info("cleaned audio saved: %s", cleaned_path)
    logger.info("segment map saved: %s  segments=%d", map_path, len(segment_map["segments"]))
    return cleaned_path


def get_segment_map_path(cleaned_audio_path: str) -> str:
    return f"{cleaned_audio_path}.segment_map_claude.json"


def load_segment_map(cleaned_audio_path: str) -> Dict:
    map_path = get_segment_map_path(cleaned_audio_path)
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Internal helpers
# =========================================================================

def _two_pass_noise_reduction(audio: np.ndarray, sr: int) -> np.ndarray:
    """Stationary pass (broadband hum) then dynamic pass (transient noise)."""
    common = dict(
        n_fft=SPECTRAL_GATE_N_FFT,
        win_length=SPECTRAL_GATE_WIN_LENGTH,
        hop_length=SPECTRAL_GATE_HOP_LENGTH,
        time_mask_smooth_ms=SPECTRAL_GATE_TIME_SMOOTH_MS,
        freq_mask_smooth_hz=SPECTRAL_GATE_FREQ_SMOOTH_HZ,
    )
    out = nr.reduce_noise(
        y=audio, sr=sr, stationary=True,
        prop_decrease=SPECTRAL_GATE_STRENGTH_STATIONARY, **common,
    ).astype(np.float32)

    out = nr.reduce_noise(
        y=out, sr=sr, stationary=False,
        prop_decrease=SPECTRAL_GATE_STRENGTH_DYNAMIC, **common,
    ).astype(np.float32)
    return out


def _bandpass_filter(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a Butterworth band-pass filter to isolate the speech band."""
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    low = max(low_hz / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    sos = butter(5, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _detect_speech_segments(audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    """Return list of (start_sample, end_sample) speech regions."""
    raw = librosa.effects.split(
        audio, top_db=VAD_TOP_DB,
        frame_length=VAD_FRAME_LENGTH, hop_length=VAD_HOP_LENGTH,
    )
    if len(raw) == 0:
        return []

    pad_start = int(round(SPEECH_PAD_START_MS * sr / 1000.0))
    pad_end = int(round(SPEECH_PAD_END_MS * sr / 1000.0))
    min_gap = int(round(MIN_SILENCE_TO_REMOVE_MS * sr / 1000.0))
    min_speech = int(round(MIN_SPEECH_MS * sr / 1000.0))
    total = len(audio)

    padded: List[Tuple[int, int]] = []
    for s, e in raw:
        s, e = int(s), int(e)
        if e - s < min_speech:
            continue
        padded.append((max(0, s - pad_start), min(total, e + pad_end)))

    if not padded:
        return []

    # Merge overlapping / close segments
    merged: List[List[int]] = [[padded[0][0], padded[0][1]]]
    for s, e in padded[1:]:
        prev = merged[-1]
        if s - prev[1] < min_gap:
            prev[1] = max(prev[1], e)
        else:
            merged.append([s, e])

    return [(int(s), int(e)) for s, e in merged if e > s]


def _energy_gate(audio: np.ndarray, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Drop segments whose RMS is too far below the loudest segment.

    This rejects far-field / background voices that are significantly
    quieter than the primary near-field speaker (~3 feet).
    """
    if not segments:
        return segments

    rms_values = []
    for s, e in segments:
        chunk = audio[s:e]
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        rms_values.append(rms)

    peak_rms = max(rms_values)
    if peak_rms <= 0:
        return segments

    peak_db = 20.0 * np.log10(peak_rms)
    kept = []
    for (s, e), rms in zip(segments, rms_values):
        if rms <= 0:
            continue
        seg_db = 20.0 * np.log10(rms)
        if peak_db - seg_db <= NEAR_FIELD_RMS_DROP_DB:
            kept.append((s, e))
        else:
            logger.debug("energy-gate dropped segment %.2fs–%.2fs (%.1f dB below peak)",
                         s / TARGET_SR, e / TARGET_SR, peak_db - seg_db)
    return kept


def _clarity_gate(audio: np.ndarray, sr: int, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Drop segments that sound noise-like (high spectral flatness).

    Spectral flatness (Wiener entropy) near 1.0 = white-noise-like;
    near 0.0 = tonal / voiced speech.  Segments above the threshold
    are unintelligible and would hurt downstream speaker-ID and analysis.
    """
    if not segments:
        return segments

    kept = []
    for s, e in segments:
        chunk = audio[s:e]
        flatness = librosa.feature.spectral_flatness(y=chunk, n_fft=min(1024, len(chunk)))[0]
        mean_flatness = float(np.mean(flatness))
        if mean_flatness < CLARITY_FLATNESS_THRESHOLD:
            kept.append((s, e))
        else:
            logger.debug("clarity-gate dropped segment %.2fs–%.2fs (flatness=%.3f)",
                         s / sr, e / sr, mean_flatness)
    return kept


def _per_segment_normalize(
    audio: np.ndarray,
    segments: List[Tuple[int, int]],
    sr: int,
) -> Tuple[np.ndarray, List[float]]:
    """Normalize each speech segment to TARGET_SEGMENT_RMS_DBFS independently.

    Returns (modified_audio, per_segment_gain_db_list).  Background
    (non-speech) samples are attenuated by BACKGROUND_ATTENUATION_DB.
    Short linear ramps at segment edges smooth the gain transition
    between background attenuation and the per-segment speech gain.
    """
    bg_gain = float(10.0 ** (BACKGROUND_ATTENUATION_DB / 20.0))
    ramp_len = int(round(GAIN_RAMP_MS * sr / 1000.0))

    # Start with background-attenuated copy
    out = (np.copy(audio) * bg_gain).astype(np.float32)
    per_seg_gains: List[float] = []

    for s, e in segments:
        chunk = audio[s:e]
        seg_rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        seg_db = 20.0 * np.log10(max(seg_rms, 1e-8))
        gain_db = float(np.clip(
            TARGET_SEGMENT_RMS_DBFS - seg_db,
            MIN_SEGMENT_BOOST_DB,
            MAX_SEGMENT_BOOST_DB,
        ))
        speech_gain = float(10.0 ** (gain_db / 20.0))
        per_seg_gains.append(gain_db)

        # Apply speech gain to the segment body
        out[s:e] = (chunk * speech_gain).astype(np.float32)

        # Smooth ramp at edges: interpolate from bg_gain to speech_gain
        seg_ramp = min(ramp_len, (e - s) // 2)
        if seg_ramp > 1:
            ramp_up = np.linspace(bg_gain / max(speech_gain, 1e-8), 1.0, seg_ramp, dtype=np.float32)
            ramp_down = np.linspace(1.0, bg_gain / max(speech_gain, 1e-8), seg_ramp, dtype=np.float32)
            out[s:s + seg_ramp] *= ramp_up
            out[e - seg_ramp:e] *= ramp_down

        logger.debug("per-seg norm %.2fs–%.2fs  rms=%.1f dB  gain=%.1f dB",
                      s / sr, e / sr, seg_db, gain_db)

    return out, per_seg_gains


def _compress_segment(chunk: np.ndarray, sr: int) -> np.ndarray:
    """Apply soft-knee dynamic range compression to a single audio chunk.

    Uses a sliding-window RMS envelope with attack / release ballistics
    to gently even out intra-segment volume variation (agent leaning
    closer / further from mic within one utterance).
    """
    if chunk.size < 2:
        return chunk.astype(np.float32)

    win_len = max(1, int(round(COMPRESSOR_WINDOW_MS * sr / 1000.0)))

    # --- RMS envelope via convolution ---
    sq = np.square(chunk, dtype=np.float64)
    kernel = np.ones(win_len, dtype=np.float64) / win_len
    env = np.sqrt(np.convolve(sq, kernel, mode="same"))
    env = np.maximum(env, 1e-10)
    env_db = (20.0 * np.log10(env)).astype(np.float64)

    # --- gain reduction curve (above threshold) ---
    above = env_db - COMPRESSOR_THRESHOLD_DBFS
    above = np.maximum(above, 0.0)
    reduction_db = above * (1.0 - 1.0 / COMPRESSOR_RATIO)

    # --- attack / release ballistics (sample-by-sample) ---
    attack_coeff = float(np.exp(-1.0 / max(COMPRESSOR_ATTACK_MS * sr / 1000.0, 1.0)))
    release_coeff = float(np.exp(-1.0 / max(COMPRESSOR_RELEASE_MS * sr / 1000.0, 1.0)))

    smoothed = np.empty_like(reduction_db)
    smoothed[0] = reduction_db[0]
    for i in range(1, len(reduction_db)):
        if reduction_db[i] > smoothed[i - 1]:
            smoothed[i] = attack_coeff * smoothed[i - 1] + (1.0 - attack_coeff) * reduction_db[i]
        else:
            smoothed[i] = release_coeff * smoothed[i - 1] + (1.0 - release_coeff) * reduction_db[i]

    # --- apply gain ---
    gain_linear = (10.0 ** (-smoothed / 20.0)).astype(np.float32)
    return (chunk * gain_linear).astype(np.float32)


def _apply_intra_segment_compression(
    audio: np.ndarray,
    segments: List[Tuple[int, int]],
    sr: int,
) -> np.ndarray:
    """Apply soft compression to each speech segment independently."""
    out = np.copy(audio).astype(np.float32)
    for s, e in segments:
        out[s:e] = _compress_segment(out[s:e], sr)
    return out


def _concat_segments_crossfade(
    audio: np.ndarray,
    segments: List[Tuple[int, int]],
    sr: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Concatenate segments with short crossfades to avoid clicks.

    Returns (output_audio, boundaries) where each boundary is
    (orig_start, orig_end, clean_start, clean_end) — contiguous and
    non-overlapping in the clean domain (crossfade split at midpoint).
    """
    chunks = [audio[s:e].astype(np.float32) for s, e in segments if e > s]
    if not chunks:
        return audio.astype(np.float32), []

    if len(chunks) == 1:
        return chunks[0], [(segments[0][0], segments[0][1], 0, len(chunks[0]))]

    # Clamp crossfade so it never exceeds half of the shortest chunk
    cf = int(round(CROSSFADE_MS * sr / 1000.0))
    min_half = min(len(c) // 2 for c in chunks)
    cf = max(min(cf, min_half), 0)

    if cf < 2:
        # Fallback to raw concatenation
        concat = np.concatenate(chunks).astype(np.float32)
        cursor = 0
        boundaries: List[Tuple[int, int, int, int]] = []
        for i, (s, e) in enumerate(segments):
            length = e - s
            boundaries.append((s, e, cursor, cursor + length))
            cursor += length
        return concat, boundaries

    total_len = sum(len(c) for c in chunks) - cf * (len(chunks) - 1)
    out = np.zeros(total_len, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)

    pos = 0
    for i, chunk in enumerate(chunks):
        c = chunk.copy()
        if i > 0:
            c[:cf] *= fade_in
        if i < len(chunks) - 1:
            c[-cf:] *= fade_out
        out[pos:pos + len(c)] += c
        if i < len(chunks) - 1:
            pos += len(c) - cf
        # last chunk: pos doesn't advance further

    # Build contiguous clean boundaries (split crossfade at midpoint)
    boundaries = []
    write_pos = 0
    for i in range(len(chunks)):
        chunk_len = len(chunks[i])
        if i == 0:
            clean_start = 0
            clean_end = write_pos + chunk_len - cf // 2
            write_pos += chunk_len - cf
        elif i == len(chunks) - 1:
            clean_start = boundaries[-1][3]
            clean_end = total_len
        else:
            clean_start = boundaries[-1][3]
            clean_end = clean_start + chunk_len - cf  # full chunk minus both half-crossfades
            write_pos += chunk_len - cf
        boundaries.append((segments[i][0], segments[i][1], clean_start, clean_end))

    # Sanity: last boundary must end at total_len
    if boundaries and boundaries[-1][3] != total_len:
        boundaries[-1] = (*boundaries[-1][:3], total_len)

    return out, boundaries


def _apply_master_normalization(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32)
    target_peak = float(10.0 ** (TARGET_PEAK_DBFS / 20.0))
    gain = min(1.0, target_peak / peak) if peak > target_peak else target_peak / peak
    out = (audio * gain).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _build_segment_map(
    boundaries: List[Tuple[int, int, int, int]],
    total_samples: int,
    cleaned_total_samples: int,
    sr: int,
    per_segment_gains_db: List[float],
) -> Dict:
    mapped = []
    for idx, (orig_start, orig_end, clean_start, clean_end) in enumerate(boundaries):
        gain_db = per_segment_gains_db[idx] if idx < len(per_segment_gains_db) else 0.0
        mapped.append({
            "index": idx,
            "orig_start_samp": int(orig_start),
            "orig_end_samp": int(orig_end),
            "clean_start_samp": int(clean_start),
            "clean_end_samp": int(clean_end),
            "orig_start": float(orig_start / sr),
            "orig_end": float(orig_end / sr),
            "clean_start": float(clean_start / sr),
            "clean_end": float(clean_end / sr),
            "gain_db": float(gain_db),
        })

    mean_gain = float(np.mean(per_segment_gains_db)) if per_segment_gains_db else 0.0

    return {
        "target_sr": int(sr),
        "original_total_samples": int(total_samples),
        "cleaned_total_samples": int(cleaned_total_samples),
        "original_duration": float(total_samples / sr),
        "cleaned_duration": float(cleaned_total_samples / sr),
        "segments": mapped,
        "processing_parameters": _processing_params(mean_gain),
    }


def _processing_params(applied_speech_gain_db: float = 0.0) -> Dict:
    """Return the full processing-parameters dict for the segment map."""
    return {
        "spectral_gate_stationary_strength": SPECTRAL_GATE_STRENGTH_STATIONARY,
        "spectral_gate_dynamic_strength": SPECTRAL_GATE_STRENGTH_DYNAMIC,
        "spectral_gate_n_fft": SPECTRAL_GATE_N_FFT,
        "spectral_gate_win_length": SPECTRAL_GATE_WIN_LENGTH,
        "spectral_gate_hop_length": SPECTRAL_GATE_HOP_LENGTH,
        "spectral_gate_time_smooth_ms": SPECTRAL_GATE_TIME_SMOOTH_MS,
        "spectral_gate_freq_smooth_hz": SPECTRAL_GATE_FREQ_SMOOTH_HZ,
        "bandpass_low_hz": BANDPASS_LOW_HZ,
        "bandpass_high_hz": BANDPASS_HIGH_HZ,
        "vad_top_db": VAD_TOP_DB,
        "vad_frame_length": VAD_FRAME_LENGTH,
        "vad_hop_length": VAD_HOP_LENGTH,
        "min_speech_ms": MIN_SPEECH_MS,
        "speech_pad_start_ms": SPEECH_PAD_START_MS,
        "speech_pad_end_ms": SPEECH_PAD_END_MS,
        "min_silence_to_remove_ms": MIN_SILENCE_TO_REMOVE_MS,
        "near_field_rms_drop_db": NEAR_FIELD_RMS_DROP_DB,
        "clarity_flatness_threshold": CLARITY_FLATNESS_THRESHOLD,
        "background_attenuation_db": BACKGROUND_ATTENUATION_DB,
        "target_segment_rms_dbfs": TARGET_SEGMENT_RMS_DBFS,
        "max_segment_boost_db": MAX_SEGMENT_BOOST_DB,
        "min_segment_boost_db": MIN_SEGMENT_BOOST_DB,
        "applied_speech_gain_db": float(applied_speech_gain_db),
        "target_peak_dbfs": TARGET_PEAK_DBFS,
        "compressor_threshold_dbfs": COMPRESSOR_THRESHOLD_DBFS,
        "compressor_ratio": COMPRESSOR_RATIO,
        "compressor_attack_ms": COMPRESSOR_ATTACK_MS,
        "compressor_release_ms": COMPRESSOR_RELEASE_MS,
        "compressor_window_ms": COMPRESSOR_WINDOW_MS,
        "crossfade_ms": CROSSFADE_MS,
    }


def _trivial_map(total_samples: int, sr: int) -> Dict:
    duration = float(total_samples / sr)
    return {
        "target_sr": int(sr),
        "original_total_samples": int(total_samples),
        "cleaned_total_samples": int(total_samples),
        "original_duration": duration,
        "cleaned_duration": duration,
        "segments": [{
            "index": 0,
            "orig_start_samp": 0,
            "orig_end_samp": int(total_samples),
            "clean_start_samp": 0,
            "clean_end_samp": int(total_samples),
            "orig_start": 0.0,
            "orig_end": duration,
            "clean_start": 0.0,
            "clean_end": duration,
            "gain_db": 0.0,
        }],
        "processing_parameters": _processing_params(0.0),
    }


def _save_audio(audio: np.ndarray, sr: int, output_path: str) -> str:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".wav", ".flac", ".ogg"):
        sf.write(output_path, audio, sr)
        return output_path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        sf.write(tmp_wav, audio, sr)
        cmd = [
            "ffmpeg", "-y", "-i", tmp_wav,
            "-ac", "1", "-ar", str(sr),
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg conversion failed")
        return output_path
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
