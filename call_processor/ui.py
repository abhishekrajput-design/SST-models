# `from __future__ import annotations` MUST be the first real statement.
# It treats every type annotation in this file as a string, so PEP 604 (`X | Y`)
# and PEP 585 (`list[T]`) syntax used below work on Python 3.9 as well as 3.10+
# (the production AWS box runs 3.9, the dev box runs 3.11).
from __future__ import annotations

import os
import re
import sys
import gc
import hashlib
import json
import base64
import shutil
import subprocess
import threading
import http.server
import socketserver
import tempfile
import time
from urllib.parse import urlparse, unquote, parse_qs
from pathlib import Path

# Use expandable CUDA segments to prevent memory fragmentation.
# Without this, after the first Parakeet/ECAPA run, PyTorch's reserved CUDA
# pool becomes fragmented and DeepFilterNet3 (needs 4 GB contiguous) fails on
# subsequent runs — even though enough VRAM is nominally free.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# SpeechBrain 1.1.0 has a Windows bug: importutils.py checks for "/inspect.py"
# (Unix path) but Windows uses "\\inspect.py".  We patch it once at startup
# in site-packages (see call_processor/src/transcribers/__init__.py comments).
# No runtime patch needed here — the file is fixed on this machine.

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PORT = 8080
PROCESSED_DIR = "data/processed"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Basic auth for the UI. OFF by default — the browser would otherwise pop a
# login prompt every visit. Enable by setting CALLPROC_AUTH_REQUIRED=1 (any
# non-empty truthy value) and optionally CALLPROC_USER / CALLPROC_PASS to
# override the defaults.
_AUTH_REQUIRED_RAW = os.environ.get("CALLPROC_AUTH_REQUIRED", "").strip().lower()
AUTH_REQUIRED = _AUTH_REQUIRED_RAW in ("1", "true", "yes", "on")
AUTH_USER = os.environ.get("CALLPROC_USER", "abhishek")
AUTH_PASS = os.environ.get("CALLPROC_PASS", "123456")
_AUTH_EXPECTED = "Basic " + base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode()).decode()
AUTH_REALM = "Call Processor"

# FFmpeg PATH on Windows (WinGet install). On Linux the system ffmpeg is used.
_FFMPEG_BIN = r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
_ENV = os.environ.copy()
if sys.platform == "win32" and os.path.isdir(_FFMPEG_BIN):
    if _FFMPEG_BIN not in _ENV.get("PATH", ""):
        _ENV["PATH"] = _FFMPEG_BIN + os.pathsep + _ENV.get("PATH", "")

# FFmpeg filter chain for call-center audio with background noise.
# Two profiles available:
#   * default  — original mild chain, safe for already-clean phone audio
#   * deep     — adds lowpass to cut high-freq radio/music, stronger afftdn
#                pass, and silenceremove to drop dead air (≥2.5 s of silence
#                below -50 dB). Turned on by SST_DEEP_ENHANCE=1.
# silenceremove parameters are deliberately conservative so quiet but real
# speech (which holds energy above -45 dB) is never cut — only dead-air.
_DEEP_ENHANCE = os.environ.get("SST_DEEP_ENHANCE", "").strip().lower() in (
    "1", "true", "yes", "on",
)

AUDIO_FILTER_DEFAULT = (
    "aresample=44100,"                        # upsample first — loudnorm needs ≥44.1k
    "highpass=f=80,"                          # strip low-freq HVAC/rumble
    "afftdn=nf=-25:nt=w,"                     # FFmpeg spectral denoiser pass
    "loudnorm=I=-16:TP=-1.5:LRA=11,"          # bring quiet phone audio up to standard level
    "dynaudnorm=p=0.9:m=100:s=5:g=15"         # boost quiet passages locally
)

AUDIO_FILTER_DEEP = (
    # DO NOT stack extra denoising before ASR (extra afftdn passes / narrow
    # lowpass cuts speech sibilants and makes Parakeet skip words). DFN3
    # (after this chain) handles radio/background noise.
    "aresample=44100,"
    "highpass=f=80,"
    "afftdn=nf=-25:nt=w,"
    # silenceremove BEFORE loudnorm/dynaudnorm — those would normalize the
    # silent regions up above the -50 dB threshold and the silence detector
    # would never fire. Drop ≥2.0 s runs where peak stays below -50 dB.
    # Conversational pauses are <2 s; quiet speech holds energy above -45 dB,
    # so this only ever cuts true dead air / radio-only sections.
    "silenceremove=stop_periods=-1"
    ":stop_duration=2.0"
    ":stop_threshold=-50dB"
    ":detection=peak,"
    "loudnorm=I=-16:TP=-1.5:LRA=11,"
    "dynaudnorm=p=0.9:m=100:s=5:g=15"
)

AUDIO_FILTER = AUDIO_FILTER_DEEP if _DEEP_ENHANCE else AUDIO_FILTER_DEFAULT
print(f"[UI] enhance profile: {'DEEP' if _DEEP_ENHANCE else 'default'}", flush=True)

# ── Pipeline status (shared) ──────────────────────────────────────────────────
_status = {
    "running": False,
    "stage_num": 0,
    "stage": "Idle",
    "message": "",
    "done": False,
    "error": None,
    "result_id": None,
    "cancel_requested": False,
    "started_at": None,
    "stage_started_at": None,
    "updated_at": None,
    "completed_at": None,
    "elapsed_seconds": 0.0,
    "stage_elapsed_seconds": 0.0,
    "processing_time_seconds": None,
}
_status_lock = threading.Lock()


class PipelineCancelled(Exception):
    """Raised between stages when the user requested a cancel."""


def _check_cancelled():
    with _status_lock:
        if _status.get("cancel_requested"):
            raise PipelineCancelled("user requested cancel")

# ── Enhancement job status (separate from main pipeline) ─────────────────────
_enhance_status: dict = {}          # call_id -> {"running", "done", "error", "paths"}
_enhance_lock = threading.Lock()

# ── Agent enrollment status ───────────────────────────────────────────────────
_enroll_status: dict = {"running": False, "done": False, "error": None, "message": ""}
_enroll_lock   = threading.Lock()
# Directory of known-agent recordings used for voice enrollment
AGENT_RECORDINGS_DIR = r"C:\Users\abhis\Desktop\SST-models\Agents-recoding\zak_recodings"

_TARGET_AGENT_ALIASES = {
    "zak": "zak_raissi_barnet",
    "zakraissi": "zak_raissi_barnet",
    "zakraissibarnet": "zak_raissi_barnet",
    "zak_raissi": "zak_raissi_barnet",
    "zak_raissi_barnet": "zak_raissi_barnet",
    "hussein": "hussein_mohamed",
    "hussien": "hussein_mohamed",
    "husseinmohamed": "hussein_mohamed",
    "hussienmohamed": "hussein_mohamed",
    "hussein_mohamed": "hussein_mohamed",
}


def _agent_slug_from_hint(*hints: str) -> str | None:
    for raw in hints:
        if not raw:
            continue
        text = str(raw).lower()
        compact = re.sub(r"[^a-z0-9]+", "", text)
        underscored = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        for key in (underscored, compact):
            if key in _TARGET_AGENT_ALIASES:
                return _TARGET_AGENT_ALIASES[key]
        if "zak" in compact and "raissi" in compact:
            return "zak_raissi_barnet"
        if "hussein" in compact or "hussien" in compact:
            return "hussein_mohamed"
    return None


def _resolve_target_agent_slug(requested: str | None, *hints: str) -> str | None:
    requested = (requested or "").strip()
    if requested and requested.lower() not in {"auto", "none", "all"}:
        resolved = _agent_slug_from_hint(requested)
        if resolved:
            return resolved
        candidate = re.sub(r"[^a-z0-9_]+", "_", requested.lower()).strip("_")
        return candidate or None
    return _agent_slug_from_hint(*hints)


def _target_presence_floor(target_agent_slug: str | None) -> float:
    env_name = "SST_TARGET_AGENT_PRESENCE_FLOOR" if target_agent_slug else "SST_AGENT_PRESENCE_FLOOR"
    default = "0.24" if target_agent_slug else "0.35"
    try:
        return float(os.getenv(env_name, default))
    except (TypeError, ValueError):
        return float(default)


def _set_status(stage_num: int, stage: str, message: str):
    now = time.time()
    with _status_lock:
        if _status.get("stage_num") != stage_num or _status.get("stage") != stage:
            _status["stage_started_at"] = now
        _status["stage_num"] = stage_num
        _status["stage"]     = stage
        _status["message"]   = message
        _status["updated_at"] = now
        started = _status.get("started_at")
        if started:
            _status["elapsed_seconds"] = round(now - float(started), 2)
        stage_started = _status.get("stage_started_at")
        if stage_started:
            _status["stage_elapsed_seconds"] = round(now - float(stage_started), 2)


def _status_snapshot() -> dict:
    now = time.time()
    with _status_lock:
        data = dict(_status)
    started = data.get("started_at")
    completed = data.get("completed_at")
    if started:
        end = float(completed) if completed else now
        data["elapsed_seconds"] = round(max(0.0, end - float(started)), 2)
    stage_started = data.get("stage_started_at")
    if stage_started:
        end = float(completed) if completed else now
        data["stage_elapsed_seconds"] = round(max(0.0, end - float(stage_started)), 2)
    return data


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Audio Enhancement Pipelines
#  All clip to max_seconds (default 300 = 5 min) so they finish fast.
#  All produce 44.1 kHz / 128 kbps / mono MP3 for browser playback.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _run_ffmpeg(cmd: list, timeout: int = 180):
    """Run an FFmpeg command with timeout + explicit env so it never hangs silently."""
    result = subprocess.run(
        cmd, env=_ENV, capture_output=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode(errors='replace')[-500:]}")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _extract_marked_json(stdout: str) -> str:
    start = stdout.find("RESULT_START")
    end = stdout.find("RESULT_END")
    if start < 0 or end < 0:
        return ""
    return stdout[start + len("RESULT_START"):end]


def _transcribe_isolated(audio_path: str, model_name: str, device: str, language: str, timeout_s: int) -> list:
    """Run a transcriber in a child process so native CUDA aborts do not kill the UI."""
    root = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.abspath(audio_path)
    code = f"""
import gc, json, os, sys
root = {root!r}
sys.path.insert(0, root)
os.chdir(root)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from src.transcribers import get_transcriber
tr = get_transcriber(sys.argv[2], device=sys.argv[3])
tr.load()
segments = tr.transcribe(sys.argv[1], language=sys.argv[4])
tr.unload()
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception:
    pass
print("RESULT_START" + json.dumps(segments, ensure_ascii=False) + "RESULT_END")
sys.stdout.flush()
"""
    fd, script = tempfile.mkstemp(suffix="_ui_transcribe.py", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        proc = subprocess.run(
            [sys.executable, "-u", script, audio_path, model_name, device, language],
            cwd=root,
            env=_ENV,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{model_name} {device} subprocess failed: {details[-1500:]}")
    raw = _extract_marked_json(proc.stdout)
    if not raw:
        details = (proc.stdout + proc.stderr).strip()
        raise RuntimeError(f"{model_name} {device} subprocess returned no transcript JSON: {details[-1500:]}")
    return json.loads(raw)


def _is_clean_audio(path: str, sample_seconds: int = 30) -> bool:
    """Heuristic: True if `path` is already broadcast-quality clean speech.

    DFN3 over-enhances pristine sources and can mute synthetic test files.
    We sniff the first 30s, compute RMS and spectral flatness; high RMS +
    low flatness ⇒ clean speech, skip DFN3.
    """
    try:
        import soundfile as sf
        import numpy as np
        info = sf.info(path)
        n = min(info.frames, info.samplerate * sample_seconds)
        a, sr = sf.read(path, frames=n, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if a.size < sr:
            return False
        rms = float(np.sqrt((a ** 2).mean()))
        # Spectral flatness: geometric / arithmetic mean of power spectrum.
        # Speech ≈ 0.05-0.20, white noise → 1.0, tone → 0.0.
        spec = np.abs(np.fft.rfft(a[:sr * 5])) ** 2 + 1e-12
        flat = float(np.exp(np.log(spec).mean()) / spec.mean())
        clean = rms > 0.10 and flat < 0.20
        print(f"[UI] clean-audio check: RMS={rms:.3f} flatness={flat:.3f} → {'CLEAN' if clean else 'NOISY'}", flush=True)
        return clean
    except Exception as e:
        print(f"[UI] clean-audio check failed ({e}); assuming noisy.", flush=True)
        return False


def _to_wav(input_path: str, max_seconds: int = None) -> str:
    """Convert any audio → 16 kHz mono WAV (optionally clipped)."""
    wav = input_path + f"_tmp{os.getpid()}.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-ac", "1", "-ar", "16000", wav]
    _run_ffmpeg(cmd)
    return wav


def _wav_to_mp3(wav_path: str, out_mp3: str):
    """44.1 kHz / 128 kbps / mono MP3 — browser-compatible."""
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", wav_path,
        "-ac", "1", "-ar", "44100", "-b:a", "128k", out_mp3,
    ])


# ── Pipeline 1: FFmpeg (highpass + afftdn + dynaudnorm) ─────────────────────
def _make_playback_loud_mp3(input_path: str, output_path: str):
    """Louder browser-playback copy only; never use this for ASR input."""
    af = (
        "aresample=44100,"
        "loudnorm=I=-13:TP=-1.0:LRA=8,"
        "dynaudnorm=p=0.98:m=35:s=10:g=22,"
        "volume=3dB,"
        "alimiter=limit=0.98"
    )
    _run_ffmpeg(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-af", af,
            "-ac", "1", "-ar", "44100", "-b:a", "128k", output_path,
        ],
        timeout=600,
    )


def _enhance_ffmpeg(input_path: str, output_path: str, max_seconds: int = 300):
    """FFmpeg DSP chain — fast, always works, full audio for main pipeline."""
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-af", AUDIO_FILTER, "-ac", "1", "-ar", "44100", "-b:a", "128k", output_path]
    _run_ffmpeg(cmd)


# ── Pipeline 2: noisereduce — spectral gating (CPU, scipy-based) ─────────────
def _enhance_noisereduce(input_path: str, output_path: str, max_seconds: int = 300):
    """
    noisereduce spectral gating.
    pip install noisereduce soundfile
    Clips audio to max_seconds so it finishes quickly.
    """
    import numpy as np
    import soundfile as sf
    import noisereduce as nr

    wav = _to_wav(input_path, max_seconds=max_seconds)
    out_wav = output_path.replace(".mp3", ".wav")
    try:
        data, rate = sf.read(wav, dtype="float32")
        reduced = nr.reduce_noise(y=data, sr=rate, stationary=False, prop_decrease=0.85)
        reduced = np.clip(reduced * 3.0, -0.98, 0.98)
        sf.write(out_wav, reduced, rate, subtype="PCM_16")
        _wav_to_mp3(out_wav, output_path)
    finally:
        for p in (wav, out_wav):
            if os.path.exists(p):
                os.remove(p)


# ── Pipeline 3: angelina 10-stage desk-recording cleanup ─────────────────────
def _enhance_angelina(input_path: str, output_path: str, max_seconds: int = 300):
    """
    Run the angelina cleanup pipeline (src/audio_cleanup.py):
    two-pass noise gate · bandpass 85-6500 Hz · VAD · energy gate ·
    clarity gate · per-segment normalization · DRC · crossfade
    Outputs 16 kHz mono WAV (re-encoded to MP3 for browser preview).
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from src.audio_cleanup import clean_audio

    wav = _to_wav(input_path, max_seconds=max_seconds)
    out_wav = output_path.replace(".mp3", ".wav")
    try:
        clean_audio(wav, out_wav)
        _wav_to_mp3(out_wav, output_path)
    finally:
        if _os.path.exists(wav):
            _os.remove(wav)


# ── Pipeline 3b: DeepFilterNet3 — neural (GPU, soundfile I/O) ────────────────
def _enhance_deepfilternet(input_path: str, output_path: str,
                           max_seconds: int = None, chunk_seconds: int = 300):
    """
    DeepFilterNet3 neural noise suppression.
    Processes audio in chunk_seconds chunks to avoid VRAM OOM on long files.
    """
    import numpy as np
    import torch
    import soundfile as sf
    from df.enhance import enhance, init_df
    import torchaudio.functional as F_ta

    wav = _to_wav(input_path, max_seconds=max_seconds)
    out_wav = output_path.replace(".mp3", ".wav")
    try:
        model, df_state, _ = init_df()
        target_sr = df_state.sr()          # 48000 Hz

        data, orig_sr = sf.read(wav, dtype="float32")
        if data.ndim == 1:
            data = data[None, :]           # [1, T]
        else:
            data = data.T
        audio = torch.from_numpy(data)

        if orig_sr != target_sr:
            audio = F_ta.resample(audio, orig_sr, target_sr)

        chunk_samples = int(chunk_seconds * target_sr)
        total = audio.shape[-1]

        if total <= chunk_samples:
            enhanced = enhance(model, df_state, audio)
            enh_np = enhanced.cpu().numpy().squeeze(0)
            del enhanced
        else:
            # Process in chunks to stay within 6 GB VRAM
            parts = []
            for start in range(0, total, chunk_samples):
                chunk = audio[:, start: start + chunk_samples]
                enh_chunk = enhance(model, df_state, chunk)
                parts.append(enh_chunk.cpu().numpy().squeeze(0))
                del enh_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            enh_np = np.concatenate(parts)

        enh_np = np.clip(enh_np * 3.0, -0.98, 0.98)
        sf.write(out_wav, enh_np, target_sr, subtype="PCM_16")
        _wav_to_mp3(out_wav, output_path)

        del model, df_state, audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        for p in (wav, out_wav):
            if os.path.exists(p):
                os.remove(p)


# ── Pipeline 4: SpeechBrain MetricGAN+ — GAN trained on PESQ metric ──────────
def _enhance_metricgan(input_path: str, output_path: str, max_seconds: int = 300):
    """
    SpeechBrain MetricGAN-Plus-VoiceBank.
    PESQ 3.15 / STOI 93.0 — trained specifically to maximise speech quality score.
    pip install speechbrain  (already installed for speaker recognition)
    Model auto-downloads from HuggingFace on first run (~50 MB).
    """
    import numpy as np
    import torch
    import soundfile as sf
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    from speechbrain.utils.fetching import LocalStrategy

    wav = _to_wav(input_path, max_seconds=max_seconds)
    out_wav = output_path.replace(".mp3", ".wav")
    try:
        data, rate = sf.read(wav, dtype="float32")    # 16 kHz mono

        enhancer = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="models/metricgan",
            local_strategy=LocalStrategy.COPY,   # Windows: no symlink privileges needed
        )

        noisy = torch.from_numpy(data).unsqueeze(0)   # [1, T]
        lengths = torch.tensor([1.0])

        with torch.no_grad():
            enhanced = enhancer.enhance_batch(noisy, lengths)

        enh_np = enhanced.squeeze(0).cpu().numpy()
        enh_np = np.clip(enh_np * 3.0, -0.98, 0.98)

        sf.write(out_wav, enh_np, rate, subtype="PCM_16")
        _wav_to_mp3(out_wav, output_path)

        del enhancer, noisy, enhanced
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        for p in (wav, out_wav):
            if os.path.exists(p):
                os.remove(p)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Helpers for path derivation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _derive_paths(original_path: str) -> dict:
    """Return expected file paths for all 4 enhancement variants."""
    d = os.path.dirname(original_path)
    fname = os.path.basename(original_path)
    return {
        "ffmpeg":      os.path.join(d, f"enhanced_{fname}").replace("\\", "/"),
        "playback_loud": os.path.join(d, f"loud_{fname}").replace("\\", "/"),
        "noisereduce": os.path.join(d, f"nr_{fname}").replace("\\", "/"),
        "deepfilter":  os.path.join(d, f"df_{fname}").replace("\\", "/"),
        "metricgan":   os.path.join(d, f"mg_{fname}").replace("\\", "/"),
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Audio trimming — remove gaps where no speech was detected
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _trim_to_speech(audio_path: str, segments: list, out_path: str,
                    pad_s: float = 0.3, merge_gap_s: float = 1.5) -> bool:
    """
    Cut the audio to only the regions covered by transcription segments.
    Segments closer than merge_gap_s are merged into one block.
    Each block gets pad_s seconds of context on each side.
    Returns True on success, False on failure (caller keeps original audio).
    """
    import tempfile
    if not segments:
        print("[UI] Trim: no segments — skipped.", flush=True)
        return False

    # 1. Build merged speech blocks
    blocks = []
    cur_start = max(0.0, segments[0]["start"] - pad_s)
    cur_end   = segments[0]["end"] + pad_s
    for seg in segments[1:]:
        seg_start = max(0.0, seg["start"] - pad_s)
        seg_end   = seg["end"] + pad_s
        if seg_start - cur_end <= merge_gap_s:
            cur_end = max(cur_end, seg_end)
        else:
            blocks.append((cur_start, cur_end))
            cur_start = seg_start
            cur_end   = seg_end
    blocks.append((cur_start, cur_end))

    if not blocks:
        print("[UI] Trim: no blocks built — skipped.", flush=True)
        return False

    total_speech = sum(e - s for s, e in blocks)
    print(f"[UI] Trimming audio: {len(blocks)} speech blocks, {total_speech:.1f}s kept", flush=True)

    # 2. Write the filter as a complex filtergraph script to avoid Windows
    #    command-line length limits with long aselect expressions.
    expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in blocks)
    filter_script = None
    try:
        fd, filter_script = tempfile.mkstemp(suffix=".txt", prefix="trim_filter_")
        with os.fdopen(fd, "w", encoding="ascii") as f:
            # complex filtergraph: label input → aselect → output
            f.write(f"[0:a]aselect='{expr}',asetpts=N/SR/TB[aout]\n")

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-filter_complex_script", filter_script,
             "-map", "[aout]",
             "-ar", "22050", "-ac", "1", "-b:a", "64k",
             out_path],
            env=_ENV, capture_output=True, timeout=180,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"[UI] Trim ffmpeg failed (rc={result.returncode}): {err}", flush=True)
            return False

        ok = os.path.exists(out_path) and os.path.getsize(out_path) > 1000
        if ok:
            sz = os.path.getsize(out_path)
            print(f"[UI] Trim done: {out_path} ({sz:,} bytes)", flush=True)
        else:
            print(f"[UI] Trim: output missing or empty at {out_path}", flush=True)
        return ok
    except Exception as e:
        print(f"[UI] Trim exception: {repr(e)}", flush=True)
        return False
    finally:
        if filter_script and os.path.exists(filter_script):
            try:
                os.unlink(filter_script)
            except OSError:
                pass


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Whisper-only transcription (fallback when HF_TOKEN not set)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _transcribe_inline(audio_path: str, whisper_model: str = "whisper-large-v3-turbo",
                       original_path: str = "", target_agent_slug: str | None = None) -> str:
    """
    Run any registered transcriber in-process (Whisper, Cohere, Parakeet, Qwen3, VibeVoice).
    Returns result_id (basename of FFmpeg-enhanced file).
    Writes result.json to data/processed/<result_id>/.
    Mono-only pipeline: all audio is downmixed to 16 kHz mono regardless of source.
    """
    import time
    from datetime import datetime
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.transcribers import get_transcriber

    requested_model = whisper_model

    def _audio_duration_s(path: str) -> float:
        try:
            import soundfile as _sf
            return float(_sf.info(path).duration)
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
                capture_output=True,
                text=True,
                env=_ENV,
                timeout=30,
            )
            return float(json.loads(r.stdout)["format"]["duration"])
        except Exception:
            return 0.0

    dur_s = _audio_duration_s(audio_path)
    target_agent_slug = _resolve_target_agent_slug(
        target_agent_slug,
        original_path,
        audio_path,
    )
    presence_floor = _target_presence_floor(target_agent_slug)

    base     = os.path.splitext(os.path.basename(audio_path))[0]
    # Use a per-model directory so each model run is stored separately
    dir_name = f"{base}__{whisper_model}"
    out_dir  = os.path.join("data", "processed", dir_name)
    os.makedirs(out_dir, exist_ok=True)


    # ── Normalize helper ──────────────────────────────────────────────────────
    norm_dir = os.path.join("data", "processed", base)
    os.makedirs(norm_dir, exist_ok=True)

    # Upsample to 44.1k before loudnorm — single-pass loudnorm on 8 kHz phone
    # sources produces a silent WAV. After loudnorm we resample down to 16 kHz
    # for the transcriber via the -ar arg below.
    _NORM_AF = "aresample=44100,loudnorm=I=-16:TP=-1.5:LRA=11,dynaudnorm=p=0.9:m=100:s=5"

    def _make_norm_wav(src: str, dst: str):
        """16 kHz mono WAV from src with normalisation. Mono-only pipeline."""
        af = f"aformat=channel_layouts=mono,{_NORM_AF}"
        _run_ffmpeg(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-af", af, dst],
            timeout=600,   # 30-min audio needs up to ~5 min for loudnorm
        )

    def _make_minimal_asr_wav(src: str, dst: str):
        """16 kHz mono WAV without gain/noise processing for low-level desk audio."""
        af = "aformat=channel_layouts=mono,aresample=16000"
        _run_ffmpeg(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-af", af, dst],
            timeout=600,
        )

    def _make_main_conversation_wav(src: str, dst: str) -> tuple[str, dict]:
        """
        Keep only high-energy near-mic conversation regions before ASR.
        Background voices/noise that never cross the strong gate are dropped.
        """
        enabled = os.getenv("SST_MAIN_CONVERSATION_GATE", "1").strip().lower()
        report = {"enabled": enabled not in {"0", "false", "no", "off"}, "applied": False}
        if not report["enabled"]:
            report["reason"] = "disabled"
            return src, report
        if not os.path.isfile(src):
            report["reason"] = "source_missing"
            return src, report

        try:
            import numpy as _np
            import soundfile as _sf

            audio, sr = _sf.read(src, dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = _np.mean(audio, axis=1)
            if audio.size < max(1, int(sr * 0.5)):
                report["reason"] = "too_short"
                return src, report

            frame_s = float(os.getenv("SST_MAIN_CONV_FRAME_SECONDS", "0.20") or "0.20")
            hop_s = float(os.getenv("SST_MAIN_CONV_HOP_SECONDS", "0.05") or "0.05")
            start_ratio = float(os.getenv("SST_MAIN_CONV_START_RATIO", "0.40") or "0.40")
            hold_ratio = float(os.getenv("SST_MAIN_CONV_HOLD_RATIO", "0.18") or "0.18")
            min_region_s = float(os.getenv("SST_MAIN_CONV_MIN_REGION_SECONDS", "0.35") or "0.35")
            pad_s = float(os.getenv("SST_MAIN_CONV_PAD_SECONDS", "0.25") or "0.25")
            merge_gap_s = float(os.getenv("SST_MAIN_CONV_MERGE_GAP_SECONDS", "0.80") or "0.80")
            min_keep_s = float(os.getenv("SST_MAIN_CONV_MIN_KEEP_SECONDS", "8.0") or "8.0")
            min_reduction_ratio = float(os.getenv("SST_MAIN_CONV_MIN_REDUCTION_RATIO", "0.03") or "0.03")

            frame = max(1, int(frame_s * sr))
            hop = max(1, int(hop_s * sr))
            starts = _np.arange(0, max(1, audio.size - frame + 1), hop, dtype=_np.int64)
            if starts.size == 0:
                report["reason"] = "no_frames"
                return src, report

            sq = _np.square(audio.astype(_np.float32, copy=False), dtype=_np.float32)
            csum = _np.concatenate(([0.0], _np.cumsum(sq, dtype=_np.float64)))
            energy = (csum[starts + frame] - csum[starts]) / float(frame)
            rms = _np.sqrt(_np.maximum(energy, 0.0))
            active_rms = rms[_np.isfinite(rms) & (rms > 1e-6)]
            if active_rms.size < 3:
                report["reason"] = "no_energy"
                return src, report

            strong_level = float(_np.percentile(active_rms, 95))
            start_threshold = strong_level * start_ratio
            hold_threshold = strong_level * hold_ratio
            high = rms >= start_threshold
            low = rms >= hold_threshold
            if not bool(_np.any(high)):
                report.update({
                    "reason": "no_main_mic_regions",
                    "strong_level": round(strong_level, 6),
                    "start_threshold": round(float(start_threshold), 6),
                })
                return src, report

            blocks: list[tuple[float, float]] = []
            i = 0
            while i < len(low):
                if not low[i]:
                    i += 1
                    continue
                j = i + 1
                while j < len(low) and low[j]:
                    j += 1
                if bool(_np.any(high[i:j])):
                    start_t = max(0.0, float(starts[i]) / sr - pad_s)
                    end_t = min(float(audio.size) / sr, float(starts[j - 1] + frame) / sr + pad_s)
                    if end_t - start_t >= min_region_s:
                        blocks.append((start_t, end_t))
                i = j

            if not blocks:
                report["reason"] = "no_blocks"
                return src, report

            merged: list[tuple[float, float]] = []
            for start_t, end_t in blocks:
                if merged and start_t - merged[-1][1] <= merge_gap_s:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end_t))
                else:
                    merged.append((start_t, end_t))

            source_duration_s = float(audio.size) / sr
            kept_s = sum(end_t - start_t for start_t, end_t in merged)
            dropped_s = max(0.0, source_duration_s - kept_s)
            if kept_s < min_keep_s:
                report.update({
                    "reason": "too_little_main_audio",
                    "source_duration_s": round(source_duration_s, 2),
                    "kept_duration_s": round(kept_s, 2),
                })
                return src, report
            if source_duration_s > 0 and dropped_s / source_duration_s < min_reduction_ratio:
                report.update({
                    "reason": "no_meaningful_reduction",
                    "source_duration_s": round(source_duration_s, 2),
                    "kept_duration_s": round(kept_s, 2),
                })
                return src, report

            chunks = []
            for start_t, end_t in merged:
                start_i = max(0, int(start_t * sr))
                end_i = min(audio.size, int(end_t * sr))
                if end_i > start_i:
                    chunks.append(audio[start_i:end_i])
            if not chunks:
                report["reason"] = "no_audio_chunks"
                return src, report
            gated_audio = _np.concatenate(chunks).astype(_np.float32, copy=False)
            _sf.write(dst, gated_audio, sr, subtype="PCM_16")

            if not os.path.isfile(dst) or os.path.getsize(dst) < 1000:
                report["reason"] = "trim_output_missing"
                return src, report

            report.update({
                "applied": True,
                "audio_file": dst.replace("\\", "/"),
                "source_duration_s": round(source_duration_s, 2),
                "kept_duration_s": round(kept_s, 2),
                "dropped_duration_s": round(dropped_s, 2),
                "kept_ratio": round(kept_s / source_duration_s, 4) if source_duration_s else 0.0,
                "blocks": len(merged),
                "start_ratio": start_ratio,
                "hold_ratio": hold_ratio,
                "start_threshold": round(float(start_threshold), 6),
                "hold_threshold": round(float(hold_threshold), 6),
            })
            return dst, report
        except Exception as exc:
            report["reason"] = f"failed: {exc}"
            return src, report

    # ── Load transcriber (shared for both channels) ───────────────────────────
    cuda_ok = _cuda_available()
    transcriber_device = "cuda" if cuda_ok else "cpu"
    is_parakeet_model = whisper_model.startswith("parakeet-tdt-")
    isolated_transcriber = is_parakeet_model
    transcriber = None
    if isolated_transcriber:
        _set_status(2, "Transcription", f"Loading {whisper_model} on {transcriber_device.upper()} (isolated)...")
        print(
            f"[UI] Loading transcriber: {whisper_model} "
            f"on {transcriber_device} in isolated subprocess",
            flush=True,
        )
    else:
        _set_status(2, "Transcription", f"Loading {whisper_model} on {transcriber_device.upper()}...")
        print(f"[UI] Loading transcriber: {whisper_model} on {transcriber_device}")
        transcriber = get_transcriber(whisper_model, device=transcriber_device)
        transcriber.load()

    def _retry_empty_transcript(
        current_segments: list,
        wav_path: str,
        reason: str,
    ) -> list:
        nonlocal transcriber, whisper_model
        if current_segments:
            return current_segments
        fallback_model = os.getenv("SST_EMPTY_TRANSCRIPT_FALLBACK", "whisper-large-v3-turbo").strip()
        if not fallback_model or fallback_model.lower() in {"0", "false", "no", "off", "none"}:
            return []
        if fallback_model == whisper_model:
            return []

        _set_status(3, "Transcription", f"{reason}; retrying with {fallback_model}...")
        print(f"[UI] {reason}; retrying with {fallback_model}", flush=True)
        if transcriber is not None:
            try:
                transcriber.unload()
            except Exception:
                pass
        whisper_model = fallback_model
        retry_device = "cuda" if _cuda_available() else "cpu"
        transcriber = get_transcriber(whisper_model, device=retry_device)
        transcriber.load()
        return transcriber.transcribe(wav_path, language="en") or []

    def _split_asr_segments_on_word_slices(
        text_segments: list,
        model_name: str,
        wav_path: str | None = None,
    ) -> tuple[list, dict]:
        """Split long Whisper segments on word-time pauses and sentence boundaries."""
        default_enabled = "0"
        enabled = os.getenv("SST_WORD_SLICE_SPLIT", default_enabled).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return text_segments, {"enabled": False, "reason": "disabled"}

        min_gap_s = float(os.getenv("SST_WORD_SLICE_MIN_GAP_S", "0.16") or "0.16")
        min_words = max(1, int(os.getenv("SST_WORD_SLICE_MIN_WORDS", "2") or "2"))
        max_words = max(min_words * 2 + 1, int(os.getenv("SST_WORD_SLICE_MAX_WORDS", "12") or "12"))
        min_chunk_s = float(os.getenv("SST_WORD_SLICE_MIN_SECONDS", "0.55") or "0.55")
        pause_window_s = float(os.getenv("SST_WORD_SLICE_AUDIO_PAUSE_WINDOW_S", "0.18") or "0.18")
        pause_ratio = float(os.getenv("SST_WORD_SLICE_AUDIO_PAUSE_RATIO", "0.80") or "0.80")
        pause_floor = float(os.getenv("SST_WORD_SLICE_AUDIO_PAUSE_FLOOR", "0.012") or "0.012")
        audio = None
        sr = None
        if wav_path and os.path.isfile(wav_path):
            try:
                import numpy as _np
                import soundfile as _sf
                audio, sr = _sf.read(wav_path, dtype="float32")
                if getattr(audio, "ndim", 1) > 1:
                    audio = _np.mean(audio, axis=1)
            except Exception:
                audio = None
                sr = None

        def _word_text(word: dict) -> str:
            return str(word.get("word") or word.get("text") or "").strip()

        def _join_words(words: list[dict]) -> str:
            text = " ".join(_word_text(w) for w in words if _word_text(w)).strip()
            text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
            text = re.sub(r"([$£€])\s+", r"\1", text)
            return re.sub(r"\s+", " ", text).strip()

        def _usable_words(seg: dict) -> list[dict]:
            out = []
            for raw in seg.get("words") or []:
                if not isinstance(raw, dict):
                    continue
                token = _word_text(raw)
                if not token:
                    continue
                try:
                    start = float(raw.get("start"))
                    end = float(raw.get("end"))
                except (TypeError, ValueError):
                    continue
                if end <= start:
                    continue
                row = dict(raw)
                row["word"] = token
                row["start"] = start
                row["end"] = end
                out.append(row)
            return out

        def _valid_cut(words: list[dict], cut_after: int) -> bool:
            left = words[: cut_after + 1]
            right = words[cut_after + 1 :]
            if len(left) < min_words or len(right) < min_words:
                return False
            left_s = float(left[-1]["end"]) - float(left[0]["start"])
            right_s = float(right[-1]["end"]) - float(right[0]["start"])
            return left_s >= min_chunk_s and right_s >= min_chunk_s

        def _rms(start_s: float, end_s: float) -> float | None:
            if audio is None or not sr:
                return None
            start = max(0, int(start_s * sr))
            end = min(len(audio), int(end_s * sr))
            if end <= start:
                return None
            try:
                import numpy as _np
                chunk = audio[start:end]
                return float(_np.sqrt(_np.mean(_np.square(chunk))))
            except Exception:
                return None

        def _has_audio_pause(words: list[dict], cut_after: int) -> bool:
            if audio is None or not sr:
                return False
            center = (float(words[cut_after]["end"]) + float(words[cut_after + 1]["start"])) / 2.0
            local = _rms(center - pause_window_s / 2.0, center + pause_window_s / 2.0)
            segment = _rms(float(words[0]["start"]), float(words[-1]["end"]))
            if local is None or segment is None or segment <= 0:
                return False
            return local <= max(pause_floor, segment * pause_ratio)

        def _boundary_reason(words: list[dict], cut_after: int) -> str | None:
            if not _valid_cut(words, cut_after):
                return None
            token = _word_text(words[cut_after])
            if re.search(r"[.!?][\"')\]]*$", token):
                return "sentence"
            gap = float(words[cut_after + 1]["start"]) - float(words[cut_after]["end"])
            if gap >= min_gap_s:
                return "pause"
            if _has_audio_pause(words, cut_after):
                return "audio_pause"
            return None

        def _enforce_max_words(chunks: list[tuple[int, int, str]], words: list[dict]) -> list[tuple[int, int, str]]:
            out = []
            for start, end, reason in chunks:
                pending = [(start, end, reason)]
                while pending:
                    c_start, c_end, c_reason = pending.pop(0)
                    count = c_end - c_start
                    if count <= max_words:
                        out.append((c_start, c_end, c_reason))
                        continue
                    sub = words[c_start:c_end]
                    target = c_start + max(min_words, count // 2) - 1
                    candidates = []
                    for cut_after in range(c_start + min_words - 1, c_end - min_words):
                        token = _word_text(words[cut_after])
                        gap = float(words[cut_after + 1]["start"]) - float(words[cut_after]["end"])
                        score = -abs(cut_after - target)
                        if gap >= min_gap_s:
                            score += 100 + gap
                            cut_reason = "pause"
                        elif _has_audio_pause(words[c_start:c_end], cut_after - c_start):
                            score += 80
                            cut_reason = "audio_pause"
                        elif re.search(r"[,;:][\"')\]]*$", token):
                            score += 40
                            cut_reason = "clause"
                        else:
                            continue
                        candidates.append((score, cut_after, cut_reason))
                    if not candidates:
                        out.append((c_start, c_end, c_reason))
                        continue
                    _score, cut_after, cut_reason = max(candidates, key=lambda item: item[0])
                    pending.insert(0, (cut_after + 1, c_end, c_reason))
                    pending.insert(0, (c_start, cut_after + 1, cut_reason))
            return out

        split_segments = []
        original_count = len(text_segments)
        split_count = 0
        skipped_no_words = 0
        for seg in text_segments:
            words = _usable_words(seg)
            if len(words) < (min_words * 2):
                if not words:
                    skipped_no_words += 1
                split_segments.append(seg)
                continue

            chunks: list[tuple[int, int, str]] = []
            start_idx = 0
            reason = "original"
            for idx in range(0, len(words) - 1):
                boundary = _boundary_reason(words[start_idx:], idx - start_idx)
                if boundary:
                    chunks.append((start_idx, idx + 1, boundary))
                    start_idx = idx + 1
                    reason = boundary
            chunks.append((start_idx, len(words), reason))
            chunks = _enforce_max_words(chunks, words)

            if len(chunks) <= 1:
                split_segments.append(seg)
                continue

            parent_start = float(seg.get("start", words[0]["start"]))
            parent_end = float(seg.get("end", words[-1]["end"]))
            parent_text = str(seg.get("text") or "")
            for slice_idx, (start, end, slice_reason) in enumerate(chunks):
                chunk_words = words[start:end]
                text = _join_words(chunk_words)
                if not text:
                    continue
                child = dict(seg)
                child["start"] = round(float(chunk_words[0]["start"]), 2)
                child["end"] = round(float(chunk_words[-1]["end"]), 2)
                child["text"] = text
                child["words"] = [
                    {
                        **w,
                        "start": round(float(w["start"]), 3),
                        "end": round(float(w["end"]), 3),
                    }
                    for w in chunk_words
                ]
                child["slice_parent_start"] = round(parent_start, 2)
                child["slice_parent_end"] = round(parent_end, 2)
                child["slice_parent_text"] = parent_text
                child["slice_index"] = slice_idx
                child["slice_count"] = len(chunks)
                child["slice_split_reason"] = slice_reason
                split_segments.append(child)
            split_count += len(chunks) - 1

        meta = {
            "enabled": True,
            "original_segments": original_count,
            "final_segments": len(split_segments),
            "added_segments": split_count,
            "skipped_no_word_timestamps": skipped_no_words,
            "min_gap_s": min_gap_s,
            "max_words": max_words,
            "min_words": min_words,
            "min_chunk_s": min_chunk_s,
            "audio_pause_window_s": pause_window_s,
            "audio_pause_ratio": pause_ratio,
        }
        return sorted(split_segments, key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0)))), meta

    t0 = time.time()
    diarization_applied = False
    speaker_stats: dict   = {}
    diar_result: dict = {}

    # ── Mono path: voiceprint-first multi-speaker diarization ────────────────
    norm_wav = os.path.join(norm_dir, f"norm_{base}.wav")
    if not os.path.exists(norm_wav):
        _set_status(1, "Transcription", "Normalizing audio to 16 kHz mono...")
        print("[UI] Normalizing audio...")
        _make_norm_wav(audio_path, norm_wav)
    _check_cancelled()

    asr_wav = norm_wav
    diar_wav = norm_wav
    segment_audio_path = audio_path
    main_conversation_gate = {"enabled": False, "applied": False, "reason": "not_started"}
    asr_audio_mode = "normalized"
    raw_asr_env = os.getenv("SST_PARAKEET_RAW_ASR", "1")
    raw_asr_enabled = str(raw_asr_env).strip().lower() not in {"0", "false", "no", "off"}
    raw_asr_min_s = float(os.getenv("SST_RAW_ASR_MIN_SECONDS", "600") or "600")
    if is_parakeet_model and raw_asr_enabled and dur_s >= raw_asr_min_s:
        asr_wav = os.path.join(norm_dir, f"asr_raw_{base}.wav")
        asr_audio_mode = "minimal_resample"
        if not os.path.exists(asr_wav):
            _set_status(1, "Transcription", "Preparing minimal-resample ASR audio...")
            print("[UI] Preparing minimal-resample ASR audio for long desk recording...")
            _make_minimal_asr_wav(audio_path, asr_wav)
        _check_cancelled()

    gate_wav = os.path.join(norm_dir, f"mainconv_{base}.wav")
    gated_wav, main_conversation_gate = _make_main_conversation_wav(asr_wav, gate_wav)
    if main_conversation_gate.get("applied"):
        asr_wav = gated_wav
        diar_wav = gated_wav
        segment_audio_path = gated_wav
        asr_audio_mode = f"{asr_audio_mode}+main_conversation_gate"
        dur_s = _audio_duration_s(gated_wav) or dur_s
        print(
            "[UI] Main conversation gate applied: "
            f"kept {main_conversation_gate.get('kept_duration_s')}s, "
            f"dropped {main_conversation_gate.get('dropped_duration_s')}s, "
            f"blocks={main_conversation_gate.get('blocks')}",
            flush=True,
        )
    else:
        print(
            f"[UI] Main conversation gate skipped: {main_conversation_gate.get('reason')}",
            flush=True,
        )
    _check_cancelled()

    _set_status(3, "Transcription", f"Transcribing with {whisper_model} on {transcriber_device.upper()}...")
    print(f"[UI] Transcribing with {whisper_model} on {transcriber_device} ({asr_audio_mode})...")
    if isolated_transcriber:
        timeout_s = max(900, int(max(dur_s, 60.0) * 6))
        segments = _transcribe_isolated(
            asr_wav,
            whisper_model,
            transcriber_device,
            "en",
            timeout_s,
        )
    else:
        segments = transcriber.transcribe(asr_wav, language="en")
    segments = _retry_empty_transcript(
        segments,
        asr_wav,
        "No transcript segments returned",
    )
    segments, asr_slice_split = _split_asr_segments_on_word_slices(segments, whisper_model, asr_wav)
    if asr_slice_split.get("added_segments"):
        print(
            f"[UI] Word-slice split: {asr_slice_split['original_segments']} -> "
            f"{asr_slice_split['final_segments']} segment(s) "
            f"(+{asr_slice_split['added_segments']})",
            flush=True,
        )
    segments, asr_repeat_loop_filter = _collapse_repeated_asr_loops(segments)
    if asr_repeat_loop_filter.get("removed_segments"):
        print(
            f"[UI] Collapsed {asr_repeat_loop_filter['removed_segments']} ASR repeat-loop segment(s) "
            f"in {asr_repeat_loop_filter.get('collapsed_runs', 0)} run(s)",
            flush=True,
        )
    _check_cancelled()

    agent_time = customer_time = 0.0
    agent_turns = customer_turns = 0
    speech_only_added = 0
    transcript_coverage: dict = {}

    def _add_untranscribed_speaker_segments(
        text_segments: list,
        speaker_segments: list,
        add_rows: bool = False,
    ) -> tuple[list, dict]:
        """Track diarized customer speech with no ASR text.

        Overlapped/background speakers are often detected by diarization even
        when ASR only emits the dominant voice. We keep this as coverage data
        by default instead of creating noisy placeholder transcript bubbles.
        """
        min_gap_s = float(os.getenv("SST_SPEECH_ONLY_MIN_SECONDS", "1.0") or "1.0")
        include_agent = os.getenv("SST_SPEECH_ONLY_INCLUDE_AGENT", "0").strip() == "1"
        covers = sorted(
            (
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
            )
            for seg in text_segments
            if str(seg.get("text") or "").strip()
        )

        def _covered_seconds(start: float, end: float) -> float:
            total = 0.0
            for c_start, c_end in covers:
                if c_end <= start:
                    continue
                if c_start >= end:
                    break
                total += max(0.0, min(end, c_end) - max(start, c_start))
            return min(max(total, 0.0), max(end - start, 0.0))

        def _uncovered_ranges(start: float, end: float) -> list[tuple[float, float]]:
            ranges = []
            cursor = start
            for c_start, c_end in covers:
                if c_end <= cursor:
                    continue
                if c_start >= end:
                    break
                if c_start > cursor and (c_start - cursor) >= min_gap_s:
                    ranges.append((cursor, min(c_start, end)))
                cursor = max(cursor, c_end)
                if cursor >= end:
                    break
            if end > cursor and (end - cursor) >= min_gap_s:
                ranges.append((cursor, end))
            return ranges

        coverage: dict = {
            "added_count": 0,
            "added_seconds": 0.0,
            "hidden_count": 0,
            "hidden_seconds": 0.0,
            "per_speaker": {},
        }
        additions = []
        for spk_seg in speaker_segments or []:
            start = float(spk_seg.get("start", 0.0))
            end = float(spk_seg.get("end", 0.0))
            dur = max(end - start, 0.0)
            if dur <= 0:
                continue
            label = spk_seg.get("display_speaker") or spk_seg.get("speaker") or "Unknown"
            role = spk_seg.get("identified_speaker") or "CUSTOMER"
            row = coverage["per_speaker"].setdefault(
                label,
                {"speaker_seconds": 0.0, "transcribed_overlap_seconds": 0.0, "speech_only_seconds": 0.0},
            )
            covered = _covered_seconds(start, end)
            row["speaker_seconds"] += dur
            row["transcribed_overlap_seconds"] += covered
            if role == "AGENT" and not include_agent:
                continue
            for gap_start, gap_end in _uncovered_ranges(start, end):
                gap_dur = max(gap_end - gap_start, 0.0)
                if gap_dur < min_gap_s:
                    continue
                if add_rows:
                    additions.append({
                        "start": round(gap_start, 3),
                        "end": round(gap_end, 3),
                        "speaker": spk_seg.get("speaker") or "SPEAKER_99",
                        "identified_speaker": role,
                        "display_speaker": label,
                        "agent_name": spk_seg.get("agent_name"),
                        "text": "Speech detected - no transcript",
                        "avg_score": None,
                        "speech_only": True,
                        "transcription_missing": True,
                    })
                    coverage["added_count"] += 1
                    coverage["added_seconds"] += gap_dur
                else:
                    coverage["hidden_count"] += 1
                    coverage["hidden_seconds"] += gap_dur
                row["speech_only_seconds"] += gap_dur

        merged = list(text_segments) + additions
        merged.sort(key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", 0.0))))
        coverage["added_seconds"] = round(float(coverage["added_seconds"]), 2)
        coverage["hidden_seconds"] = round(float(coverage["hidden_seconds"]), 2)
        for row in coverage["per_speaker"].values():
            row["speaker_seconds"] = round(float(row["speaker_seconds"]), 2)
            row["transcribed_overlap_seconds"] = round(float(row["transcribed_overlap_seconds"]), 2)
            row["speech_only_seconds"] = round(float(row["speech_only_seconds"]), 2)
        return merged, coverage

    def _split_text_segments_on_speaker_turns(
        text_segments: list,
        speaker_segments: list,
    ) -> tuple[list, dict]:
        """Split ASR rows where diarization shows a real speaker turn inside.

        This uses only timestamps and diarized speaker intervals. Text is split
        proportionally so the UI can show one row per acoustic speaker change.
        """
        enabled = os.getenv("SST_SPEAKER_TURN_TEXT_SPLIT", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return text_segments, {"enabled": False, "reason": "disabled"}
        if not text_segments or not speaker_segments:
            return text_segments, {"enabled": True, "reason": "missing_inputs", "added_segments": 0}

        min_parent_s = float(os.getenv("SST_SPEAKER_TURN_SPLIT_MIN_PARENT_SECONDS", "1.00") or "1.00")
        min_child_s = float(os.getenv("SST_SPEAKER_TURN_SPLIT_MIN_CHILD_SECONDS", "0.45") or "0.45")
        min_child_ratio = float(os.getenv("SST_SPEAKER_TURN_SPLIT_MIN_CHILD_RATIO", "0.12") or "0.12")
        min_coverage_ratio = float(os.getenv("SST_SPEAKER_TURN_SPLIT_MIN_COVERAGE_RATIO", "0.55") or "0.55")
        max_gap_s = float(os.getenv("SST_SPEAKER_TURN_SPLIT_MERGE_GAP", "0.08") or "0.08")
        max_parts = max(2, int(os.getenv("SST_SPEAKER_TURN_SPLIT_MAX_PARTS", "5") or "5"))
        min_words_per_child = max(1, int(os.getenv("SST_SPEAKER_TURN_SPLIT_MIN_WORDS", "2") or "2"))

        def _word_slices(text: str, weights: list[float]) -> list[str] | None:
            words = str(text or "").split()
            if len(words) < len(weights) * min_words_per_child:
                return None
            total = sum(max(w, 0.001) for w in weights)
            cuts = [0]
            running = 0.0
            for weight in weights[:-1]:
                running += max(weight, 0.001)
                cuts.append(round((running / total) * len(words)))
            cuts.append(len(words))
            chunks = []
            for idx in range(len(cuts) - 1):
                start = int(cuts[idx])
                end = int(cuts[idx + 1])
                min_start = idx * min_words_per_child
                max_end = len(words) - ((len(cuts) - 2 - idx) * min_words_per_child)
                start = max(start, min_start)
                end = min(max(end, start + min_words_per_child), max_end)
                chunks.append(" ".join(words[start:end]).strip())
            if any(not chunk for chunk in chunks):
                return None
            return chunks

        def _merge_parts(parts: list[dict]) -> list[dict]:
            if not parts:
                return []
            merged = [dict(parts[0])]
            for part in parts[1:]:
                cur = merged[-1]
                if (
                    part["speaker"] == cur["speaker"]
                    and part["start"] - cur["end"] <= max_gap_s
                ):
                    cur["end"] = max(cur["end"], part["end"])
                    cur["duration"] = cur["end"] - cur["start"]
                else:
                    merged.append(dict(part))
            return merged

        out = []
        original_count = len(text_segments)
        split_parents = 0
        added_segments = 0
        skipped_short_text = 0
        skipped_low_coverage = 0

        for seg in text_segments:
            if seg.get("speech_only"):
                out.append(seg)
                continue
            text = str(seg.get("text") or "").strip()
            start_s = float(seg.get("start", 0.0) or 0.0)
            end_s = float(seg.get("end", start_s) or start_s)
            dur_s = max(end_s - start_s, 0.0)
            if dur_s < min_parent_s or not text:
                out.append(seg)
                continue

            base_speaker = str(seg.get("speaker") or "")
            overlaps = []
            boundaries = {start_s, end_s}
            for spk_seg in speaker_segments:
                ss_start = float(spk_seg.get("start", 0.0) or 0.0)
                ss_end = float(spk_seg.get("end", ss_start) or ss_start)
                overlap_start = max(start_s, ss_start)
                overlap_end = min(end_s, ss_end)
                overlap = max(overlap_end - overlap_start, 0.0)
                if overlap <= 0:
                    continue
                boundaries.add(overlap_start)
                boundaries.add(overlap_end)
                overlaps.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "duration": overlap,
                    "speaker": spk_seg.get("speaker") or "SPEAKER_99",
                    "identified_speaker": spk_seg.get("identified_speaker") or "UNKNOWN",
                    "display_speaker": spk_seg.get("display_speaker") or "UNKNOWN",
                    "agent_name": spk_seg.get("agent_name"),
                    "agent_slug": spk_seg.get("agent_slug"),
                })

            timeline_parts = []
            points = sorted(boundaries)
            base_prefer_ratio = float(os.getenv("SST_SPEAKER_TURN_SPLIT_BASE_PREFER_RATIO", "0.70") or "0.70")
            for left, right in zip(points, points[1:]):
                interval = max(right - left, 0.0)
                if interval <= 0.04:
                    continue
                candidates = []
                for item in overlaps:
                    covered = max(0.0, min(right, item["end"]) - max(left, item["start"]))
                    if covered > 0:
                        candidates.append((covered, item))
                if not candidates:
                    continue
                best_cover, best_item = max(candidates, key=lambda x: x[0])
                if base_speaker:
                    base_candidates = [
                        (covered, item)
                        for covered, item in candidates
                        if str(item.get("speaker") or "") == base_speaker
                    ]
                    if base_candidates:
                        base_cover, base_item = max(base_candidates, key=lambda x: x[0])
                        if base_cover >= best_cover * base_prefer_ratio:
                            best_item = base_item
                timeline_parts.append({
                    **best_item,
                    "start": left,
                    "end": right,
                    "duration": interval,
                })

            parts = _merge_parts(sorted(timeline_parts, key=lambda p: (p["start"], p["end"])))
            parts = [
                p for p in parts
                if p["duration"] >= min_child_s and (p["duration"] / dur_s) >= min_child_ratio
            ]
            if len({p["speaker"] for p in parts}) < 2:
                out.append(seg)
                continue
            if len(parts) > max_parts:
                parts = sorted(parts, key=lambda p: p["duration"], reverse=True)[:max_parts]
                parts = sorted(parts, key=lambda p: (p["start"], p["end"]))
            coverage = sum(p["duration"] for p in parts)
            if (coverage / dur_s) < min_coverage_ratio:
                skipped_low_coverage += 1
                out.append(seg)
                continue

            chunks = _word_slices(text, [p["duration"] for p in parts])
            if not chunks:
                skipped_short_text += 1
                out.append(seg)
                continue

            for idx, (part, chunk_text) in enumerate(zip(parts, chunks)):
                child = dict(seg)
                child["start"] = round(float(part["start"]), 3)
                child["end"] = round(float(part["end"]), 3)
                child["text"] = chunk_text
                child["speaker"] = part["speaker"]
                child["identified_speaker"] = part["identified_speaker"]
                child["display_speaker"] = part["display_speaker"]
                if part.get("agent_name"):
                    child["agent_name"] = part.get("agent_name")
                else:
                    child.pop("agent_name", None)
                if part.get("agent_slug"):
                    child["agent_slug"] = part.get("agent_slug")
                else:
                    child.pop("agent_slug", None)
                child["speaker_turn_split"] = True
                child["speaker_turn_parent_start"] = round(start_s, 3)
                child["speaker_turn_parent_end"] = round(end_s, 3)
                child["speaker_turn_index"] = idx
                child["speaker_turn_count"] = len(parts)
                child["speaker_turn_source"] = "diarization_dominant_speaker_timeline"
                out.append(child)

            split_parents += 1
            added_segments += len(parts) - 1

        out.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
        return out, {
            "enabled": True,
            "source": "diarization_timestamp_overlap",
            "original_segments": original_count,
            "final_segments": len(out),
            "split_parent_segments": split_parents,
            "added_segments": added_segments,
            "min_parent_seconds": min_parent_s,
            "min_child_seconds": min_child_s,
            "min_child_ratio": min_child_ratio,
            "min_coverage_ratio": min_coverage_ratio,
            "skipped_short_text": skipped_short_text,
            "skipped_low_coverage": skipped_low_coverage,
        }

    def _smooth_short_unknown_segments(text_segments: list) -> int:
        """Assign tiny unknown snippets to a close known neighbor."""
        max_dur_s = float(os.getenv("SST_UNKNOWN_SMOOTH_MAX_SECONDS", "3.5") or "3.5")
        max_gap_s = float(os.getenv("SST_UNKNOWN_SMOOTH_MAX_GAP", "0.8") or "0.8")

        def _known(seg: dict) -> bool:
            role = seg.get("identified_speaker")
            return bool(role and role != "UNKNOWN" and not seg.get("speech_only"))

        smoothed = 0
        for idx, seg in enumerate(text_segments):
            role = seg.get("identified_speaker")
            display = seg.get("display_speaker")
            if role != "UNKNOWN" and display != "UNKNOWN":
                continue
            if seg.get("speech_only"):
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if max(end - start, 0.0) > max_dur_s:
                continue

            candidates: list[tuple[float, dict]] = []
            for prev in reversed(text_segments[:idx]):
                if _known(prev):
                    gap = start - float(prev.get("end", start))
                    if 0 <= gap <= max_gap_s:
                        candidates.append((gap, prev))
                    break
            for nxt in text_segments[idx + 1:]:
                if _known(nxt):
                    gap = float(nxt.get("start", end)) - end
                    if 0 <= gap <= max_gap_s:
                        candidates.append((gap, nxt))
                    break
            if not candidates:
                continue

            _, neighbor = sorted(candidates, key=lambda item: item[0])[0]
            seg["speaker"] = neighbor.get("speaker", seg.get("speaker", "SPEAKER_99"))
            seg["identified_speaker"] = neighbor.get("identified_speaker", "CUSTOMER")
            seg["display_speaker"] = neighbor.get("display_speaker") or (
                neighbor.get("agent_name") if neighbor.get("identified_speaker") == "AGENT" else "Customer"
            )
            if seg["identified_speaker"] == "AGENT":
                if neighbor.get("agent_name"):
                    seg["agent_name"] = neighbor.get("agent_name")
            else:
                seg.pop("agent_name", None)
            seg["role_smoothed"] = True
            smoothed += 1
        return smoothed

    def _fallback_unknown_text_to_customer(text_segments: list) -> int:
        """Keep unmatched transcript text out of agent enrollment/role claims."""
        fallback_count = 0
        for seg in text_segments:
            if seg.get("speech_only"):
                continue
            if seg.get("identified_speaker") != "UNKNOWN" and seg.get("display_speaker") != "UNKNOWN":
                continue
            if not str(seg.get("text", "")).strip():
                continue
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = "Customer"
            seg["role_fallback"] = "customer_no_speaker_overlap"
            seg.pop("agent_name", None)
            seg.pop("agent_slug", None)
            fallback_count += 1
        return fallback_count

    def _repair_agent_roles_from_text_cues(
        text_segments: list,
        agent_name: str | None,
        audio_duration_s: float,
        speaker_count: int,
        agent_slug: str | None = None,
        cluster_match_table: dict | None = None,
    ) -> dict:
        """Disabled: role identity must come from voiceprint matching only."""
        return {
            "enabled": False,
            "promoted_speakers": [],
            "promoted_segments": 0,
            "reason": "voiceprint_only",
        }

    def _uses_segment_role_voiceprint(entry: dict) -> bool:
        """Return True only for voiceprints allowed in segment-role scoring."""
        explicit = entry.get("use_for_segment_role")
        if isinstance(explicit, str):
            explicit_norm = explicit.strip().lower()
            if explicit_norm in {"0", "false", "no", "off"}:
                return False
            if explicit_norm in {"1", "true", "yes", "on"}:
                return True
        elif explicit is False:
            return False
        elif explicit:
            return True

        legacy = entry.get("segment_role_voiceprint")
        if isinstance(legacy, str):
            legacy_norm = legacy.strip().lower()
            if legacy_norm in {"0", "false", "no", "off"}:
                return False
            if legacy_norm in {"1", "true", "yes", "on"}:
                return True
        elif legacy is False:
            return False
        elif legacy:
            return True

        return entry.get("source") == "verified_transcript_labels"

    def _repair_agent_roles_from_voiceprint_clusters(
        text_segments: list,
        agent_name: str | None,
        agent_slug: str | None,
        cluster_match_table: dict | None,
    ) -> dict:
        """Promote split advisor clusters that strongly match the selected agent.

        Sortformer often splits one nearby desk advisor into several clusters.
        Voice matching initially picks only the strongest cluster as AGENT. This
        pass repairs the other same-agent clusters, but only when the target
        agent is the cluster's top acoustic match and the cluster has enough
        transcribed duration to avoid promoting short background speech.
        """
        enabled = os.getenv("SST_AGENT_CLUSTER_ROLE_REPAIR", "0").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return {"enabled": False, "promoted_speakers": [], "promoted_segments": 0}
        if not agent_slug or not agent_name or agent_name in {"None", "Unknown Agent"}:
            return {"enabled": True, "promoted_speakers": [], "promoted_segments": 0, "reason": "no_identified_agent"}
        if not cluster_match_table:
            return {"enabled": True, "promoted_speakers": [], "promoted_segments": 0, "reason": "no_cluster_match_table"}

        min_sim = float(os.getenv("SST_AGENT_CLUSTER_ROLE_REPAIR_MIN_SIM", "0.72") or "0.72")
        min_margin = float(os.getenv("SST_AGENT_CLUSTER_ROLE_REPAIR_MIN_MARGIN", "0.10") or "0.10")
        target_only_min_sim = float(os.getenv("SST_AGENT_CLUSTER_ROLE_REPAIR_TARGET_ONLY_MIN_SIM", "0.72") or "0.72")
        min_seconds = float(os.getenv("SST_AGENT_CLUSTER_ROLE_REPAIR_MIN_TEXT_S", "12.0") or "12.0")
        min_segments = int(os.getenv("SST_AGENT_CLUSTER_ROLE_REPAIR_MIN_SEGMENTS", "2") or "2")

        speakers: dict[str, dict] = {}
        for seg in text_segments:
            if seg.get("speech_only"):
                continue
            if seg.get("identified_speaker") != "CUSTOMER":
                continue
            spk = str(seg.get("speaker") or "")
            if not spk or spk == "SPEAKER_99":
                continue
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            bucket = speakers.setdefault(spk, {"segments": [], "seconds": 0.0})
            bucket["segments"].append(seg)
            bucket["seconds"] += max(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 0.0)

        promoted_speakers = []
        promoted_segments = 0
        for spk, bucket in speakers.items():
            matches = cluster_match_table.get(spk) or {}
            if not isinstance(matches, dict) or agent_slug not in matches:
                continue
            scored = sorted(
                (
                    (slug, float((data or {}).get("similarity") or 0.0))
                    for slug, data in matches.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if not scored:
                continue
            top_slug, top_sim = scored[0]
            target_sim = float((matches.get(agent_slug) or {}).get("similarity") or 0.0)
            best_other = max((sim for slug, sim in scored if slug != agent_slug), default=0.0)
            margin = target_sim - best_other
            target_only = not any(slug != agent_slug for slug, _sim in scored)
            if top_slug != agent_slug:
                continue
            if target_only:
                if target_sim < target_only_min_sim:
                    continue
            elif target_sim < min_sim or margin < min_margin:
                continue
            if bucket["seconds"] < min_seconds or len(bucket["segments"]) < min_segments:
                continue

            for seg in bucket["segments"]:
                seg["identified_speaker"] = "AGENT"
                seg["display_speaker"] = agent_name
                seg["agent_name"] = agent_name
                seg["agent_slug"] = agent_slug
                seg["role_voiceprint_repair"] = "same_agent_split_cluster"
                promoted_segments += 1
            promoted_speakers.append(
                {
                    "speaker": spk,
                    "segments": len(bucket["segments"]),
                    "seconds": round(float(bucket["seconds"]), 2),
                    "target_similarity": round(float(target_sim), 3),
                    "margin": round(float(margin), 3),
                    "top_slug": top_slug,
                    "target_only": target_only,
                }
            )

        return {
            "enabled": True,
            "promoted_speakers": promoted_speakers,
            "promoted_segments": promoted_segments,
            "min_similarity": min_sim,
            "min_margin": min_margin,
            "target_only_min_similarity": target_only_min_sim,
            "min_seconds": min_seconds,
        }

    def _foreground_customer_blocks_agent(
        seg: dict,
        seconds: float,
        target_similarity: float | None = None,
        target_margin: float | None = None,
        trusted_segment_voiceprint: bool = False,
    ) -> bool:
        """Block background-agent bleed from overriding a clean customer speaker slice."""
        enabled = os.getenv("SST_FOREGROUND_ROLE_GUARD", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return False
        if seg.get("identified_speaker") != "CUSTOMER":
            return False
        if trusted_segment_voiceprint:
            try:
                sim = float(target_similarity)
                margin = float(target_margin)
            except (TypeError, ValueError):
                sim = margin = 0.0
            trusted_min_sim = float(
                os.getenv(
                    "SST_FOREGROUND_TRUSTED_VOICE_MIN_SIM",
                    os.getenv("SST_AGENT_SEGMENT_VOICE_MIN_SIM", "0.40"),
                ) or "0.40"
            )
            trusted_min_margin = float(
                os.getenv(
                    "SST_FOREGROUND_TRUSTED_VOICE_MIN_MARGIN",
                    os.getenv("SST_AGENT_SEGMENT_VOICE_MIN_MARGIN", "0.03"),
                ) or "0.03"
            )
            if sim >= trusted_min_sim and margin >= trusted_min_margin:
                seg["foreground_role_guard_bypassed"] = "trusted_segment_voiceprint"
                return False
        overlap = seg.get("role_overlap")
        if not isinstance(overlap, dict):
            return False
        try:
            agent_s = float(overlap.get("agent", 0.0) or 0.0)
            customer_s = float(overlap.get("customer", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        dur = max(float(seconds), 0.001)
        customer_ratio = customer_s / dur
        agent_ratio = agent_s / dur
        min_customer_ratio = float(os.getenv("SST_FOREGROUND_CUSTOMER_MIN_RATIO", "0.80") or "0.80")
        min_customer_s = float(os.getenv("SST_FOREGROUND_CUSTOMER_MIN_SECONDS", "0.35") or "0.35")
        max_agent_ratio = float(os.getenv("SST_FOREGROUND_AGENT_MAX_RATIO", "0.06") or "0.06")
        max_agent_s = float(os.getenv("SST_FOREGROUND_AGENT_MAX_SECONDS", "0.05") or "0.05")
        return (
            customer_s >= min(min_customer_s, max(dur * 0.50, 0.05))
            and customer_ratio >= min_customer_ratio
            and agent_s <= max_agent_s
            and agent_ratio <= max_agent_ratio
        )

    def _repair_agent_roles_from_segment_voiceprints(
        text_segments: list,
        agent_name: str | None,
        agent_slug: str | None,
        audio_file: str,
    ) -> dict:
        """Assign each segment role from its own audio embedding.

        This is intentionally voice-only. The transcript text is never scored;
        the text segment only supplies the start/end window to embed.
        """
        enabled = os.getenv("SST_AGENT_SEGMENT_VOICE_REPAIR", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return {"enabled": False, "reason": "disabled", "evaluated_segments": 0}
        if not agent_slug or not agent_name or agent_name in {"None", "Unknown Agent"}:
            return {"enabled": True, "reason": "no_identified_agent", "evaluated_segments": 0}
        if not audio_file or not os.path.isfile(audio_file):
            return {"enabled": True, "reason": "missing_audio", "evaluated_segments": 0}

        try:
            import numpy as _np
            import soundfile as _sf
            import torch as _torch
            import torchaudio.functional as _F_ta
            from src.diar_multi import _load_voiceprints
            from src.embedding_campp import get_model, l2_norm
            from src.voiceprints import resolve_voiceprint_path
        except Exception as exc:
            return {
                "enabled": True,
                "reason": f"dependency_error:{type(exc).__name__}",
                "evaluated_segments": 0,
            }

        voiceprints = _load_voiceprints()
        target = voiceprints.get(agent_slug)
        if not target:
            return {"enabled": True, "reason": "target_voiceprint_missing", "evaluated_segments": 0}
        target_name, target_stack = target
        if target_stack.ndim != 2 or not len(target_stack):
            return {"enabled": True, "reason": "target_voiceprint_invalid", "evaluated_segments": 0}

        segment_role_target_stack = None
        segment_role_source = "all_target_voiceprints"
        segment_role_min_sims: list[float] = []
        segment_role_min_margins: list[float] = []
        try:
            agents_index = Path(__file__).parent / "data" / "agent_voiceprints" / "agents.json"
            agents_data = json.loads(agents_index.read_text(encoding="utf-8"))
            agent_info = agents_data.get(agent_slug) or {}
            marked = []
            for entry in agent_info.get("voiceprints") or []:
                if not isinstance(entry, dict):
                    continue
                if not _uses_segment_role_voiceprint(entry):
                    continue
                raw_path = entry.get("path") or entry.get("voiceprint_path")
                vp_path = resolve_voiceprint_path(raw_path, str(agents_index))
                if not vp_path or not os.path.isfile(vp_path):
                    continue
                vp = _np.load(vp_path).astype(_np.float32).squeeze()
                if vp.ndim != 1 or vp.shape[0] != target_stack.shape[1]:
                    continue
                marked.append(l2_norm(vp))
                if entry.get("segment_role_min_similarity") is not None:
                    segment_role_min_sims.append(float(entry.get("segment_role_min_similarity")))
                if entry.get("segment_role_min_margin") is not None:
                    segment_role_min_margins.append(float(entry.get("segment_role_min_margin")))
            if marked:
                segment_role_target_stack = _np.stack(marked).astype(_np.float32)
                segment_role_source = "segment_role_voiceprints"
        except Exception:
            segment_role_target_stack = None
        if segment_role_target_stack is not None:
            target_stack = segment_role_target_stack

        try:
            audio, sr = _sf.read(audio_file, dtype="float32", always_2d=True)
        except Exception as exc:
            return {
                "enabled": True,
                "reason": f"audio_read_error:{type(exc).__name__}",
                "evaluated_segments": 0,
            }
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        if sr != 16000:
            audio = _F_ta.resample(_torch.from_numpy(audio.astype(_np.float32)), sr, 16000).numpy()
            sr = 16000

        target_dim = int(target_stack.shape[1])
        other_voiceprints = [
            (slug, stack)
            for slug, (_name, stack) in voiceprints.items()
            if slug != agent_slug and getattr(stack, "ndim", 0) == 2 and stack.shape[1] == target_dim and len(stack)
        ]
        if not other_voiceprints:
            target_only = True
        else:
            target_only = False

        min_sim = float(os.getenv("SST_AGENT_SEGMENT_VOICE_MIN_SIM", "0.40") or "0.40")
        min_margin = float(os.getenv("SST_AGENT_SEGMENT_VOICE_MIN_MARGIN", "0.03") or "0.03")
        if segment_role_target_stack is not None and not os.getenv("SST_AGENT_SEGMENT_VOICE_MIN_SIM"):
            if segment_role_min_sims:
                min_sim = min(segment_role_min_sims)
        if segment_role_target_stack is not None and not os.getenv("SST_AGENT_SEGMENT_VOICE_MIN_MARGIN"):
            if segment_role_min_margins:
                min_margin = min(segment_role_min_margins)
        target_only_min_sim = float(os.getenv("SST_AGENT_SEGMENT_VOICE_TARGET_ONLY_MIN_SIM", "0.55") or "0.55")
        min_seconds = float(os.getenv("SST_AGENT_SEGMENT_VOICE_MIN_SECONDS", "0.30") or "0.30")
        pad_seconds = float(os.getenv("SST_AGENT_SEGMENT_VOICE_PAD_SECONDS", "0.12") or "0.12")
        smooth_short_s = float(os.getenv("SST_AGENT_SEGMENT_VOICE_SMOOTH_SHORT_SECONDS", "0.80") or "0.80")

        embedder = get_model(force_cpu=True)
        evaluated = changed_to_agent = changed_to_customer = kept_agent = kept_customer = skipped = 0
        foreground_guard_blocks = 0
        score_rows: list[dict] = []
        sample_changes: list[dict] = []
        display_name = target_name or agent_name
        pad = int(max(pad_seconds, 0.0) * sr)

        def _set_agent(seg: dict, reason: str) -> None:
            seg["identified_speaker"] = "AGENT"
            seg["display_speaker"] = display_name
            seg["agent_name"] = display_name
            seg["agent_slug"] = agent_slug
            seg["role_voiceprint_repair"] = reason

        def _set_customer(seg: dict, reason: str) -> None:
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = "Customer"
            seg["role_voiceprint_repair"] = reason
            seg.pop("agent_name", None)
            seg.pop("agent_slug", None)

        for idx, seg in enumerate(text_segments):
            if seg.get("speech_only"):
                skipped += 1
                continue
            start_s = float(seg.get("start", 0.0) or 0.0)
            end_s = float(seg.get("end", 0.0) or 0.0)
            seconds = max(end_s - start_s, 0.0)
            if seconds < min_seconds:
                skipped += 1
                continue
            start = max(0, int(start_s * sr) - pad)
            end = min(len(audio), int(end_s * sr) + pad)
            if end <= start:
                skipped += 1
                continue
            emb = embedder.embed_chunk(audio[start:end].astype(_np.float32), sr=sr)
            if emb is None or getattr(emb, "shape", (0,))[0] != target_dim:
                skipped += 1
                continue
            emb = l2_norm(_np.asarray(emb, dtype=_np.float32))
            target_sim = float(_np.max(target_stack @ emb))
            best_other_slug = None
            best_other_sim = 0.0
            for slug, stack in other_voiceprints:
                sim = float(_np.max(stack @ emb))
                if sim > best_other_sim:
                    best_other_slug = slug
                    best_other_sim = sim
            margin = target_sim - best_other_sim
            is_agent = (
                target_sim >= target_only_min_sim
                if target_only
                else target_sim >= min_sim and margin >= min_margin
            )
            if is_agent and _foreground_customer_blocks_agent(
                seg,
                seconds,
                target_sim,
                margin,
                segment_role_source == "segment_role_voiceprints",
            ):
                is_agent = False
                foreground_guard_blocks += 1
                seg["foreground_role_guard"] = "customer_diarization_blocks_background_agent"

            previous = seg.get("identified_speaker")
            if is_agent:
                _set_agent(seg, "target_segment_voiceprint")
                if previous != "AGENT":
                    changed_to_agent += 1
                else:
                    kept_agent += 1
            else:
                _set_customer(seg, "target_segment_voiceprint_miss")
                if previous == "AGENT":
                    changed_to_customer += 1
                else:
                    kept_customer += 1

            seg["_segment_voice_target_similarity"] = round(target_sim, 3)
            seg["_segment_voice_margin"] = round(margin, 3)
            seg["_segment_voice_best_other_slug"] = best_other_slug
            evaluated += 1
            row = {
                "idx": idx,
                "segment": seg,
                "seconds": seconds,
                "is_agent": is_agent,
                "target_similarity": target_sim,
                "margin": margin,
            }
            score_rows.append(row)
            if previous != seg.get("identified_speaker") and len(sample_changes) < 50:
                sample_changes.append({
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "from": previous,
                    "to": seg.get("identified_speaker"),
                    "target_similarity": round(target_sim, 3),
                    "margin": round(margin, 3),
                    "best_other_slug": best_other_slug,
                    "text": str(seg.get("text") or "")[:120],
                })

        # Short embedded segments can be noisy. Smooth only from neighbouring
        # voiceprint decisions, never from transcript words.
        smoothed = 0
        idx_to_row = {row["idx"]: row for row in score_rows}
        for row in score_rows:
            if row["seconds"] >= smooth_short_s:
                continue
            idx = row["idx"]
            votes = {"AGENT": 0, "CUSTOMER": 0}
            for near in (idx - 2, idx - 1, idx + 1, idx + 2):
                other = idx_to_row.get(near)
                if not other or other["seconds"] < smooth_short_s:
                    continue
                votes["AGENT" if other["is_agent"] else "CUSTOMER"] += 1
            if votes["AGENT"] == votes["CUSTOMER"]:
                continue
            voted_agent = votes["AGENT"] > votes["CUSTOMER"]
            if voted_agent == row["is_agent"]:
                continue
            # Neighbour smoothing is only a tie-breaker for weak short-window
            # embeddings. Do not let neighbours override a clear direct
            # voiceprint hit, and do not promote a segment whose own audio does
            # not pass a slightly stronger target gate.
            if voted_agent:
                if (
                    row["target_similarity"] < (min_sim + 0.05)
                    or row["margin"] < (min_margin + 0.05)
                ):
                    continue
            else:
                if (
                    row["target_similarity"] >= (min_sim + 0.15)
                    and row["margin"] >= (min_margin + 0.10)
                ):
                    continue
            seg = row["segment"]
            if voted_agent:
                _set_agent(seg, "target_segment_voiceprint_neighbor")
            else:
                _set_customer(seg, "target_segment_voiceprint_neighbor")
            row["is_agent"] = voted_agent
            smoothed += 1

        agent_segments = sum(1 for row in score_rows if row["is_agent"])
        customer_segments = len(score_rows) - agent_segments
        return {
            "enabled": True,
            "mode": "target_segment_voiceprint",
            "target_agent_slug": agent_slug,
            "evaluated_segments": evaluated,
            "agent_segments": agent_segments,
            "customer_segments": customer_segments,
            "changed_to_agent": changed_to_agent,
            "changed_to_customer": changed_to_customer,
            "kept_agent": kept_agent,
            "kept_customer": kept_customer,
            "skipped_segments": skipped,
            "foreground_guard_blocks": foreground_guard_blocks,
            "smoothed_segments": smoothed,
            "min_similarity": min_sim,
            "min_margin": min_margin,
            "target_only": target_only,
            "target_voiceprint_source": segment_role_source,
            "target_only_min_similarity": target_only_min_sim,
            "min_seconds": min_seconds,
            "pad_seconds": pad_seconds,
            "sample_changes": sample_changes,
        }

    def _repair_agent_roles_from_target_speaker_vad(
        text_segments: list,
        agent_name: str | None,
        agent_slug: str | None,
        audio_file: str,
    ) -> dict:
        """Experimental target-speaker VAD role engine.

        This remains voiceprint-only. The transcript text is never inspected;
        segment timestamps are used only to average target-speaker VAD windows.
        """
        role_engine = (os.getenv("SST_ROLE_ENGINE", "tsvad").strip().lower()
                       or "tsvad")
        env_enabled = os.getenv("SST_TARGET_SPEAKER_VAD", "").strip().lower()
        enabled = role_engine in {"tsvad", "target_speaker_vad", "sortformer_campp_tsvad"} or env_enabled in {
            "1", "true", "yes", "on",
        }
        if not enabled:
            return {"enabled": False, "reason": "disabled", "role_engine": role_engine}
        if not agent_slug:
            return {"enabled": True, "reason": "no_target_agent_slug", "role_engine": role_engine}
        if not audio_file or not os.path.isfile(audio_file):
            return {"enabled": True, "reason": "missing_audio", "role_engine": role_engine}

        try:
            import numpy as _np
            from src.diar_multi import _load_voiceprints
            from src.embedding_campp import l2_norm
            from src.target_speaker_vad import TargetSpeakerVAD
            from src.voiceprints import resolve_voiceprint_path
        except Exception as exc:
            return {
                "enabled": True,
                "reason": f"dependency_error:{type(exc).__name__}",
                "role_engine": role_engine,
            }

        voiceprints = _load_voiceprints()
        target = voiceprints.get(agent_slug)
        if not target:
            return {"enabled": True, "reason": "target_voiceprint_missing", "role_engine": role_engine}
        target_name, target_stack = target
        if getattr(target_stack, "ndim", 0) != 2 or not len(target_stack):
            return {"enabled": True, "reason": "target_voiceprint_invalid", "role_engine": role_engine}

        target_stack = _np.asarray(target_stack, dtype=_np.float32)
        target_dim = int(target_stack.shape[1])
        source = "all_target_voiceprints"
        threshold_hints: list[float] = []
        margin_hints: list[float] = []
        use_marked = os.getenv("SST_TSVAD_USE_SEGMENT_ROLE_VOICEPRINTS", "1").strip().lower()
        if use_marked not in {"0", "false", "no", "off"}:
            try:
                agents_index = Path(__file__).parent / "data" / "agent_voiceprints" / "agents.json"
                agents_data = json.loads(agents_index.read_text(encoding="utf-8"))
                agent_info = agents_data.get(agent_slug) or {}
                marked = []
                for entry in agent_info.get("voiceprints") or []:
                    if not isinstance(entry, dict):
                        continue
                    if not _uses_segment_role_voiceprint(entry):
                        continue
                    raw_path = entry.get("path") or entry.get("voiceprint_path")
                    vp_path = resolve_voiceprint_path(raw_path, str(agents_index))
                    if not vp_path or not os.path.isfile(vp_path):
                        continue
                    vp = _np.load(vp_path).astype(_np.float32).squeeze()
                    if vp.ndim != 1 or vp.shape[0] != target_dim:
                        continue
                    marked.append(l2_norm(vp))
                    if entry.get("segment_role_min_similarity") is not None:
                        threshold_hints.append(float(entry.get("segment_role_min_similarity")))
                    if entry.get("segment_role_min_margin") is not None:
                        margin_hints.append(float(entry.get("segment_role_min_margin")))
                if marked:
                    target_stack = _np.stack(marked).astype(_np.float32)
                    source = "segment_role_voiceprints"
            except Exception:
                pass

        other_stacks = [
            _np.asarray(stack, dtype=_np.float32)
            for slug, (_name, stack) in voiceprints.items()
            if slug != agent_slug
            and getattr(stack, "ndim", 0) == 2
            and stack.shape[1] == target_dim
            and len(stack)
        ]
        background_stack = _np.concatenate(other_stacks, axis=0) if other_stacks else None
        target_only = background_stack is None

        min_sim = float(os.getenv("SST_TSVAD_MIN_SIM", "0.33") or "0.33")
        min_margin = float(os.getenv("SST_TSVAD_MIN_MARGIN", "0.03") or "0.03")
        if source == "segment_role_voiceprints" and not os.getenv("SST_TSVAD_MIN_SIM") and threshold_hints:
            min_sim = min(threshold_hints)
        if source == "segment_role_voiceprints" and not os.getenv("SST_TSVAD_MIN_MARGIN") and margin_hints:
            min_margin = min(margin_hints)
        window_s = float(os.getenv("SST_TSVAD_WINDOW_SECONDS", "1.50") or "1.50")
        stride_s = float(os.getenv("SST_TSVAD_STRIDE_SECONDS", "1.00") or "1.00")
        min_overlap_ratio = float(os.getenv("SST_TSVAD_MIN_OVERLAP_RATIO", "0.25") or "0.25")
        min_overlap_seconds = float(os.getenv("SST_TSVAD_MIN_OVERLAP_SECONDS", "0.20") or "0.20")
        decision_mode = (os.getenv("SST_TSVAD_DECISION_MODE", "tsvad").strip().lower() or "tsvad")
        blend_segment_weight = float(os.getenv("SST_TSVAD_BLEND_SEGMENT_WEIGHT", "0.50") or "0.50")
        blend_segment_weight = max(0.0, min(blend_segment_weight, 1.0))
        blend_min_sim = float(os.getenv("SST_TSVAD_BLEND_MIN_SIM", "0.42") or "0.42")
        blend_min_margin = float(os.getenv("SST_TSVAD_BLEND_MIN_MARGIN", "-0.05") or "-0.05")
        keep_strong_segment = os.getenv("SST_TSVAD_KEEP_STRONG_SEGMENT_AGENT", "0").strip().lower()
        keep_strong_segment = keep_strong_segment not in {"0", "false", "no", "off"}
        strong_segment_min_sim = float(os.getenv("SST_TSVAD_STRONG_SEGMENT_MIN_SIM", "0.42") or "0.42")
        strong_segment_min_margin = float(os.getenv("SST_TSVAD_STRONG_SEGMENT_MIN_MARGIN", "0.03") or "0.03")

        try:
            tsvad = TargetSpeakerVAD(
                target_stack,
                threshold=min_sim,
                window_s=window_s,
                stride_s=stride_s,
                background_voiceprints=background_stack,
                margin=0.0 if target_only else min_margin,
            )
            windows = tsvad.detect(audio_file)
        except Exception as exc:
            return {
                "enabled": True,
                "reason": f"runtime_error:{type(exc).__name__}:{str(exc)[:160]}",
                "role_engine": role_engine,
                "target_agent_slug": agent_slug,
            }
        if not windows:
            return {
                "enabled": True,
                "reason": "no_windows",
                "role_engine": role_engine,
                "target_agent_slug": agent_slug,
            }

        display_name = target_name or agent_name or "Agent"
        evaluated = skipped = changed_to_agent = changed_to_customer = kept_agent = kept_customer = 0
        foreground_guard_blocks = 0
        strong_segment_kept_agent = 0
        agent_segments = customer_segments = 0
        sample_changes: list[dict] = []

        def _set_agent(seg: dict) -> None:
            seg["identified_speaker"] = "AGENT"
            seg["display_speaker"] = display_name
            seg["agent_name"] = display_name
            seg["agent_slug"] = agent_slug
            seg["role_voiceprint_repair"] = "target_speaker_vad"

        def _set_customer(seg: dict) -> None:
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = "Customer"
            seg["role_voiceprint_repair"] = "target_speaker_vad_miss"
            seg.pop("agent_name", None)
            seg.pop("agent_slug", None)

        for seg in text_segments:
            if seg.get("speech_only"):
                skipped += 1
                continue
            start_s = float(seg.get("start", 0.0) or 0.0)
            end_s = float(seg.get("end", 0.0) or 0.0)
            seconds = max(end_s - start_s, 0.001)
            overlapping = [
                w for w in windows
                if float(w["end"]) > start_s and float(w["start"]) < end_s
            ]
            if not overlapping:
                skipped += 1
                continue
            weights = [
                max(0.0, min(float(w["end"]), end_s) - max(float(w["start"]), start_s))
                for w in overlapping
            ]
            overlap_seconds = sum(weights)
            if overlap_seconds < min_overlap_seconds or (overlap_seconds / seconds) < min_overlap_ratio:
                skipped += 1
                continue
            target_score = sum(float(w.get("target_cosine", w.get("cosine", 0.0))) * wt for w, wt in zip(overlapping, weights)) / overlap_seconds
            best_other = sum(float(w.get("best_other_cosine", 0.0)) * wt for w, wt in zip(overlapping, weights)) / overlap_seconds
            score_margin = target_score - best_other
            decision_score = target_score
            decision_margin = score_margin
            decision_mode_used = decision_mode
            if decision_mode == "blend":
                try:
                    segment_target = float(seg.get("_segment_voice_target_similarity"))
                    segment_margin = float(seg.get("_segment_voice_margin"))
                    decision_score = (
                        blend_segment_weight * segment_target
                        + (1.0 - blend_segment_weight) * target_score
                    )
                    decision_margin = (
                        blend_segment_weight * segment_margin
                        + (1.0 - blend_segment_weight) * score_margin
                    )
                    is_agent = decision_score >= blend_min_sim and (
                        target_only or decision_margin >= blend_min_margin
                    )
                except (TypeError, ValueError):
                    is_agent = target_score >= min_sim and (target_only or score_margin >= min_margin)
                    decision_mode_used = "tsvad_fallback"
            else:
                is_agent = target_score >= min_sim and (target_only or score_margin >= min_margin)
            previous = seg.get("identified_speaker")
            if not is_agent and keep_strong_segment and previous == "AGENT":
                try:
                    segment_target = float(seg.get("_segment_voice_target_similarity"))
                    segment_margin = float(seg.get("_segment_voice_margin"))
                except (TypeError, ValueError):
                    segment_target = segment_margin = 0.0
                if segment_target >= strong_segment_min_sim and segment_margin >= strong_segment_min_margin:
                    is_agent = True
                    strong_segment_kept_agent += 1
                    seg["tsvad_segment_voice_override"] = "strong_segment_voiceprint_kept_agent"
            if is_agent and _foreground_customer_blocks_agent(
                seg,
                seconds,
                decision_score,
                decision_margin,
                source == "segment_role_voiceprints",
            ):
                is_agent = False
                foreground_guard_blocks += 1
                seg["foreground_role_guard"] = "customer_diarization_blocks_background_agent"

            if is_agent:
                _set_agent(seg)
                agent_segments += 1
                if previous != "AGENT":
                    changed_to_agent += 1
                else:
                    kept_agent += 1
            else:
                _set_customer(seg)
                customer_segments += 1
                if previous == "AGENT":
                    changed_to_customer += 1
                else:
                    kept_customer += 1

            seg["_tsvad_target_similarity"] = round(float(target_score), 3)
            seg["_tsvad_best_other_similarity"] = round(float(best_other), 3)
            seg["_tsvad_margin"] = round(float(score_margin), 3)
            seg["_tsvad_decision_score"] = round(float(decision_score), 3)
            seg["_tsvad_decision_margin"] = round(float(decision_margin), 3)
            seg["_tsvad_decision_mode"] = decision_mode_used
            seg["_tsvad_overlap_ratio"] = round(float(overlap_seconds / seconds), 3)
            evaluated += 1
            if previous != seg.get("identified_speaker") and len(sample_changes) < 50:
                sample_changes.append({
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "from": previous,
                    "to": seg.get("identified_speaker"),
                    "target_similarity": round(float(target_score), 3),
                    "best_other_similarity": round(float(best_other), 3),
                    "margin": round(float(score_margin), 3),
                    "decision_score": round(float(decision_score), 3),
                    "decision_margin": round(float(decision_margin), 3),
                    "text": str(seg.get("text") or "")[:120],
                })

        return {
            "enabled": True,
            "mode": "target_speaker_vad",
            "role_engine": role_engine,
            "target_agent_slug": agent_slug,
            "target_voiceprint_source": source,
            "target_only": target_only,
            "decision_mode": decision_mode,
            "blend_segment_weight": blend_segment_weight,
            "blend_min_similarity": blend_min_sim,
            "blend_min_margin": blend_min_margin,
            "window_count": len(windows),
            "window_seconds": window_s,
            "stride_seconds": stride_s,
            "evaluated_segments": evaluated,
            "skipped_segments": skipped,
            "foreground_guard_blocks": foreground_guard_blocks,
            "strong_segment_kept_agent": strong_segment_kept_agent,
            "agent_segments": agent_segments,
            "customer_segments": customer_segments,
            "changed_to_agent": changed_to_agent,
            "changed_to_customer": changed_to_customer,
            "kept_agent": kept_agent,
            "kept_customer": kept_customer,
            "min_similarity": min_sim,
            "min_margin": 0.0 if target_only else min_margin,
            "min_overlap_ratio": min_overlap_ratio,
            "min_overlap_seconds": min_overlap_seconds,
            "keep_strong_segment_agent": keep_strong_segment,
            "strong_segment_min_similarity": strong_segment_min_sim,
            "strong_segment_min_margin": strong_segment_min_margin,
            "sample_changes": sample_changes,
        }

    def _demote_agent_roles_from_customer_text_cues(text_segments: list) -> dict:
        """Disabled: role identity must come from voiceprint matching only."""
        return {
            "enabled": False,
            "demoted_segments": 0,
            "segments": [],
            "reason": "voiceprint_only",
        }

    # Free unused memory before speaker identification. diar_multi can use CUDA
    # for supported embedding backends; CAM++ remains CPU inside WeSpeaker.
    try:
        import torch as _torch, gc as _gc
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
    except Exception:
        pass

    _set_status(3, "Transcription", "Identifying speakers (speaker-first voice matching)...")
    try:
        from src.diar_clean import diarize_clean
        max_speakers = int(os.getenv("SST_MAX_SPEAKERS", "4") or "4")
        backend = os.getenv("SST_SPEAKER_DIAR_BACKEND", "sortformer").strip() or "sortformer"
        streaming_env = os.getenv("SST_SORTFORMER_STREAMING", "").strip().lower()
        streaming_min_s = float(os.getenv("SST_SORTFORMER_STREAMING_MIN_SECONDS", "600") or "600")
        if streaming_env in {"1", "true", "yes", "on"}:
            sortformer_streaming = True
        elif streaming_env in {"0", "false", "no", "off"}:
            sortformer_streaming = False
        else:
            sortformer_streaming = backend == "sortformer" and dur_s >= streaming_min_s
        env_target_slug = os.getenv("SST_TARGET_AGENT_SLUG", "").strip() or None
        if env_target_slug:
            target_agent_slug = _resolve_target_agent_slug(env_target_slug)
            presence_floor = _target_presence_floor(target_agent_slug)
        print(
            f"[UI] Running speaker-first diarization ({backend}, max_speakers={max_speakers}, "
            f"streaming={sortformer_streaming}, target={target_agent_slug or 'auto'}, "
            f"floor={presence_floor:.2f})...",
            flush=True,
        )
        def _run_diarization(use_streaming: bool):
            return diarize_clean(
                audio_path=diar_wav,
                transcribed_segments=segments,
                backend=backend,
                max_speakers=max_speakers,
                sortformer_streaming=use_streaming,
                target_agent_slug=target_agent_slug,
                presence_floor=presence_floor,
                hf_token=os.getenv("HF_TOKEN"),
            )

        try:
            diar_result = _run_diarization(sortformer_streaming)
        except Exception:
            if backend != "sortformer" or sortformer_streaming:
                raise
            print("[UI] Full Sortformer failed; retrying with streaming Sortformer...", flush=True)
            try:
                import torch as _torch, gc as _gc
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _torch.cuda.synchronize()
            except Exception:
                pass
            sortformer_streaming = True
            diar_result = _run_diarization(True)
        segments     = diar_result["segments"]
        show_untranscribed = (
            os.getenv("SST_SHOW_UNTRANSCRIBED_SPEAKERS", "0").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        segments, transcript_coverage = _add_untranscribed_speaker_segments(
            segments,
            diar_result.get("speaker_segments", []),
            add_rows=show_untranscribed,
        )
        speech_only_added = int(transcript_coverage.get("added_count") or 0)
        hidden_speech_only = int(transcript_coverage.get("hidden_count") or 0)
        if speech_only_added:
            print(
                f"[UI] Added {speech_only_added} speech-only speaker rows "
                f"({transcript_coverage.get('added_seconds', 0)}s without ASR text)",
                flush=True,
            )
        elif hidden_speech_only:
            print(
                f"[UI] Hidden {hidden_speech_only} untranscribed speaker gaps "
                f"({transcript_coverage.get('hidden_seconds', 0)}s without ASR text)",
                flush=True,
            )
        agent_name_id = diar_result.get("agent_name", "Unknown Agent")
        agent_sim     = diar_result.get("agent_similarity", 0.0)
        backend_dim   = diar_result.get("matched_backend_dim")
        print(
            f"[UI] Agent identified: {agent_name_id} "
            f"(cosine={agent_sim:.3f}, dim={backend_dim})",
            flush=True,
        )
        print(
            f"[UI] Speaker mode: {diar_result.get('speaker_mode', 'unknown')} "
            f"speaker_count={diar_result.get('speaker_count', 'n/a')} "
            f"agent_speaker_id={diar_result.get('agent_speaker_id', 'n/a')}",
            flush=True,
        )
        print(f"[UI] Speakers: {list(diar_result.get('per_speaker', {}).keys())}", flush=True)

        segments, speaker_turn_text_split = _split_text_segments_on_speaker_turns(
            segments,
            diar_result.get("speaker_segments", []),
        )
        if speaker_turn_text_split.get("added_segments"):
            print(
                f"[UI] Speaker-turn text split: "
                f"{speaker_turn_text_split['original_segments']} -> "
                f"{speaker_turn_text_split['final_segments']} segment(s) "
                f"(+{speaker_turn_text_split['added_segments']})",
                flush=True,
            )

        unknown_segments_smoothed = _smooth_short_unknown_segments(segments)
        if unknown_segments_smoothed:
            print(f"[UI] Smoothed {unknown_segments_smoothed} short unknown speaker segments", flush=True)
        unknown_segments_customer_fallback = _fallback_unknown_text_to_customer(segments)
        if unknown_segments_customer_fallback:
            print(
                f"[UI] Marked {unknown_segments_customer_fallback} unmatched text snippets as customer",
                flush=True,
            )
        cluster_match_table = (
            diar_result.get("cluster_report")
            or diar_result.get("cluster_match_table")
            or {}
        )
        agent_role_voiceprint_repair = _repair_agent_roles_from_voiceprint_clusters(
            segments,
            agent_name_id,
            diar_result.get("agent_slug"),
            cluster_match_table,
        )
        if agent_role_voiceprint_repair.get("promoted_segments"):
            print(
                f"[UI] Voiceprint role repair promoted "
                f"{agent_role_voiceprint_repair['promoted_segments']} segment(s) across "
                f"{len(agent_role_voiceprint_repair.get('promoted_speakers') or [])} same-agent speaker(s)",
                flush=True,
            )
        agent_role_segment_voiceprint_repair = _repair_agent_roles_from_segment_voiceprints(
            segments,
            agent_name_id,
            diar_result.get("agent_slug"),
            diar_wav,
        )
        if agent_role_segment_voiceprint_repair.get("evaluated_segments"):
            print(
                f"[UI] Segment voiceprint role assignment evaluated "
                f"{agent_role_segment_voiceprint_repair['evaluated_segments']} segment(s): "
                f"agent={agent_role_segment_voiceprint_repair.get('agent_segments', 0)} "
                f"customer={agent_role_segment_voiceprint_repair.get('customer_segments', 0)} "
                f"changed_to_agent={agent_role_segment_voiceprint_repair.get('changed_to_agent', 0)} "
                f"changed_to_customer={agent_role_segment_voiceprint_repair.get('changed_to_customer', 0)}",
                flush=True,
            )
        role_agent_slug = diar_result.get("agent_slug") or target_agent_slug
        target_speaker_vad_role_refinement = _repair_agent_roles_from_target_speaker_vad(
            segments,
            agent_name_id,
            role_agent_slug,
            diar_wav,
        )
        if target_speaker_vad_role_refinement.get("evaluated_segments"):
            print(
                f"[UI] Target-speaker VAD role engine evaluated "
                f"{target_speaker_vad_role_refinement['evaluated_segments']} segment(s): "
                f"agent={target_speaker_vad_role_refinement.get('agent_segments', 0)} "
                f"customer={target_speaker_vad_role_refinement.get('customer_segments', 0)} "
                f"changed_to_agent={target_speaker_vad_role_refinement.get('changed_to_agent', 0)} "
                f"changed_to_customer={target_speaker_vad_role_refinement.get('changed_to_customer', 0)}",
                flush=True,
            )
        agent_role_text_repair = {
            "enabled": False,
            "promoted_speakers": [],
            "promoted_segments": 0,
            "reason": "voiceprint_only",
        }
        customer_role_text_repair = {
            "enabled": False,
            "demoted_segments": 0,
            "segments": [],
            "reason": "voiceprint_only",
        }

        for seg in segments:
            dur = float(seg["end"]) - float(seg["start"])
            if seg.get("identified_speaker") == "AGENT":
                agent_time  += dur;  agent_turns  += 1
            else:
                customer_time += dur; customer_turns += 1

        diarization_applied = True
    except Exception as _diar_err:
        print(f"[UI] Speaker-first diarization failed ({repr(_diar_err)})", flush=True)
        import traceback; traceback.print_exc()
        for seg in segments:
            seg["speaker"] = seg.get("speaker") or "SPEAKER_99"
            seg["identified_speaker"] = "CUSTOMER"
            seg.pop("agent_name", None)
            seg["display_speaker"] = "Unknown"
    _check_cancelled()

    elapsed = round(time.time() - t0, 2)

    if transcriber is not None:
        transcriber.unload()

    # Force-free GPU/CPU memory
    try:
        import torch, gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # Speaker stats (used by both paths)
    speaker_stats = _recompute_role_stats(segments)
    agent_time = float(speaker_stats["AGENT"]["time_s"])
    customer_time = float(speaker_stats["CUSTOMER"]["time_s"])
    agent_turns = int(speaker_stats["AGENT"]["turns"])
    customer_turns = int(speaker_stats["CUSTOMER"]["turns"])

    if diarization_applied:
        print(
            f"[UI] Speaker ID done — agent={agent_time:.0f}s/{agent_turns}t  "
            f"customer={customer_time:.0f}s/{customer_turns}t",
            flush=True,
        )

    # Build transcription_json in the requested format:
    #   [{start: "HH:MM:SS.mmm", end: "...", speaker: "SPEAKER_XX"|"UNKNOWN",
    #     phrase: "...", avg_score: float|null}, ...]
    def _fmt_ts(s: float) -> str:
        h = int(s // 3600); m = int((s % 3600) // 60)
        sec = s - (h*3600 + m*60)
        return f"{h:02d}:{m:02d}:{sec:06.3f}"

    identified_agent_name = next(
        (s.get("agent_name") for s in segments if s.get("agent_name")),
        locals().get("agent_name_id", "Unknown Agent"),
    )

    transcription_json = []
    for seg in segments:
        conf = seg.get("confidence", 0.0)
        identified_speaker = seg.get("identified_speaker", seg.get("speaker", "UNKNOWN"))
        display_speaker = seg.get("display_speaker")
        if not display_speaker:
            if identified_speaker == "AGENT":
                display_speaker = seg.get("agent_name") or identified_agent_name
            elif identified_speaker == "CUSTOMER":
                display_speaker = "Customer"
            else:
                display_speaker = identified_speaker
        transcription_json.append({
            "start":     _fmt_ts(float(seg["start"])),
            "end":       _fmt_ts(float(seg["end"])),
            "speaker":   seg.get("speaker", "UNKNOWN"),
            "phrase":    seg.get("text", "").strip(),
            "avg_score": round(float(conf), 3) if conf else None,
            "identified_speaker": identified_speaker,
            "display_speaker": display_speaker,
            "agent_name": seg.get("agent_name"),
        })

    # ── Trim audio to speech-only regions ────────────────────────────────────
    _set_status(3, "Transcription", "Trimming audio to speech regions...")
    trimmed_path = os.path.join(out_dir, "trimmed_audio.mp3")
    # pad_s=1.0: keep 1 s around each block so word edges aren't clipped
    # merge_gap_s=5.0: join blocks separated by ≤5 s — avoids many tiny cuts
    #   in noisy recordings where VAD fires in short bursts
    trim_ok = _trim_to_speech(segment_audio_path, segments, trimmed_path,
                               pad_s=1.0, merge_gap_s=5.0)
    trimmed_audio_file = trimmed_path.replace("\\", "/") if trim_ok else None
    if not trim_ok:
        print("[UI] Trim skipped — using original enhanced audio.", flush=True)

    # Collect identified agent name (set on segments during multi-agent ID)
    _identified_agent = identified_agent_name

    result = {
        "audio_file":               segment_audio_path.replace("\\", "/"),
        "source_audio_file":        audio_path.replace("\\", "/"),
        "asr_audio_file":           asr_wav.replace("\\", "/"),
        "asr_audio_mode":           asr_audio_mode,
        "diarization_audio_file":   diar_wav.replace("\\", "/"),
        "trimmed_audio_file":       trimmed_audio_file,
        "model":                    whisper_model,
        "requested_model":          requested_model,
        "fallback_used":            requested_model != whisper_model,
        "transcriber_device":       transcriber_device,
        "transcriber_isolated":     isolated_transcriber,
        "processed_at":             datetime.utcnow().isoformat() + "Z",
        "processing_time_seconds":  elapsed,
        "total_segments":           len(segments),
        "segments":                 segments,
        "transcription_json":       transcription_json,
        "asr_slice_split":          locals().get("asr_slice_split", {}),
        "diarization":              "diar_multi_voiceprint" if diarization_applied else "none",
        "speaker_stats":            speaker_stats,
        "identified_agent":         _identified_agent,
        "identified_agent_slug":    diar_result.get("agent_slug") if diarization_applied else None,
        "speaker_id_backend_dim":   diar_result.get("matched_backend_dim"),
        "voiceprint_dims":          diar_result.get("voiceprint_dims", {}),
        "speaker_id_warning":       (
            diar_result.get("speaker_id_warning") or diar_result.get("warning")
        ),
        "speaker_id_mode":          diar_result.get("speaker_mode"),
        "speaker_id_backend":       diar_result.get("speaker_id_backend") or diar_result.get("backend"),
        "speaker_id_sortformer_streaming": bool(
            diar_result.get("sortformer_streaming", sortformer_streaming)
        ),
        "target_agent_slug":        target_agent_slug,
        "speaker_id_presence_floor": presence_floor,
        "speaker_id_cluster_report": (
            diar_result.get("cluster_report")
            or diar_result.get("cluster_match_table")
            or {}
        ),
        "agent_cluster_decision": diar_result.get("agent_cluster_decision", {}),
        "cluster_durations": diar_result.get("cluster_durations", {}),
        "speaker_boundary_refinement": diar_result.get("boundary_refinement", {}),
        "speech_only_segments_added": speech_only_added,
        "transcript_coverage": transcript_coverage,
        "speaker_turn_text_split": locals().get("speaker_turn_text_split", {}),
        "main_conversation_gate": locals().get("main_conversation_gate", {}),
        "asr_repeat_loop_filter": locals().get("asr_repeat_loop_filter", {}),
        "unknown_segments_smoothed": locals().get("unknown_segments_smoothed", 0),
        "unknown_segments_customer_fallback": locals().get("unknown_segments_customer_fallback", 0),
        "agent_role_voiceprint_repair": locals().get("agent_role_voiceprint_repair", {}),
        "agent_role_segment_voiceprint_repair": locals().get("agent_role_segment_voiceprint_repair", {}),
        "target_speaker_vad_role_refinement": locals().get("target_speaker_vad_role_refinement", {}),
        "role_engine": os.getenv("SST_ROLE_ENGINE", "tsvad").strip().lower() or "tsvad",
        "agent_role_text_repair": locals().get("agent_role_text_repair", {}),
        "customer_role_text_repair": locals().get("customer_role_text_repair", {}),
        "note": (
            f"Requested {requested_model}; transcribed with {whisper_model}"
            if requested_model != whisper_model
            else f"Transcribed with {whisper_model}"
        ),
    }
    _check_cancelled()
    os.makedirs(out_dir, exist_ok=True)  # re-create if deleted during long transcription
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[UI] result.json saved → {result_path} ({len(segments)} segs, {elapsed:.0f}s)")
    return dir_name


def _flush_result(path: str, audio_path: str, segments: list, elapsed: float):
    """Write partial result.json so the UI can show progress mid-transcription."""
    from datetime import datetime
    tmp = {
        "audio_file":              audio_path.replace("\\", "/"),
        "processed_at":            datetime.utcnow().isoformat() + "Z",
        "processing_time_seconds": round(elapsed, 2),
        "total_segments":          len(segments),
        "segments":                segments,
        "note": "Partial — transcription in progress",
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tmp, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Main pipeline thread
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _run_pipeline(
    upload_path: str,
    filename: str,
    whisper_model: str = "large-v3",
    target_agent_slug: str | None = None,
):
    """
    Stage 0a — FFmpeg enhancement  (full audio, used by AI pipeline)
    Stage 0b — noisereduce          (first 5 min)
    Stage 0c — DeepFilterNet3       (first 5 min)
    Stage 0d — SpeechBrain MetricGAN+ (first 5 min)
    Stages 1-3 — diarize → speaker ID → transcribe
    Patch result.json with all enhancement paths.
    """
    pipeline_started_at = time.time()
    paths = _derive_paths(upload_path)
    os.makedirs("data/raw_calls", exist_ok=True)

    enhancement_paths: dict = {}

    with _status_lock:
        _status.update(running=True, done=False, error=None, result_id=None,
                       cancel_requested=False, started_at=pipeline_started_at,
                       stage_started_at=pipeline_started_at,
                       updated_at=pipeline_started_at, completed_at=None,
                       elapsed_seconds=0.0, stage_elapsed_seconds=0.0,
                       processing_time_seconds=None)

    try:
        _check_cancelled()
        # If the file is already an enhanced output, skip re-enhancement to
        # avoid double-processing (which truncates the audio to garbage).
        already_enhanced = os.path.basename(upload_path).startswith("enhanced_")

        if already_enhanced:
            _set_status(0, "Enhancing Audio", "File already enhanced — skipping re-processing...")
            print("[UI] Skipping enhancement: file already starts with 'enhanced_'.")
            paths["ffmpeg"] = upload_path.replace("\\", "/")
            enhancement_paths["ffmpeg"] = upload_path.replace("\\", "/")
            pipeline_audio = paths["ffmpeg"]
        else:
            # ── 0a FFmpeg — LIGHT format normalisation (highpass + loudnorm) ─
            _set_status(0, "Enhancing Audio", "[1/2] FFmpeg · format normalisation...")
            print("[UI] Stage 0a: FFmpeg (light)...")
            try:
                _enhance_ffmpeg(upload_path, paths["ffmpeg"], max_seconds=None)
                enhancement_paths["ffmpeg"] = paths["ffmpeg"]
                print("[UI] FFmpeg done.")
            except Exception as e:
                print(f"[UI] FFmpeg failed: {e}")
                paths["ffmpeg"] = upload_path
                enhancement_paths["ffmpeg"] = upload_path.replace("\\", "/")

            # ── 0b DeepFilterNet3 — neural denoising (full audio) ────────────
            # Skip on already-clean audio (synthetic test files, broadcast-grade
            # recordings) since DFN3 can over-process them into silence.
            # IMPORTANT: DFN3 on Parakeet ASR input DROPS WORDS (the model is
            # tuned for stationary noise and tends to remove quiet speech).
            # That's why SST_PARAKEET_SKIP_DFN defaults to 1 and we no longer
            # let SST_DEEP_ENHANCE override it.
            pipeline_audio = paths["ffmpeg"]   # fallback
            skip_dfn_for_parakeet = (
                whisper_model.startswith("parakeet-tdt-")
                and os.getenv("SST_PARAKEET_SKIP_DFN", "1").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            if skip_dfn_for_parakeet:
                _set_status(0, "Enhancing Audio", "[2/2] Parakeet DS mode - skipping neural denoise")
                print("[UI] Stage 0b: skipped DeepFilterNet3 for Parakeet ASR.")
            elif _is_clean_audio(paths["ffmpeg"]):
                _set_status(0, "Enhancing Audio", "[2/2] Source already clean — skipping DFN3")
                print("[UI] Stage 0b: skipped (audio already clean)")
            else:
                _set_status(0, "Enhancing Audio", "[2/2] DeepFilterNet3 · neural denoising...")
                print("[UI] Stage 0b: DeepFilterNet3...")
                try:
                    _enhance_deepfilternet(paths["ffmpeg"], paths["deepfilter"])
                    enhancement_paths["deepfilter"] = paths["deepfilter"]
                    pipeline_audio = paths["deepfilter"]
                    print("[UI] DeepFilterNet3 done.")
                except ImportError:
                    print("[UI] deepfilternet not installed — using FFmpeg output.")
                except Exception as e:
                    print(f"[UI] DeepFilterNet3 failed: {e} — using FFmpeg output.")

            # ASR continues to use the raw upload for Parakeet — feeding the
            # enhanced (silenceremove'd) audio to ASR would shift timestamps
            # and the silenceremove threshold has been observed dropping real
            # words from quiet speakers. Silence removal therefore only
            # affects the BROWSER PLAYBACK file, not the transcription input.
            prefer_original_parakeet = (
                whisper_model.startswith("parakeet-tdt-")
                and os.getenv("SST_PARAKEET_ORIGINAL_SOURCE", "1").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            if prefer_original_parakeet:
                pipeline_audio = upload_path
                enhancement_paths["asr_source"] = upload_path.replace("\\", "/")
                print("[UI] Parakeet ASR source: original upload audio.")

        playback_source = paths.get("ffmpeg") or upload_path
        if playback_source and os.path.exists(playback_source):
            try:
                _set_status(0, "Enhancing Audio", "Creating louder playback audio...")
                _make_playback_loud_mp3(playback_source, paths["playback_loud"])
                enhancement_paths["playback_loud"] = paths["playback_loud"]
                print("[UI] Loud playback audio ready.")
            except Exception as e:
                print(f"[UI] Loud playback audio failed: {e}")

        _check_cancelled()
        # run_e2e.py (pyannote diarization) only used when enrolled agent
        # embeddings exist. Without enrollment, all models use inline path.
        _WHISPER_MODELS = {
            "whisper-large-v3-turbo", "whisper-large-v3",
            "large-v3-turbo", "large-v3",
            "distil-large-v3", "distil-large-v3.5",
        }
        # Always use the inline path — it has our improved pipeline:
        #   light FFmpeg + DeepFilterNet3 + ECAPA/pyannote hybrid diarization.
        # The run_e2e.py subprocess path used the old over-processing chain
        # and enrolled-agent matching which produces CUSTOMER/AGENT labels
        # that chopped utterances incorrectly ("Hello," + "August.").
        use_inline = True

        if use_inline:
            label = whisper_model if whisper_model in _WHISPER_MODELS else whisper_model
            _set_status(1, "Transcription", f"Transcribing with {label}...")
            print(f"[UI] Inline transcription mode ({label}).")
            result_id = _transcribe_inline(
                pipeline_audio,
                whisper_model,
                original_path=upload_path,
                target_agent_slug=target_agent_slug,
            )
            _check_cancelled()
        else:
            # Full pipeline via run_e2e.py subprocess (Whisper + pyannote diarization)
            _set_status(1, "Speaker Diarization", "Loading pyannote · detecting who speaks when...")
            print("[UI] Stage 1: AI pipeline...")
            cmd = [
                "python", "run_e2e.py",
                "--hf-token", HF_TOKEN,
                "--device", "cuda",
                "--whisper-model", whisper_model,
                "--skip-extraction", "--skip-enrollment",
                "--test-audio", pipeline_audio,
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    print(f"[Pipeline] {line}", flush=True)
                except UnicodeEncodeError:
                    print(f"[Pipeline] {line.encode('ascii', errors='replace').decode('ascii')}", flush=True)

                if "Stage 1" in line or "iarization" in line:
                    _set_status(1, "Speaker Diarization", "Detecting who speaks when...")
                elif "Stage 2" in line or "Speaker Identification" in line or "embedding" in line.lower():
                    _set_status(2, "Speaker Identification", "Matching voices to enrolled agents...")
                elif "Stage 3" in line or "Transcription" in line or "Transcribing" in line:
                    _set_status(3, "Transcription", "Converting speech to text with Whisper...")
                elif "Pipeline complete" in line or "PROCESSING COMPLETE" in line:
                    _set_status(4, "Finalizing", "Saving results...")

            proc.wait()
            if proc.returncode != 0:
                with _status_lock:
                    _status.update(error="Pipeline failed. Check terminal.", running=False)
                return

            result_id   = os.path.splitext(os.path.basename(paths["ffmpeg"]))[0]

        # ── Patch result.json ─────────────────────────────────────────────
        result_path = os.path.join("data", "processed", result_id, "result.json")
        if os.path.isfile(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                pipeline_elapsed = round(time.time() - pipeline_started_at, 2)
                inner_elapsed = rdata.get("processing_time_seconds")
                if inner_elapsed is not None:
                    rdata["model_processing_time_seconds"] = inner_elapsed
                rdata["processing_time_seconds"] = pipeline_elapsed
                rdata["pipeline_time_seconds"] = pipeline_elapsed
                rdata["enhancements"] = enhancement_paths
                gate = rdata.get("main_conversation_gate") or {}
                if gate.get("applied") and rdata.get("audio_file"):
                    rdata["playback_audio_file"] = rdata["audio_file"]
                    rdata["enhanced_file"] = rdata["audio_file"]
                elif enhancement_paths.get("playback_loud"):
                    rdata["playback_audio_file"] = enhancement_paths["playback_loud"]
                    rdata["enhanced_file"] = enhancement_paths["playback_loud"]
                elif enhancement_paths.get("ffmpeg"):
                    rdata["playback_audio_file"] = enhancement_paths["ffmpeg"]
                    rdata["enhanced_file"] = enhancement_paths["ffmpeg"]

                # Cache orig_meta once so /api/calls never needs to run ffprobe.
                if not rdata.get("orig_meta"):
                    orig_audio = upload_path
                    try:
                        r = subprocess.run(
                            ["ffprobe", "-v", "quiet", "-print_format", "json",
                             "-show_format", "-show_streams", orig_audio],
                            capture_output=True, text=True, timeout=15, env=_ENV,
                        )
                        if r.returncode == 0:
                            fd = json.loads(r.stdout)
                            fmt_d  = fd.get("format", {})
                            streams = fd.get("streams", [{}])
                            rdata["orig_meta"] = {
                                "duration_s":   round(float(fmt_d.get("duration", 0))),
                                "bitrate_kbps": int(fmt_d.get("bit_rate", 0)) // 1000,
                                "size_mb":      round(int(fmt_d.get("size", 0)) / 1024 / 1024, 1),
                                "sample_rate":  streams[0].get("sample_rate", "?"),
                                "channels":     streams[0].get("channels", 1),
                            }
                    except Exception as _e:
                        print(f"[UI] orig_meta ffprobe failed: {_e}")

                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(rdata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[UI] result.json patch failed: {e}")

        completed_at = time.time()
        pipeline_elapsed = round(completed_at - pipeline_started_at, 2)
        with _status_lock:
            _status.update(done=True, running=False, stage_num=4,
                           stage="Complete", message="Done! Results saved.",
                           result_id=result_id, completed_at=completed_at,
                           updated_at=completed_at,
                           elapsed_seconds=pipeline_elapsed,
                           stage_elapsed_seconds=0.0,
                           processing_time_seconds=pipeline_elapsed)
        print(f"[UI] Pipeline complete. Result: {result_id}")

    except PipelineCancelled:
        print("[UI] Pipeline cancelled by user.", flush=True)
        with _status_lock:
            completed_at = time.time()
            _status.update(running=False, done=False, error=None,
                           stage_num=0, stage="Cancelled",
                           message="Cancelled by user", cancel_requested=False,
                           completed_at=completed_at, updated_at=completed_at,
                           elapsed_seconds=round(completed_at - pipeline_started_at, 2))
    except BaseException as e:
        import traceback as _tb
        print(f"[UI] Pipeline error: {e}", flush=True)
        _tb.print_exc()
        with _status_lock:
            completed_at = time.time()
            _status.update(error=str(e), running=False, cancel_requested=False,
                           completed_at=completed_at, updated_at=completed_at,
                           elapsed_seconds=round(completed_at - pipeline_started_at, 2))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  On-demand enhancement for existing calls  (async background thread)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _enhance_existing_worker(call_id: str, result_path: str):
    """Background thread: generate all 4 enhancements for an existing call."""
    with _enhance_lock:
        _enhance_status[call_id] = {"running": True, "done": False, "error": None, "paths": {}}

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            rdata = json.load(f)

        ffmpeg_path = rdata["audio_file"].replace("\\", "/")
        # Derive original filename by stripping "enhanced_" prefix
        parts    = ffmpeg_path.split("/")
        orig_fn  = re.sub(r"^enhanced_", "", parts[-1])
        parts[-1] = orig_fn
        original_path = "/".join(parts)

        existing = rdata.get("enhancements", {})
        existing["ffmpeg"] = ffmpeg_path

        raw_dir  = "/".join(parts[:-1])

        runners = [
            ("noisereduce", f"{raw_dir}/nr_{orig_fn}",  _enhance_noisereduce),
            ("deepfilter",  f"{raw_dir}/df_{orig_fn}",  _enhance_deepfilternet),
            ("metricgan",   f"{raw_dir}/mg_{orig_fn}",  _enhance_metricgan),
        ]

        for key, out_path, fn in runners:
            if key in existing and os.path.isfile(existing[key]):
                print(f"[Enhance] {key} already exists, skipping.")
                continue
            if os.path.isfile(out_path):
                print(f"[Enhance] {key} file exists on disk, skipping run.")
                existing[key] = out_path
                continue
            print(f"[Enhance] Running {key} on {original_path} (5-min cap)...")
            try:
                fn(original_path, out_path, max_seconds=300)
                existing[key] = out_path
                print(f"[Enhance] {key} done -> {out_path}")
            except Exception as e:
                print(f"[Enhance] {key} failed: {e}")

        # Patch result.json
        rdata["enhancements"] = existing
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(rdata, f, indent=2, ensure_ascii=False)

        with _enhance_lock:
            _enhance_status[call_id].update(running=False, done=True, paths=existing)
        print(f"[Enhance] All done for {call_id}.")

    except Exception as e:
        print(f"[Enhance] Worker error: {e}")
        with _enhance_lock:
            _enhance_status[call_id].update(running=False, done=False, error=str(e))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Agent Enrollment  (background thread)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _enroll_worker(recordings_dir: str):
    """Process all agent recordings and save an ECAPA voiceprint."""
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    with _enroll_lock:
        _enroll_status.update(running=True, done=False, error=None, message="Starting...")
    try:
        from src.speaker_role import enroll_agent
        def _progress(i: int, total: int, fname: str):
            with _enroll_lock:
                _enroll_status["message"] = f"Processing {i+1}/{total}: {fname}"
        msg = enroll_agent(recordings_dir, progress_cb=_progress)
        with _enroll_lock:
            _enroll_status.update(running=False, done=True, message=msg)
        print(f"[Enroll] Done: {msg}", flush=True)
    except Exception as e:
        with _enroll_lock:
            _enroll_status.update(running=False, done=False, error=str(e), message=str(e))
        print(f"[Enroll] Error: {e}", flush=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Auto-Training  (background thread for daily daemon)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_train_lock   = threading.Lock()
_train_status: dict = {"running": False, "done": False, "error": None,
                       "message": "", "started_at": None,
                       "selected_agents": [], "active_agent": "",
                       "results": {}, "log": [], "exit_code": None}

_VOICEPRINT_DIR = os.path.join(os.path.dirname(__file__), "data", "agent_voiceprints")
_AGENTS_JSON    = os.path.join(_VOICEPRINT_DIR, "agents.json")
_TRAINING_HIST  = os.path.join(_VOICEPRINT_DIR, "training_history.json")


def _read_agents_json() -> dict:
    """Read agents.json, returning {} on any failure."""
    if not os.path.isfile(_AGENTS_JSON):
        return {}
    try:
        with open(_AGENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_training_history() -> dict:
    if not os.path.isfile(_TRAINING_HIST):
        return {}
    try:
        with open(_TRAINING_HIST, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_last_training_reports() -> dict:
    reports_dir = os.path.join(_VOICEPRINT_DIR, "daily_reports")
    reports: dict = {}
    if not os.path.isdir(reports_dir):
        return reports
    for name in os.listdir(reports_dir):
        if not name.endswith(".last_training_report.json"):
            continue
        path = os.path.join(reports_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
            slug = report.get("agent_slug") or name.replace(".last_training_report.json", "")
            reports[slug] = {
                "path": path,
                "agent_slug": slug,
                "agent_name": report.get("agent_name"),
                "training_rows": report.get("training_rows"),
                "customer_calibration_rows": report.get("customer_calibration_rows"),
                "activation_eligible": report.get("activation_eligible"),
                "activated": report.get("activated"),
                "blocked_by_existing": report.get("blocked_by_existing"),
                "dry_run": report.get("dry_run"),
                "same_data_accuracy": report.get("same_data_accuracy") or {},
                "loco_result": report.get("loco_result") or {},
                "artifacts": report.get("artifacts") or {},
            }
        except Exception:
            continue
    return reports


_role_correction_lock = threading.Lock()


def _result_path_for_call(call_id: str) -> Path:
    safe_id = str(call_id or "").replace("\\", "/").strip("/")
    base_dir = Path(__file__).resolve().parent
    candidates = [
        Path(PROCESSED_DIR) / safe_id / "result.json",
        base_dir / PROCESSED_DIR / safe_id / "result.json",
        base_dir / "data" / "processed" / safe_id / "result.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _fmt_ts_seconds(seconds: float) -> str:
    s = max(0.0, float(seconds or 0.0))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _normalise_manual_role(role: str) -> str:
    value = str(role or "").strip().upper()
    if value in {"AGENT", "CUSTOMER"}:
        return value
    raise ValueError("role must be AGENT or CUSTOMER")


def _display_for_role(role: str, agent_name: str) -> str:
    return agent_name if role == "AGENT" else "Customer"


def _apply_manual_role(
    item: dict,
    role: str,
    agent_name: str,
    stamp: str,
    source: str = "ui_line_correction",
    voice_verified: bool = False,
) -> None:
    item["identified_speaker"] = role
    item["display_speaker"] = _display_for_role(role, agent_name)
    item["manual_role_correction"] = True
    item["role_correction_source"] = source
    item["role_correction_at"] = stamp
    if voice_verified:
        item["manual_voice_role_correction"] = True
    else:
        item.pop("manual_voice_role_correction", None)
    if role == "AGENT":
        item["agent_name"] = agent_name
    else:
        item.pop("agent_name", None)


def _sync_transcription_role(
    rdata: dict,
    idx: int,
    role: str,
    agent_name: str,
    stamp: str,
    source: str = "ui_line_correction",
    voice_verified: bool = False,
) -> None:
    rows = rdata.get("transcription_json")
    if not isinstance(rows, list) or not (0 <= idx < len(rows)):
        return
    _apply_manual_role(rows[idx], role, agent_name, stamp, source, voice_verified)


def _split_item_for_manual_role(
    item: dict,
    role: str,
    agent_name: str,
    stamp: str,
    source: str,
    voice_verified: bool,
    selection: dict | None,
) -> tuple[list[dict], int]:
    """Apply a manual role to a selected word range inside one segment."""
    base = dict(item)
    if not isinstance(selection, dict):
        _apply_manual_role(base, role, agent_name, stamp, source, voice_verified)
        return [base], 0

    try:
        seg_start = float(base.get("start", 0.0))
        seg_end = float(base.get("end", 0.0))
        sel_start = float(selection.get("start"))
        sel_end = float(selection.get("end"))
    except Exception:
        _apply_manual_role(base, role, agent_name, stamp, source, voice_verified)
        return [base], 0

    text = str(base.get("text") or base.get("phrase") or "").strip()
    words = re.findall(r"\S+", text)
    seg_dur = seg_end - seg_start
    if len(words) <= 1 or seg_dur <= 0 or sel_end <= sel_start:
        _apply_manual_role(base, role, agent_name, stamp, source, voice_verified)
        return [base], 0

    sel_start = max(seg_start, min(seg_end, sel_start))
    sel_end = max(seg_start, min(seg_end, sel_end))
    per_word = seg_dur / len(words)
    selected = [
        i for i in range(len(words))
        if (seg_start + ((i + 1) * per_word)) > sel_start
        and (seg_start + (i * per_word)) < sel_end
    ]
    if not selected:
        _apply_manual_role(base, role, agent_name, stamp, source, voice_verified)
        return [base], 0

    first = selected[0]
    last = selected[-1]
    if first == 0 and last == len(words) - 1:
        _apply_manual_role(base, role, agent_name, stamp, source, voice_verified)
        return [base], 0

    parent_start = seg_start
    parent_end = seg_end

    def make_chunk(start_word: int, end_word: int, selected_chunk: bool) -> dict | None:
        if start_word > end_word:
            return None
        chunk = dict(base)
        chunk["start"] = round(seg_start + (start_word * per_word), 3)
        chunk["end"] = round(seg_start + ((end_word + 1) * per_word), 3)
        chunk["text"] = " ".join(words[start_word:end_word + 1])
        chunk["manual_word_split"] = True
        chunk["manual_word_split_parent_start"] = parent_start
        chunk["manual_word_split_parent_end"] = parent_end
        if selected_chunk:
            _apply_manual_role(chunk, role, agent_name, stamp, source, voice_verified)
        return chunk

    chunks: list[dict] = []
    selected_offset = 0
    before = make_chunk(0, first - 1, False)
    if before:
        chunks.append(before)
        selected_offset = 1
    selected_chunk = make_chunk(first, last, True)
    if selected_chunk:
        chunks.append(selected_chunk)
    after = make_chunk(last + 1, len(words) - 1, False)
    if after:
        chunks.append(after)
    return chunks, selected_offset


def _recompute_role_stats(segments: list) -> dict:
    stats = {
        "AGENT": {"time_s": 0.0, "turns": 0, "first_words": ""},
        "CUSTOMER": {"time_s": 0.0, "turns": 0, "first_words": ""},
    }
    for seg in segments or []:
        role = str(seg.get("identified_speaker") or "").upper()
        if role not in stats:
            continue
        try:
            dur = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        except Exception:
            dur = 0.0
        stats[role]["time_s"] += dur
        stats[role]["turns"] += 1
        if not stats[role]["first_words"]:
            text = str(seg.get("text") or seg.get("phrase") or "").strip()
            stats[role]["first_words"] = text[:120]
    for role in stats:
        stats[role]["time_s"] = round(stats[role]["time_s"], 1)
    return stats


def _normalise_asr_loop_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _collapse_repeated_asr_loops(segments: list) -> tuple[list, dict]:
    """Drop model repeat loops without any phrase-specific word list."""
    if not isinstance(segments, list) or not segments:
        return segments, {"enabled": True, "removed_segments": 0, "collapsed_runs": 0}

    min_run = int(os.getenv("SST_ASR_REPEAT_LOOP_MIN_RUN", "4") or "4")
    keep_count = max(1, int(os.getenv("SST_ASR_REPEAT_LOOP_KEEP", "1") or "1"))
    max_item_s = float(os.getenv("SST_ASR_REPEAT_LOOP_MAX_ITEM_SECONDS", "3.0") or "3.0")
    max_gap_s = float(os.getenv("SST_ASR_REPEAT_LOOP_MAX_GAP_SECONDS", "1.0") or "1.0")
    max_chars = int(os.getenv("SST_ASR_REPEAT_LOOP_MAX_CHARS", "80") or "80")

    remove_indexes: set[int] = set()
    runs: list[dict] = []
    current: list[tuple[int, dict, float, float]] = []
    current_key = ""
    last_end = None

    def close_run() -> None:
        nonlocal current, current_key, last_end
        if len(current) >= min_run:
            dropped = current[keep_count:]
            if dropped:
                for idx, _seg, _start, _end in dropped:
                    remove_indexes.add(idx)
                runs.append({
                    "text": current_key[:80],
                    "start": round(current[0][2], 3),
                    "end": round(current[-1][3], 3),
                    "count": len(current),
                    "removed": len(dropped),
                })
        current = []
        current_key = ""
        last_end = None

    for idx, seg in enumerate(segments):
        key = _normalise_asr_loop_text(seg.get("text") or seg.get("phrase") or "")
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except Exception:
            start = end = 0.0
        dur = max(0.0, end - start)
        gap = 0.0 if last_end is None else start - last_end
        usable = bool(key) and len(key) <= max_chars and dur <= max_item_s
        if not usable or gap > max_gap_s or (current_key and key != current_key):
            close_run()
        if usable:
            current.append((idx, seg, start, end))
            current_key = key
            last_end = end
    close_run()

    if not remove_indexes:
        return segments, {"enabled": True, "removed_segments": 0, "collapsed_runs": 0}

    return [seg for idx, seg in enumerate(segments) if idx not in remove_indexes], {
        "enabled": True,
        "removed_segments": len(remove_indexes),
        "collapsed_runs": len(runs),
        "kept_per_run": keep_count,
        "removed_indexes": sorted(remove_indexes),
        "runs": runs[:20],
    }


def _resolve_result_audio_path(rdata: dict, result_path: Path) -> Path | None:
    base_dir = Path(__file__).resolve().parent
    roots = [Path.cwd(), base_dir, base_dir.parent, result_path.parent]
    for key in (
        "diarization_audio_file",
        "asr_audio_file",
        "playback_audio_file",
        "enhanced_file",
        "audio_file",
    ):
        raw = rdata.get(key)
        if not raw:
            continue
        text = str(raw).replace("\\", "/")
        candidates: list[Path] = [Path(text)]
        if "call_processor/" in text:
            tail = text.split("call_processor/", 1)[1]
            candidates.extend(root / tail for root in roots)
        candidates.extend(root / text for root in roots)
        candidates.append(result_path.parent / Path(text).name)
        for path in candidates:
            try:
                if path.is_file():
                    return path
            except OSError:
                continue
    return None


def _l2_norm_np(vec):
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32).squeeze()
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _resolve_voiceprint_file(raw: str) -> Path:
    path = Path(str(raw or ""))
    if path.is_file():
        return path
    return Path(_VOICEPRINT_DIR) / str(raw or "")


def _existing_voiceprint_matches(centroid, agent_slug: str) -> dict:
    import numpy as np

    matches = []
    agents = _read_agents_json()
    for slug, info in agents.items():
        if not isinstance(info, dict):
            continue
        raw_paths = []
        vps = info.get("voiceprints")
        if isinstance(vps, list):
            for entry in vps:
                raw = entry.get("path") if isinstance(entry, dict) else entry
                if raw:
                    raw_paths.append(raw)
        if not raw_paths:
            raw = info.get("voiceprint_path") or info.get("voiceprint")
            if raw:
                raw_paths.append(raw)
        best = None
        for raw in raw_paths:
            path = _resolve_voiceprint_file(raw)
            if not path.is_file():
                continue
            try:
                vp = _l2_norm_np(np.load(path))
            except Exception:
                continue
            if vp.shape != centroid.shape:
                continue
            sim = float(np.dot(vp, centroid))
            best = sim if best is None else max(best, sim)
        if best is not None:
            matches.append({
                "agent_slug": slug,
                "agent_name": info.get("agent_name") or info.get("name") or slug,
                "similarity": round(best, 4),
            })
    matches.sort(key=lambda row: row["similarity"], reverse=True)
    target = next((row for row in matches if row["agent_slug"] == agent_slug), None)
    other = next((row for row in matches if row["agent_slug"] != agent_slug), None)
    return {
        "target_existing_similarity": target["similarity"] if target else None,
        "next_best_other": other,
        "margin_to_next_best": (
            round((target["similarity"] - other["similarity"]), 4)
            if target and other else None
        ),
        "top_matches": matches[:5],
    }


def _manual_training_signature(agent_slug: str, segments: list) -> str:
    payload = [
        {
            "agent_slug": agent_slug,
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "speaker": seg.get("speaker"),
            "text": str(seg.get("text") or seg.get("phrase") or "").strip(),
        }
        for seg in segments
    ]
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _train_manual_role_voiceprint(
    result_path: Path,
    rdata: dict,
    agent_slug: str,
    agent_name: str,
    source_indexes: set[int] | None = None,
) -> dict:
    import numpy as np
    import soundfile as sf

    segments = [
        seg for idx, seg in enumerate(rdata.get("segments", []))
        if str(seg.get("identified_speaker") or "").upper() == "AGENT"
        and seg.get("manual_role_correction")
        and seg.get("manual_voice_role_correction")
        and (source_indexes is None or idx in source_indexes)
    ]
    min_segments = int(os.getenv("SST_MANUAL_ROLE_TRAIN_MIN_SEGMENTS", "2"))
    min_total_s = float(os.getenv("SST_MANUAL_ROLE_TRAIN_MIN_SECONDS", "3.0"))
    min_seg_s = float(os.getenv("SST_MANUAL_ROLE_TRAIN_MIN_SEGMENT_SECONDS", "0.45"))
    max_total_s = float(os.getenv("SST_MANUAL_ROLE_TRAIN_MAX_SECONDS", "90.0"))
    pad_s = float(os.getenv("SST_MANUAL_ROLE_TRAIN_PAD_SECONDS", "0.12"))

    eligible = []
    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except Exception:
            continue
        dur = end - start
        if dur >= min_seg_s:
            eligible.append((start, end, dur, seg))

    total_s = sum(row[2] for row in eligible)
    if len(eligible) < min_segments or total_s < min_total_s:
        return {
            "trained": False,
            "reason": "not_enough_verified_agent_audio",
            "corrected_agent_segments": len(eligible),
            "corrected_agent_seconds": round(total_s, 2),
            "required_segments": min_segments,
            "required_seconds": min_total_s,
        }

    signature = _manual_training_signature(agent_slug, [row[3] for row in eligible])
    previous = rdata.get("manual_role_training") or {}
    if previous.get("signature") == signature and previous.get("voiceprint_file"):
        return {
            "trained": False,
            "reason": "already_trained_for_current_corrections",
            "voiceprint_file": previous.get("voiceprint_file"),
            "signature": signature,
        }

    audio_path = _resolve_result_audio_path(rdata, result_path)
    if not audio_path:
        return {"trained": False, "reason": "source_audio_not_found"}

    try:
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception as exc:
        return {"trained": False, "reason": f"source_audio_read_failed: {exc}"}
    mono = audio.mean(axis=1)

    selected = []
    running_s = 0.0
    for start, end, dur, seg in sorted(eligible, key=lambda row: row[0]):
        if selected and running_s >= max_total_s:
            break
        selected.append((start, end, dur, seg))
        running_s += dur

    from src.embedding_campp import get_model

    existing_target_stack = None
    existing_other_stacks: list[tuple[str, np.ndarray]] = []
    try:
        from src.diar_multi import _load_voiceprints

        loaded_voiceprints = _load_voiceprints()
        target = loaded_voiceprints.get(agent_slug)
        if target and getattr(target[1], "ndim", 0) == 2 and len(target[1]):
            existing_target_stack = np.asarray(target[1], dtype=np.float32)
            existing_other_stacks = [
                (slug, np.asarray(stack, dtype=np.float32))
                for slug, (_name, stack) in loaded_voiceprints.items()
                if slug != agent_slug
                and getattr(stack, "ndim", 0) == 2
                and stack.shape[1] == existing_target_stack.shape[1]
                and len(stack)
            ]
    except Exception:
        existing_target_stack = None
        existing_other_stacks = []

    embedder = get_model(force_cpu=True)
    embedder.load(force_cpu=True)
    embs = []
    rejected_by_existing_voiceprint = 0
    train_min_existing_sim = float(os.getenv("SST_MANUAL_ROLE_TRAIN_MIN_EXISTING_SIM", "0.40") or "0.40")
    train_min_existing_margin = float(os.getenv("SST_MANUAL_ROLE_TRAIN_MIN_EXISTING_MARGIN", "0.03") or "0.03")
    segment_quality_samples = []
    for start, end, _dur, _seg in selected:
        s0 = max(0, int((start - pad_s) * sr))
        s1 = min(len(mono), int((end + pad_s) * sr))
        if s1 <= s0:
            continue
        try:
            emb = embedder.embed_chunk(mono[s0:s1], sr=sr)
        except Exception:
            emb = None
        if emb is not None:
            emb = _l2_norm_np(emb)
            target_sim = None
            margin = None
            if (
                existing_target_stack is not None
                and emb.shape[0] == existing_target_stack.shape[1]
            ):
                target_sim = float(np.max(existing_target_stack @ emb))
                best_other = 0.0
                best_other_slug = None
                for slug, stack in existing_other_stacks:
                    sim = float(np.max(stack @ emb))
                    if sim > best_other:
                        best_other = sim
                        best_other_slug = slug
                margin = target_sim - best_other
                if target_sim < train_min_existing_sim or (
                    existing_other_stacks and margin < train_min_existing_margin
                ):
                    rejected_by_existing_voiceprint += 1
                    if len(segment_quality_samples) < 20:
                        segment_quality_samples.append({
                            "start": round(start, 2),
                            "end": round(end, 2),
                            "accepted": False,
                            "target_similarity": round(target_sim, 4),
                            "margin": round(margin, 4),
                            "best_other_slug": best_other_slug,
                            "text": str(_seg.get("text") or "")[:120],
                        })
                    continue
            embs.append(emb)
            if len(segment_quality_samples) < 20:
                segment_quality_samples.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "accepted": True,
                    "target_similarity": round(target_sim, 4) if target_sim is not None else None,
                    "margin": round(margin, 4) if margin is not None else None,
                    "text": str(_seg.get("text") or "")[:120],
                })

    if len(embs) < min_segments:
        return {
            "trained": False,
            "reason": "embedding_failed_for_verified_audio",
            "embedded_segments": len(embs),
            "rejected_by_existing_voiceprint": rejected_by_existing_voiceprint,
            "required_segments": min_segments,
        }

    centroid = _l2_norm_np(np.mean(np.stack(embs), axis=0))
    validation = _existing_voiceprint_matches(centroid, agent_slug)

    os.makedirs(_VOICEPRINT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    file_name = f"{agent_slug}_ui_corrected_{stamp}.npy"
    vp_path = Path(_VOICEPRINT_DIR) / file_name
    np.save(vp_path, centroid.astype(np.float32))

    agents = _read_agents_json()
    info = agents.setdefault(agent_slug, {})
    info["agent_name"] = agent_name or info.get("agent_name") or agent_slug
    vps = info.get("voiceprints")
    if not isinstance(vps, list):
        vps = []
        legacy = info.get("voiceprint_path") or info.get("voiceprint")
        if legacy:
            vps.append({"path": legacy, "source": "legacy"})
        info["voiceprints"] = vps
    entry = {
        "path": file_name,
        "source": "ui_role_correction",
        "embedding_model": embedder.model_name,
        "embedding_dim": int(centroid.shape[0]),
        "manual_verified": True,
        "segment_role_voiceprint": True,
        "source_result": str(result_path.parent.name),
        "source_audio": str(audio_path),
        "source_segment_count": len(selected),
        "source_seconds": round(sum(row[2] for row in selected), 2),
        "signature": signature,
        "created_at_epoch": int(time.time()),
        "validation": validation,
    }
    vps.append(entry)
    info["n_voiceprints"] = len(vps)
    info["embedding_model"] = embedder.model_name
    info["embedding_dim"] = int(centroid.shape[0])
    info["updated_at_epoch"] = int(time.time())

    if os.path.isfile(_AGENTS_JSON):
        backup = Path(_VOICEPRINT_DIR) / f"agents.backup.{agent_slug}.ui_role_correction.{stamp}.json"
        shutil.copy2(_AGENTS_JSON, backup)
    else:
        backup = None
    with open(_AGENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)

    reports_dir = Path(_VOICEPRINT_DIR) / "manual_role_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{agent_slug}_{result_path.parent.name}_{stamp}.json"
    report = {
        "trained": True,
        "agent_slug": agent_slug,
        "agent_name": agent_name,
        "voiceprint_file": file_name,
        "embedding_model": embedder.model_name,
        "embedding_dim": int(centroid.shape[0]),
        "source_result": str(result_path.parent.name),
        "source_audio": str(audio_path),
        "corrected_agent_segments": len(selected),
        "corrected_agent_seconds": round(sum(row[2] for row in selected), 2),
        "embedded_segments": len(embs),
        "rejected_by_existing_voiceprint": rejected_by_existing_voiceprint,
        "train_min_existing_similarity": train_min_existing_sim,
        "train_min_existing_margin": train_min_existing_margin,
        "segment_quality_samples": segment_quality_samples,
        "signature": signature,
        "agents_backup": str(backup) if backup else None,
        "validation": validation,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    rdata["manual_role_training"] = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "signature": signature,
        "voiceprint_file": file_name,
        "report": str(report_path),
        "corrected_agent_segments": len(selected),
        "corrected_agent_seconds": round(sum(row[2] for row in selected), 2),
        "embedding_model": embedder.model_name,
        "embedding_dim": int(centroid.shape[0]),
    }
    return report


def _auto_train_worker(agents_filter: list | None, days: int,
                       activate: bool, dry_run: bool,
                       audiofy_username: str = "",
                       audiofy_password: str = ""):
    """Run daily_training_daemon.py as subprocess, streaming log to status."""
    selected_agents = agents_filter or []
    with _train_lock:
        _train_status.update(running=True, done=False, error=None,
                             message="Starting auto-training daemon...",
                             started_at=time.time(), selected_agents=selected_agents,
                             active_agent="", results={}, log=[], exit_code=None)
    try:
        daemon_script = os.path.join(
            os.path.dirname(__file__), "scripts", "daily_training_daemon.py"
        )
        # -u keeps the child Python's stdout unbuffered so the UI's
        # /api/auto-train-status log_tail streams live instead of waiting
        # for the daemon to exit.
        cmd = [sys.executable, "-u", daemon_script,
               "--days", str(days),
               "--work-dir", os.path.join(
                   os.path.dirname(os.path.dirname(__file__)),
                   "traning_data", "_daily_auto")]
        if agents_filter:
            cmd.append("--agents")
            cmd.extend(agents_filter)
            if len(agents_filter) == 1:
                cmd.extend(["--user-name", str(agents_filter[0])])
        if activate and not dry_run:
            cmd.append("--activate")
        if dry_run:
            cmd.append("--dry-run")

        safe_cmd = " ".join(cmd)
        print(f"[AutoTrain] Running: {safe_cmd}", flush=True)
        child_env = _ENV.copy()
        child_env["PYTHONUNBUFFERED"] = "1"   # belt-and-suspenders w/ -u flag above
        if audiofy_username:
            child_env["AUDIOFY_USERNAME"] = audiofy_username
        if audiofy_password:
            child_env["AUDIOFY_PASSWORD"] = audiofy_password
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,                # line-buffered on the parent's read end
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=child_env,
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            with _train_lock:
                _train_status["log"].append(line)
                if len(_train_status["log"]) > 500:
                    _train_status["log"] = _train_status["log"][-300:]
                # Parse status from output
                if "[agent]" in line:
                    agent_text = line.split("[agent]")[-1].strip()
                    _train_status["active_agent"] = agent_text
                    _train_status["message"] = f"Training {agent_text}"
                elif "[result]" in line:
                    _train_status["message"] = line.split("[result]")[-1].strip()
                elif "DAILY TRAINING SUMMARY" in line:
                    _train_status["message"] = "Generating summary..."

        proc.wait()
        reports = _read_last_training_reports()
        with _train_lock:
            _train_status["exit_code"] = proc.returncode
            _train_status["results"] = reports
            if proc.returncode == 0:
                _train_status.update(running=False, done=True,
                                     message="Training complete!")
            else:
                _train_status.update(
                    running=False, done=False,
                    error=f"Daemon exited with code {proc.returncode}",
                    message=f"Failed (exit {proc.returncode})")

    except Exception as e:
        with _train_lock:
            _train_status.update(running=False, done=False,
                                 error=str(e), message=str(e))
        print(f"[AutoTrain] Error: {e}", flush=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  HTTP Request Handler
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class RequestHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        if "favicon" not in self.path:
            super().log_message(format, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def guess_type(self, path):
        """Add `charset=utf-8` to text MIME types so browsers don't fall back
        to Latin-1 and render UTF-8 characters (emoji, em-dash, ellipsis) as
        mojibake. The HTML's <meta charset> tag is also respected, but the
        HTTP header takes precedence and some browsers stick with the first
        bytes before parsing the meta tag."""
        ctype = super().guess_type(path)
        if isinstance(ctype, str) and ctype.startswith("text/") and "charset" not in ctype:
            return f"{ctype}; charset=utf-8"
        if ctype == "application/javascript" and "charset" not in (ctype or ""):
            return "application/javascript; charset=utf-8"
        return ctype

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        """HTTP Basic Auth gate. Returns True if authorized.

        No-op (always True) unless CALLPROC_AUTH_REQUIRED is set, so the
        browser does not pop a login prompt by default. When enabled, sends
        401 + WWW-Authenticate on missing/incorrect credentials and returns
        False so callers can `return` immediately.
        """
        if not AUTH_REQUIRED:
            return True
        if self.headers.get("Authorization", "") == _AUTH_EXPECTED:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{AUTH_REALM}"')
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Authentication required")
        return False

    def do_GET(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        path   = parsed.path

        # /api/status
        if path == "/api/status":
            self._json(_status_snapshot())
            return

        # /api/calls
        if path == "/api/calls":
            calls = []
            if os.path.exists(PROCESSED_DIR):
                for d in sorted(os.listdir(PROCESSED_DIR), reverse=True):
                    rp = os.path.join(PROCESSED_DIR, d, "result.json")
                    if not os.path.isfile(rp):
                        continue
                    try:
                        with open(rp, encoding="utf-8") as f:
                            rdata = json.load(f)
                        audio_file = rdata.get("audio_file", "")
                        orig_file = audio_file.replace("enhanced_", "").replace("\\", "/")
                        # orig_meta is written into result.json by _run_pipeline at
                        # completion — never computed here to keep this endpoint fast.
                        orig_meta = rdata.get("orig_meta") or {}
                        calls.append({
                            "id": d,
                            "model": rdata.get("model", "unknown"),
                            "segments": rdata.get("total_segments", 0),
                            "processed_at": rdata.get("processed_at", ""),
                            "processing_time_s": rdata.get("processing_time_seconds", 0),
                            "audio_file": audio_file,
                            "orig_file": orig_file,
                            "orig_meta": orig_meta,
                        })
                    except Exception:
                        calls.append({"id": d, "model": "unknown", "segments": 0,
                                      "processed_at": "", "processing_time_s": 0,
                                      "audio_file": "", "orig_file": "", "orig_meta": {}})
            self._json(calls)
            return

        # /api/benchmark
        if path == "/api/benchmark":
            BENCHMARK = [
                # rank = relative quality score 1-10 for sorting/display
                {"model": "deepgram-nova-3",           "label": "Deepgram Nova-3",          "type": "Cloud API",  "speed_s": 6,    "segments": 60,   "notes": "Best accuracy + diarization",    "rank": 1, "wer": "~7%",   "status": "ok"},
                {"model": "whisper-large-v3",          "label": "Whisper Large-v3",         "type": "Local GPU",  "speed_s": 78,   "segments": 306,  "notes": "Best quality local Whisper",     "rank": 2, "wer": "8.1%",  "status": "ok"},
                {"model": "whisper-large-v3-turbo",    "label": "Whisper Large-v3-Turbo",   "type": "Local GPU",  "speed_s": 35,   "segments": 307,  "notes": "Fast, near large-v3 quality",    "rank": 3, "wer": "8.4%",  "status": "ok"},
                {"model": "distil-whisper-large-v3.5", "label": "Distil-Whisper v3.5",      "type": "Local GPU",  "speed_s": 29,   "segments": 433,  "notes": "Fastest local, most granular",   "rank": 4, "wer": "8.6%",  "status": "ok"},
                {"model": "parakeet-tdt-0.6b-v3",      "label": "NVIDIA Parakeet TDT v3",   "type": "Local GPU",  "speed_s": 45,   "segments": 126,  "notes": "GPU subprocess isolation protects UI from native CUDA aborts", "rank": 5, "wer": "~5.5%", "status": "ok"},
                {"model": "qwen3-asr-1.7b",             "label": "Qwen3-ASR 1.7B",           "type": "Local GPU",  "speed_s": 0,    "segments": 0,    "notes": "Experimental local model for noisy-call ASR checks", "rank": 6, "wer": "test", "status": "experimental"},
                {"model": "cohere-transcribe-03-2026", "label": "Cohere Transcribe 03-2026","type": "Local GPU",  "speed_s": 52,   "segments": 60,   "notes": "Lowest WER on leaderboard",      "rank": 7, "wer": "5.42%", "status": "ok"},
                {"model": "deepgram-nova-2-phonecall", "label": "Deepgram Nova-2 Phone",    "type": "Cloud API",  "speed_s": 6,    "segments": 33,   "notes": "Optimised for phone call audio", "rank": 8, "wer": "~9%",   "status": "ok"},
                {"model": "deepgram-nova-2-meeting",   "label": "Deepgram Nova-2 Meeting",  "type": "Cloud API",  "speed_s": 8,    "segments": 51,   "notes": "Multi-speaker meetings",         "rank": 9, "wer": "~9%",   "status": "ok"},
            ]
            # Merge in any live data from model_comparison.json
            mc_path = os.path.join("data", "model_comparison.json")
            if os.path.isfile(mc_path):
                try:
                    with open(mc_path) as f:
                        mc = {r["model"]: r for r in json.load(f) if "model" in r}
                    for row in BENCHMARK:
                        live = mc.get(row["model"])
                        if live:
                            if live.get("transcribe_s"):
                                row["speed_s"] = round(live["transcribe_s"])
                            if live.get("segments"):
                                row["segments"] = live["segments"]
                except Exception:
                    pass
            self._json(BENCHMARK)
            return

        # /api/call/<id>
        if path.startswith("/api/call/"):
            call_id     = unquote(path.split("/api/call/")[1])
            result_path = os.path.join(PROCESSED_DIR, call_id, "result.json")
            if os.path.isfile(result_path):
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                with open(result_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        # /api/enhance/<call_id>  — start async job
        if path.startswith("/api/enhance/"):
            call_id     = unquote(path.split("/api/enhance/")[1])
            result_path = os.path.join(PROCESSED_DIR, call_id, "result.json")
            if not os.path.isfile(result_path):
                self._json({"status": "error", "message": "Call not found"}, 404)
                return
            with _enhance_lock:
                st = _enhance_status.get(call_id, {})
            if st.get("running"):
                self._json({"status": "running"})
                return
            if st.get("done"):
                self._json({"status": "done", "paths": st["paths"]})
                return
            # Start background worker
            t = threading.Thread(target=_enhance_existing_worker,
                                 args=(call_id, result_path), daemon=True)
            t.start()
            self._json({"status": "started"})
            return

        # /api/enhance_status/<call_id>  — poll job
        if path.startswith("/api/enhance_status/"):
            call_id = unquote(path.split("/api/enhance_status/")[1])
            with _enhance_lock:
                st = dict(_enhance_status.get(call_id, {}))
            self._json(st if st else {"status": "not_started"})
            return

        # /api/test-results — Load actual test results from file if available
        if path == "/api/test-results":
            test_results = None
            # Use absolute path relative to the call_processor directory
            base_dir = Path(__file__).parent
            test_file = base_dir / "data" / "test_results_top5_parakeet.json"

            if test_file.exists():
                try:
                    with open(test_file) as f:
                        test_results = json.load(f)
                except Exception:
                    pass

            # Fallback to placeholder if file doesn't exist yet
            if not test_results:
                test_results = {
                    "test_date": "2026-05-05",
                    "test_type": "Top 5 Agents Test (In Progress)",
                    "status": "Test running...",
                    "message": "Check back in a few minutes for results"
                }

            self._json(test_results)
            return

        # /api/enrollment-status
        if path == "/api/enrollment-status":
            with _enroll_lock:
                st = dict(_enroll_status)
            st["recordings_dir"] = AGENT_RECORDINGS_DIR
            try:
                from src.voiceprints import voiceprint_inventory
                inv = voiceprint_inventory()
            except Exception as _vp_err:
                inv = {"enrolled": False, "agent_count": 0, "voiceprint_dims": {},
                       "legacy_enrolled": False, "legacy_agent_name": "",
                       "missing_count": 0, "error": str(_vp_err)}
            st["enrolled"] = bool(inv.get("enrolled"))
            st["agent_count"] = int(inv.get("agent_count") or 0)
            st["voiceprint_dims"] = inv.get("voiceprint_dims", {})
            st["missing_voiceprints"] = int(inv.get("missing_count") or 0)
            st["legacy_enrolled"] = bool(inv.get("legacy_enrolled"))
            if st["agent_count"] > 1:
                st["agent_name"] = f"{st['agent_count']} agents enrolled"
            elif st["agent_count"] == 1:
                st["agent_name"] = (inv.get("agent_names") or ["Agent"])[0]
            elif inv.get("legacy_agent_name"):
                st["agent_name"] = inv.get("legacy_agent_name")
            if inv.get("error"):
                st["error"] = inv.get("error")
            self._json(st)
            return

        # /api/agents — list all enrolled agents from agents.json
        if path == "/api/agents":
            agents_data = _read_agents_json()
            last_reports = _read_last_training_reports()
            result = []
            for slug, info in sorted(agents_data.items(),
                                     key=lambda kv: kv[1].get("agent_name", "")):
                report = last_reports.get(slug, {})
                same_data = report.get("same_data_accuracy") or {}
                loco = report.get("loco_result") or {}
                result.append({
                    "slug": slug,
                    "name": info.get("agent_name", slug),
                    "model": info.get("embedding_model", "ecapa"),
                    "n_voiceprints": info.get("n_voiceprints",
                                              len(info.get("voiceprints", [])) or 1),
                    "mean_inside_sim": info.get("mean_inside_sim"),
                    "max_outside_sim": info.get("max_outside_sim"),
                    "source": info.get("source", "unknown"),
                    "n_training_segments": info.get("n_training_segments"),
                    "total_training_seconds": info.get("total_training_seconds"),
                    "updated_at_epoch": info.get("updated_at_epoch"),
                    "last_report_path": report.get("path"),
                    "last_training_rows": report.get("training_rows"),
                    "last_customer_rows": report.get("customer_calibration_rows"),
                    "last_overall_accuracy": same_data.get("overall_accuracy"),
                    "last_agent_accuracy": same_data.get("agent_accuracy"),
                    "last_customer_accuracy": same_data.get("customer_accuracy"),
                    "last_loco_accuracy": loco.get("overall_accuracy"),
                    "last_activation_eligible": report.get("activation_eligible"),
                    "last_activated": report.get("activated"),
                    "last_dry_run": report.get("dry_run"),
                })
            self._json(result)
            return

        # /api/training-history — day-by-day quality tracking
        if path == "/api/training-history":
            self._json(_read_training_history())
            return

        # /api/auto-train-status — poll training daemon status
        if path == "/api/auto-train-status":
            with _train_lock:
                st = {
                    "running": _train_status["running"],
                    "done":    _train_status["done"],
                    "error":   _train_status.get("error"),
                    "message": _train_status.get("message", ""),
                    "started_at": _train_status.get("started_at"),
                    "selected_agents": _train_status.get("selected_agents", []),
                    "active_agent": _train_status.get("active_agent", ""),
                    "exit_code": _train_status.get("exit_code"),
                    "results": _train_status.get("results", {}),
                    "log_tail": _train_status["log"][-30:],
                }
            self._json(st)
            return

        # /api/enroll-agent  — start enrollment in background
        if path == "/api/enroll-agent":
            with _enroll_lock:
                if _enroll_status.get("running"):
                    self._json({"status": "already_running",
                                "message": _enroll_status.get("message", "")})
                    return
            if not os.path.isdir(AGENT_RECORDINGS_DIR):
                self._json({"status": "error",
                            "message": f"Directory not found: {AGENT_RECORDINGS_DIR}"}, 400)
                return
            threading.Thread(target=_enroll_worker, args=(AGENT_RECORDINGS_DIR,),
                             daemon=True).start()
            self._json({"status": "started"})
            return

        # Audio files (Range support for seeking)
        audio_exts   = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
        decoded_path = unquote(path)
        if any(decoded_path.lower().endswith(ext) for ext in audio_exts):
            self._serve_audio(decoded_path)
            return

        if path == "/" or path == "/index.html":
            # Serve fresh HTML/JS — disable browser cache so UI fixes hit immediately.
            try:
                with open("index.html", "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(body)
                return
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                return
        return super().do_GET()

    def _serve_audio(self, url_path: str):
        fs_path = os.path.normpath(url_path.lstrip("/"))
        if not os.path.isfile(fs_path):
            self.send_response(404)
            self.end_headers()
            return
        ext  = os.path.splitext(fs_path)[1].lower()
        mime = {".mp3": "audio/mpeg", ".wav": "audio/wav",
                ".flac": "audio/flac", ".m4a": "audio/mp4",
                ".ogg": "audio/ogg"}.get(ext, "audio/mpeg")
        size   = os.path.getsize(fs_path)
        rng    = self.headers.get("Range", "")
        m      = re.match(r"bytes=(\d+)-(\d*)", rng)
        if m:
            start  = int(m.group(1))
            end    = int(m.group(2)) if m.group(2) else size - 1
            end    = min(end, size - 1)
            length = end - start + 1
            with open(fs_path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            self.send_response(206)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
        else:
            with open(fs_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)

    def do_POST(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)

        # POST /api/call/<id>/role-corrections
        # Persist reviewer-confirmed AGENT/CUSTOMER labels and, when requested,
        # append corrected agent audio as a verified voiceprint for future runs.
        role_correction_path = parsed.path.rstrip("/")
        if role_correction_path.startswith("/api/call/") and "/role-corrections" in role_correction_path:
            call_id = unquote(
                role_correction_path[len("/api/call/"):].rsplit("/role-corrections", 1)[0]
            )
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            except Exception as exc:
                self._json({"status": "error", "message": f"Invalid JSON: {exc}"}, 400)
                return

            updates = payload.get("updates") or []
            if isinstance(updates, dict):
                updates = [updates]
            if not isinstance(updates, list) or not updates:
                self._json({"status": "error", "message": "No role updates supplied"}, 400)
                return

            result_path = _result_path_for_call(call_id)
            if not result_path.is_file():
                self._json({"status": "error", "message": "Call not found"}, 404)
                return

            with _role_correction_lock:
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        rdata = json.load(f)
                except Exception as exc:
                    self._json({"status": "error", "message": f"Could not read result: {exc}"}, 500)
                    return

                agent_slug = (
                    payload.get("agent_slug")
                    or rdata.get("identified_agent_slug")
                    or rdata.get("target_agent_slug")
                    or _resolve_target_agent_slug(None, rdata.get("identified_agent") or "", call_id)
                )
                agents = _read_agents_json()
                agent_info = agents.get(agent_slug, {}) if agent_slug else {}
                agent_name = (
                    payload.get("agent_name")
                    or rdata.get("identified_agent")
                    or agent_info.get("agent_name")
                    or agent_info.get("name")
                    or "Agent"
                )
                if not agent_slug:
                    agent_slug = _resolve_target_agent_slug(agent_name, call_id)
                if not agent_slug:
                    self._json({"status": "error", "message": "No target agent is known for this call"}, 400)
                    return

                segments = rdata.get("segments")
                if not isinstance(segments, list):
                    self._json({"status": "error", "message": "Result has no segment list"}, 400)
                    return

                correction_mode = str(payload.get("correction_mode") or "line").strip().lower()
                if correction_mode not in {"line", "voice"}:
                    correction_mode = "line"
                stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                applied = []
                for update in updates:
                    try:
                        idx = int(update.get("index"))
                        role = _normalise_manual_role(update.get("role"))
                    except Exception as exc:
                        self._json({"status": "error", "message": f"Bad update: {exc}"}, 400)
                        return
                    if idx < 0 or idx >= len(segments):
                        continue
                    update_mode = str(update.get("correction_mode") or correction_mode).strip().lower()
                    if update_mode not in {"line", "voice"}:
                        update_mode = "line"
                    voice_verified = update_mode == "voice"
                    source = "ui_voiceprint_feedback" if voice_verified else "ui_line_correction"
                    selection = update.get("selection") if isinstance(update.get("selection"), dict) else None
                    if selection:
                        rows = rdata.get("transcription_json")
                        can_split_rows = isinstance(rows, list) and len(rows) == len(segments)
                        replacement, selected_offset = _split_item_for_manual_role(
                            segments[idx],
                            role,
                            agent_name,
                            stamp,
                            source,
                            voice_verified,
                            selection,
                        )
                        segments[idx:idx + 1] = replacement
                        if can_split_rows:
                            row_replacement, _ = _split_item_for_manual_role(
                                rows[idx],
                                role,
                                agent_name,
                                stamp,
                                source,
                                voice_verified,
                                selection,
                            )
                            rows[idx:idx + 1] = row_replacement
                        applied_idx = idx + selected_offset
                        applied.append({
                            "index": applied_idx,
                            "original_index": idx,
                            "role": role,
                            "correction_mode": update_mode,
                            "word_split": len(replacement) > 1,
                        })
                    else:
                        _apply_manual_role(segments[idx], role, agent_name, stamp, source, voice_verified)
                        _sync_transcription_role(rdata, idx, role, agent_name, stamp, source, voice_verified)
                        applied.append({"index": idx, "role": role, "correction_mode": update_mode})

                if not applied:
                    self._json({"status": "error", "message": "No matching segment indexes"}, 400)
                    return

                rdata["identified_agent"] = agent_name
                rdata["identified_agent_slug"] = agent_slug
                rdata["target_agent_slug"] = rdata.get("target_agent_slug") or agent_slug
                rdata["speaker_stats"] = _recompute_role_stats(segments)
                rdata["manual_role_corrections_count"] = sum(
                    1 for seg in segments if seg.get("manual_role_correction")
                )
                rdata["manual_role_corrections_updated_at"] = stamp

                training_report = {"trained": False, "reason": "training_not_requested"}
                if payload.get("train", True):
                    try:
                        train_indexes = {
                            int(item["index"])
                            for item in applied
                            if item.get("role") == "AGENT"
                            and item.get("correction_mode") == "voice"
                        }
                        training_report = _train_manual_role_voiceprint(
                            result_path,
                            rdata,
                            agent_slug,
                            agent_name,
                            None if train_indexes else set(),
                        )
                    except Exception as exc:
                        training_report = {"trained": False, "reason": f"training_failed: {exc}"}

                try:
                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(rdata, f, indent=2, ensure_ascii=False)
                except Exception as exc:
                    self._json({"status": "error", "message": f"Could not write result: {exc}"}, 500)
                    return

            self._json({
                "status": "ok",
                "updated": len(applied),
                "applied": applied,
                "identified_agent": rdata.get("identified_agent"),
                "identified_agent_slug": rdata.get("identified_agent_slug"),
                "speaker_stats": rdata.get("speaker_stats", {}),
                "segments": rdata.get("segments", []),
                "training_report": training_report,
            })
            return

        # POST /api/call/<id>/swap-roles — flip AGENT ↔ CUSTOMER in result.json
        if parsed.path.startswith("/api/call/") and parsed.path.endswith("/swap-roles"):
            call_id     = unquote(parsed.path[len("/api/call/"):-len("/swap-roles")])
            result_path = os.path.join(PROCESSED_DIR, call_id, "result.json")
            if not os.path.isfile(result_path):
                self._json({"status": "error", "message": "Not found"}, 404)
                return
            with open(result_path, "r", encoding="utf-8") as f:
                rdata = json.load(f)
            _SWAP = {"AGENT": "CUSTOMER", "CUSTOMER": "AGENT"}
            agent_label = rdata.get("identified_agent") or "Agent"

            def _apply_swapped_display(item: dict) -> None:
                new_role = _SWAP.get(
                    item.get("identified_speaker", ""),
                    item.get("identified_speaker", ""),
                )
                item["identified_speaker"] = new_role
                if new_role == "AGENT":
                    item["agent_name"] = item.get("agent_name") or agent_label
                    item["display_speaker"] = item["agent_name"]
                elif new_role == "CUSTOMER":
                    item.pop("agent_name", None)
                    item["display_speaker"] = "Customer"

            for seg in rdata.get("segments", []):
                _apply_swapped_display(seg)
            for seg in rdata.get("transcription_json", []):
                _apply_swapped_display(seg)
            ss = rdata.get("speaker_stats", {})
            rdata["speaker_stats"] = {
                "AGENT":    ss.get("CUSTOMER", {}),
                "CUSTOMER": ss.get("AGENT", {}),
            }
            rdata["roles_swapped"] = not rdata.get("roles_swapped", False)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(rdata, f, indent=2, ensure_ascii=False)
            self._json({"status": "ok", "swapped": rdata["roles_swapped"]})
            return

        if parsed.path == "/api/enroll-clean-upload":
            # Multipart upload of clean single-speaker audio files for enrollment
            import cgi as _cgi
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._json({"status": "error", "message": "Expected multipart/form-data"}, 400)
                return
            with _enroll_lock:
                if _enroll_status.get("running"):
                    self._json({"status": "already_running",
                                "message": _enroll_status.get("message", "")}, 409)
                    return
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            import io as _io
            environ = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type,
                       "CONTENT_LENGTH": str(n)}
            try:
                fs = _cgi.FieldStorage(
                    fp=_io.BytesIO(body),
                    environ=environ,
                    keep_blank_values=True,
                )
            except Exception as _pe:
                self._json({"status": "error", "message": f"Parse error: {_pe}"}, 400)
                return

            clean_dir = os.path.join("data", "clean_agent_recordings")
            os.makedirs(clean_dir, exist_ok=True)

            agent_name = ""
            if "agent_name" in fs:
                agent_name = fs["agent_name"].value.strip()

            saved = 0
            files_field = fs["files"] if "files" in fs else []
            if not isinstance(files_field, list):
                files_field = [files_field]
            for item in files_field:
                if not item.filename:
                    continue
                fname = os.path.basename(item.filename)
                dest  = os.path.join(clean_dir, fname)
                with open(dest, "wb") as _fout:
                    _fout.write(item.file.read())
                saved += 1

            if saved == 0:
                self._json({"status": "error", "message": "No files received"}, 400)
                return

            # Persist agent name
            if agent_name:
                with open(os.path.join("data", "enrolled_agent_name.txt"), "w") as _nf:
                    _nf.write(agent_name)

            # Trigger enrollment — uses KMeans to isolate the agent voice from
            # call recordings that contain both the agent and customers.
            def _clean_enroll_worker():
                with _enroll_lock:
                    _enroll_status.update(running=True, done=False, error=None,
                                          message=f"Enrolling {agent_name or 'agent'}…")
                try:
                    from src.speaker_role import enroll_agent
                    def _prog(i, tot, fname):
                        with _enroll_lock:
                            _enroll_status["message"] = f"{i+1}/{tot}: {fname}"
                    msg = enroll_agent(clean_dir, progress_cb=_prog)
                    with _enroll_lock:
                        _enroll_status.update(running=False, done=True, message=msg)
                except Exception as _e:
                    with _enroll_lock:
                        _enroll_status.update(running=False, done=False,
                                              error=str(_e), message=str(_e))

            threading.Thread(target=_clean_enroll_worker, daemon=True).start()
            self._json({"status": "started", "files_saved": saved,
                        "agent_name": agent_name})
            return

        if parsed.path == "/api/cancel":
            with _status_lock:
                running = _status["running"]
                if running:
                    _status["cancel_requested"] = True
            self._json({"status": "cancelling" if running else "idle"})
            return

        # POST /api/auto-train — trigger daily training daemon
        if parsed.path == "/api/auto-train":
            with _train_lock:
                if _train_status["running"]:
                    self._json({"status": "already_running",
                                "message": _train_status.get("message", "")})
                    return
            # Parse JSON body for options
            n = int(self.headers.get("Content-Length", 0))
            opts = {}
            if n > 0:
                try:
                    opts = json.loads(self.rfile.read(n))
                except Exception:
                    pass
            agents_filter = opts.get("agents")      # list of agent names
            if isinstance(agents_filter, str):
                agents_filter = [agents_filter] if agents_filter.strip() else None
            days          = int(opts.get("days", 7))
            activate      = bool(opts.get("activate", False))
            dry_run       = bool(opts.get("dry_run", True))
            audiofy_username = str(opts.get("audiofy_username") or "").strip()
            audiofy_password = str(opts.get("audiofy_password") or "")

            threading.Thread(
                target=_auto_train_worker,
                args=(agents_filter, days, activate, dry_run,
                      audiofy_username, audiofy_password),
                daemon=True,
            ).start()
            self._json({"status": "started", "days": days,
                        "agents": agents_filter, "activate": activate,
                        "dry_run": dry_run})
            return

        if parsed.path == "/api/upload":
            with _status_lock:
                busy = _status["running"]
            if busy:
                self._json({"status": "busy", "message": "Pipeline already running"}, 409)
                return
            query    = parse_qs(parsed.query)
            filename = query.get("filename", ["upload.mp3"])[0]
            model    = query.get("model",    ["parakeet-tdt-0.6b-v3"])[0]
            target_agent_slug = _resolve_target_agent_slug(
                query.get("agent_slug", ["auto"])[0],
                filename,
            )
            os.makedirs("data/raw_calls", exist_ok=True)
            upload_path = os.path.join("data", "raw_calls", filename)
            n = int(self.headers.get("Content-Length", 0))
            if n > 0:
                with open(upload_path, "wb") as f:
                    f.write(self.rfile.read(n))
            # Clear any stale error from a previous run
            with _status_lock:
                _status.update(running=False, done=False, error=None, result_id=None,
                               stage_num=0, stage="Idle", message="",
                               started_at=None, stage_started_at=None,
                               updated_at=time.time(), completed_at=None,
                               elapsed_seconds=0.0, stage_elapsed_seconds=0.0,
                               processing_time_seconds=None)
            # Send the response BEFORE starting the pipeline thread so the client
            # gets its confirmation even when the server becomes CPU/GPU-bound.
            self._json({
                "status": "started",
                "filename": filename,
                "model": model,
                "target_agent_slug": target_agent_slug,
            })
            threading.Thread(
                target=_run_pipeline,
                args=(upload_path, filename, model, target_agent_slug),
                daemon=True,
            ).start()
            return
        self.send_response(404)
        self.end_headers()

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/call/"):
            call_id = unquote(parsed.path.split("/")[-1])
            target  = os.path.join(PROCESSED_DIR, call_id)
            if os.path.exists(target):
                try:
                    shutil.rmtree(target)
                    self._json({"status": "success"})
                except Exception as e:
                    self._json({"status": "error", "message": str(e)}, 500)
                return
        self.send_response(404)
        self.end_headers()


# ── Auto-enrollment at startup ────────────────────────────────────────────────
_enrolled_path = os.path.join("data", "enrolled_agent.npy")
if os.path.isdir(AGENT_RECORDINGS_DIR) and not os.path.exists(_enrolled_path):
    print("[Startup] Agent voiceprint not found — auto-enrolling from recordings...", flush=True)
    threading.Thread(target=_enroll_worker, args=(AGENT_RECORDINGS_DIR,), daemon=True).start()
elif os.path.exists(_enrolled_path):
    print("[Startup] Agent voiceprint loaded — cosine similarity active.", flush=True)

# ── Startup: garbage-collect orphan / half-finished result dirs ──────────────
def _gc_orphan_processed_dirs() -> None:
    """Remove data/processed/* dirs that have no result.json (failed runs).

    Keeps norm_*.wav directories (data/processed/<base>/) since those are
    intermediate artifacts shared across model runs — they have no result.json
    by design. Only deletes per-model dirs (suffix '__<model>') with no result.
    """
    if not os.path.isdir(PROCESSED_DIR):
        return
    removed = 0
    for name in os.listdir(PROCESSED_DIR):
        full = os.path.join(PROCESSED_DIR, name)
        if not os.path.isdir(full):
            continue
        if "__" not in name:
            continue   # intermediate norm dir, keep
        if os.path.isfile(os.path.join(full, "result.json")):
            continue   # successful run, keep
        try:
            shutil.rmtree(full)
            removed += 1
        except Exception as e:
            print(f"[Startup] Could not GC {name}: {e}", flush=True)
    if removed:
        print(f"[Startup] GC'd {removed} orphan result dir(s)", flush=True)


_gc_orphan_processed_dirs()

# ── Server startup ────────────────────────────────────────────────────────────
import faulthandler as _fh
_fh.enable()   # dump traceback to stderr on SIGSEGV / fatal Python errors

socketserver.ThreadingTCPServer.allow_reuse_address = True

with socketserver.ThreadingTCPServer(("", PORT), RequestHandler) as httpd:
    print(f"\n{'='*50}")
    print(f"  UI Dashboard  →  http://localhost:{PORT}")
    print(f"{'='*50}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
