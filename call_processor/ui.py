import os
import re
import sys
import gc
import json
import shutil
import subprocess
import threading
import http.server
import socketserver
from urllib.parse import urlparse, unquote, parse_qs
from pathlib import Path

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

# FFmpeg filter: aggressive desk-recording voice enhancement chain
# 1. highpass 100 Hz   → remove AC rumble / desk thumps
# 2. lowpass 7.5 kHz   → remove hiss (voice sits <6 kHz anyway)
# 3. afftdn  nr=25     → aggressive spectral noise reduction (−25 dB floor)
# 4. EQ −4 dB @ 300 Hz → tame muddiness
# 5. EQ +5 dB @ 3 kHz  → voice presence / intelligibility
# 6. EQ +4 dB @ 5 kHz  → consonant clarity ("s", "t", "sh")
# 7. compand           → heavy compression = consistent perceived loudness
# 8. volume 6x         → loud boost (≈ +15 dB)
# 9. loudnorm −14 LUFS → broadcast-loud target (most streaming platforms)
# 10. alimiter         → final peak limiter so nothing clips
AUDIO_FILTER = (
    "highpass=f=100,"
    "lowpass=f=7500,"
    "afftdn=nr=25:nf=-25:nt=w,"
    "equalizer=f=300:t=q:w=2:g=-4,"
    "equalizer=f=3000:t=q:w=1.5:g=5,"
    "equalizer=f=5000:t=q:w=1.5:g=4,"
    "compand=attacks=0:points=-80/-80|-40/-20|-20/-10|-5/-5|0/-3:soft-knee=6,"
    "volume=6.0,"
    "loudnorm=I=-14:TP=-1.5:LRA=5,"
    "alimiter=level_in=1:level_out=0.97:limit=0.97"
)

# ── Pipeline status (shared) ──────────────────────────────────────────────────
_status = {
    "running": False,
    "stage_num": 0,
    "stage": "Idle",
    "message": "",
    "done": False,
    "error": None,
    "result_id": None,
}
_status_lock = threading.Lock()

# ── Enhancement job status (separate from main pipeline) ─────────────────────
_enhance_status: dict = {}          # call_id -> {"running", "done", "error", "paths"}
_enhance_lock = threading.Lock()


def _set_status(stage_num: int, stage: str, message: str):
    with _status_lock:
        _status["stage_num"] = stage_num
        _status["stage"]     = stage
        _status["message"]   = message


# ══════════════════════════════════════════════════════════════════════════════
#  Audio Enhancement Pipelines
#  All clip to max_seconds (default 300 = 5 min) so they finish fast.
#  All produce 44.1 kHz / 128 kbps / mono MP3 for browser playback.
# ══════════════════════════════════════════════════════════════════════════════

def _to_wav(input_path: str, max_seconds: int = None) -> str:
    """Convert any audio → 16 kHz mono WAV (optionally clipped)."""
    wav = input_path + f"_tmp{os.getpid()}.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-ac", "1", "-ar", "16000", wav]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav


def _wav_to_mp3(wav_path: str, out_mp3: str):
    """44.1 kHz / 128 kbps / mono MP3 — browser-compatible."""
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-ac", "1", "-ar", "44100", "-b:a", "128k", out_mp3,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Pipeline 1: FFmpeg (highpass + afftdn + volume + loudnorm) ───────────────
def _enhance_ffmpeg(input_path: str, output_path: str, max_seconds: int = 300):
    """FFmpeg DSP chain — fast, always works, full audio for main pipeline."""
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-af", AUDIO_FILTER, "-ac", "1", "-ar", "44100", "-b:a", "128k", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
def _enhance_deepfilternet(input_path: str, output_path: str, max_seconds: int = 300):
    """
    DeepFilterNet3 neural noise suppression.
    pip install deepfilternet
    Uses soundfile instead of torchaudio.load (torchaudio 2.x compat fix).
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

        enhanced = enhance(model, df_state, audio)
        enh_np = enhanced.cpu().numpy().squeeze(0)
        enh_np = np.clip(enh_np * 3.0, -0.98, 0.98)

        sf.write(out_wav, enh_np, target_sr, subtype="PCM_16")
        _wav_to_mp3(out_wav, output_path)

        del model, df_state, audio, enhanced
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


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers for path derivation
# ══════════════════════════════════════════════════════════════════════════════

def _derive_paths(original_path: str) -> dict:
    """Return expected file paths for all 4 enhancement variants."""
    d = os.path.dirname(original_path)
    fname = os.path.basename(original_path)
    return {
        "ffmpeg":      os.path.join(d, f"enhanced_{fname}").replace("\\", "/"),
        "noisereduce": os.path.join(d, f"nr_{fname}").replace("\\", "/"),
        "deepfilter":  os.path.join(d, f"df_{fname}").replace("\\", "/"),
        "metricgan":   os.path.join(d, f"mg_{fname}").replace("\\", "/"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Whisper-only transcription (fallback when HF_TOKEN not set)
# ══════════════════════════════════════════════════════════════════════════════

def _transcribe_inline(audio_path: str, whisper_model: str = "whisper-large-v3-turbo") -> str:
    """
    Run any registered transcriber in-process (Whisper, Cohere, Parakeet, Qwen3, VibeVoice).
    Returns result_id (basename of FFmpeg-enhanced file).
    Writes result.json to data/processed/<result_id>/.
    """
    import time
    from datetime import datetime
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.transcribers import get_transcriber

    base     = os.path.splitext(os.path.basename(audio_path))[0]
    # Use a per-model directory so each model run is stored separately
    dir_name = f"{base}__{whisper_model}"
    out_dir  = os.path.join("data", "processed", dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Normalize to 16 kHz mono WAV once (shared across models for the same audio)
    norm_dir = os.path.join("data", "processed", base)
    os.makedirs(norm_dir, exist_ok=True)
    norm_wav = os.path.join(norm_dir, f"norm_{base}.wav")
    if not os.path.exists(norm_wav):
        _set_status(1, "Transcription", "Normalizing audio to 16 kHz mono...")
        print("[UI] Normalizing audio...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ar", "16000", "-ac", "1",
             "-af", "loudnorm=I=-23:TP=-1:LRA=7",
             norm_wav],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    _set_status(2, "Transcription", f"Loading {whisper_model}...")
    print(f"[UI] Loading transcriber: {whisper_model}")
    transcriber = get_transcriber(whisper_model, device="cuda")
    transcriber.load()

    _set_status(3, "Transcription", f"Transcribing with {whisper_model}...")
    print(f"[UI] Transcribing with {whisper_model}...")
    t0 = time.time()
    segments = transcriber.transcribe(norm_wav, language="en")
    elapsed = round(time.time() - t0, 2)

    transcriber.unload()

    # Force-free GPU/CPU memory so the next model starts with a clean slate
    try:
        import torch, gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    result = {
        "audio_file":               audio_path.replace("\\", "/"),
        "model":                    whisper_model,
        "processed_at":             datetime.utcnow().isoformat() + "Z",
        "processing_time_seconds":  elapsed,
        "total_segments":           len(segments),
        "segments":                 segments,
        "note": f"Transcribed with {whisper_model} (no diarization unless model provides it)",
    }
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


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline thread
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(upload_path: str, filename: str, whisper_model: str = "large-v3"):
    """
    Stage 0a — FFmpeg enhancement  (full audio, used by AI pipeline)
    Stage 0b — noisereduce          (first 5 min)
    Stage 0c — DeepFilterNet3       (first 5 min)
    Stage 0d — SpeechBrain MetricGAN+ (first 5 min)
    Stages 1-3 — diarize → speaker ID → transcribe
    Patch result.json with all enhancement paths.
    """
    paths = _derive_paths(upload_path)
    os.makedirs("data/raw_calls", exist_ok=True)

    enhancement_paths: dict = {}

    with _status_lock:
        _status.update(running=True, done=False, error=None, result_id=None)

    try:
        # If the file is already an enhanced output, skip re-enhancement to
        # avoid double-processing (which truncates the audio to garbage).
        already_enhanced = os.path.basename(upload_path).startswith("enhanced_")

        if already_enhanced:
            _set_status(0, "Enhancing Audio", "File already enhanced — skipping re-processing...")
            print("[UI] Skipping enhancement: file already starts with 'enhanced_'.")
            paths["ffmpeg"] = upload_path.replace("\\", "/")
            enhancement_paths["ffmpeg"] = upload_path.replace("\\", "/")
        else:
            # ── 0a FFmpeg (full audio, no clip — used by pipeline) ───────────
            _set_status(0, "Enhancing Audio", "[1/4] FFmpeg · highpass · afftdn · 3× volume...")
            print("[UI] Stage 0a: FFmpeg...")
            try:
                _enhance_ffmpeg(upload_path, paths["ffmpeg"], max_seconds=None)
                enhancement_paths["ffmpeg"] = paths["ffmpeg"]
                print("[UI] FFmpeg done.")
            except Exception as e:
                print(f"[UI] FFmpeg failed: {e}")
                paths["ffmpeg"] = upload_path
                enhancement_paths["ffmpeg"] = upload_path.replace("\\", "/")

            # ── 0b noisereduce (5 min preview) ───────────────────────────────
            _set_status(0, "Enhancing Audio", "[2/4] noisereduce · spectral gating (5-min)...")
            print("[UI] Stage 0b: noisereduce...")
            try:
                _enhance_noisereduce(upload_path, paths["noisereduce"], max_seconds=300)
                enhancement_paths["noisereduce"] = paths["noisereduce"]
                print("[UI] noisereduce done.")
            except ImportError:
                print("[UI] noisereduce not installed.")
            except Exception as e:
                print(f"[UI] noisereduce failed: {e}")

            # ── 0c angelina desk-recording cleanup (5 min preview) ───────────
            _set_status(0, "Enhancing Audio", "[3/4] angelina · 10-stage desk-recording cleanup (5-min)...")
            print("[UI] Stage 0c: angelina cleanup...")
            try:
                _enhance_angelina(upload_path, paths["deepfilter"], max_seconds=300)
                enhancement_paths["deepfilter"] = paths["deepfilter"]
                print("[UI] angelina done.")
            except ImportError:
                print("[UI] librosa not installed — skipping angelina.")
            except Exception as e:
                print(f"[UI] angelina failed: {e}")

            # ── 0d SpeechBrain MetricGAN+ (5 min preview) ────────────────────
            _set_status(0, "Enhancing Audio", "[4/4] SpeechBrain MetricGAN+ · PESQ-optimised (5-min)...")
            print("[UI] Stage 0d: SpeechBrain MetricGAN+...")
            try:
                _enhance_metricgan(upload_path, paths["metricgan"], max_seconds=300)
                enhancement_paths["metricgan"] = paths["metricgan"]
                print("[UI] MetricGAN+ done.")
            except ImportError:
                print("[UI] speechbrain not installed.")
            except Exception as e:
                print(f"[UI] MetricGAN+ failed: {e}")

        # ── Stages 1-3: AI pipeline (or transcription-only fallback) ────────
        pipeline_audio = paths["ffmpeg"]

        # run_e2e.py (pyannote diarization) only used when enrolled agent
        # embeddings exist. Without enrollment, all models use inline path.
        _WHISPER_MODELS = {
            "whisper-large-v3-turbo", "whisper-large-v3",
            "large-v3-turbo", "large-v3",
            "distil-large-v3", "distil-large-v3.5",
        }
        embeddings_exist = os.path.isfile(
            os.path.join("data", "embeddings", "agent_embeddings.pkl")
        )
        use_inline = (
            (not HF_TOKEN)
            or (whisper_model not in _WHISPER_MODELS)
            or (not embeddings_exist)
        )

        if use_inline:
            label = whisper_model if whisper_model in _WHISPER_MODELS else whisper_model
            _set_status(1, "Transcription", f"Transcribing with {label}...")
            print(f"[UI] Inline transcription mode ({label}).")
            result_id = _transcribe_inline(pipeline_audio, whisper_model)
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
                rdata["enhancements"] = enhancement_paths
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(rdata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[UI] result.json patch failed: {e}")

        with _status_lock:
            _status.update(done=True, running=False, stage_num=4,
                           stage="Complete", message="Done! Results saved.",
                           result_id=result_id)
        print(f"[UI] Pipeline complete. Result: {result_id}")

    except Exception as e:
        print(f"[UI] Pipeline error: {e}")
        with _status_lock:
            _status.update(error=str(e), running=False)


# ══════════════════════════════════════════════════════════════════════════════
#  On-demand enhancement for existing calls  (async background thread)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP Request Handler
# ══════════════════════════════════════════════════════════════════════════════

class RequestHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        if "favicon" not in self.path:
            super().log_message(format, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        # /api/status
        if path == "/api/status":
            with _status_lock:
                self._json(dict(_status))
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
                        # Derive original file path (strip enhanced_ prefix)
                        orig_file = audio_file.replace("enhanced_", "").replace("\\", "/")
                        # Get original audio metadata via ffprobe
                        orig_meta = {}
                        if orig_file and os.path.isfile(orig_file):
                            try:
                                import subprocess as _sp
                                r = _sp.run(
                                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                                     "-show_format", "-show_streams", orig_file],
                                    capture_output=True, text=True, timeout=10
                                )
                                if r.returncode == 0:
                                    fd = json.loads(r.stdout)
                                    fmt = fd.get("format", {})
                                    streams = fd.get("streams", [{}])
                                    orig_meta = {
                                        "duration_s": round(float(fmt.get("duration", 0))),
                                        "bitrate_kbps": int(fmt.get("bit_rate", 0)) // 1000,
                                        "size_mb": round(int(fmt.get("size", 0)) / 1024 / 1024, 1),
                                        "sample_rate": streams[0].get("sample_rate", "?"),
                                        "channels": streams[0].get("channels", 1),
                                    }
                            except Exception:
                                pass
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
                {"model": "deepgram-nova-3",              "label": "Deepgram Nova-3",             "type": "Cloud API",  "speed_s": 5,    "segments": None, "notes": "Best accuracy + diarization",       "rank": 1,  "wer": "~7%",   "status": "ok"},
                {"model": "groq-whisper-large-v3-turbo",  "label": "Groq Whisper v3-Turbo",       "type": "Cloud API",  "speed_s": 8,    "segments": None, "notes": "Ultra-fast cloud Whisper, ~5-10s",   "rank": 2,  "wer": "~8.4%", "status": "ok"},
                {"model": "whisper-large-v3-turbo",       "label": "Whisper Large-v3-Turbo",      "type": "Local GPU",  "speed_s": 52,   "segments": 555,  "notes": "Best local model, word timestamps",  "rank": 3,  "wer": "~8.4%", "status": "ok"},
                {"model": "parakeet-tdt-0.6b-v3",         "label": "NVIDIA Parakeet TDT v3",      "type": "Local GPU",  "speed_s": 94,   "segments": 300,  "notes": "Fastest local, English only",       "rank": 4,  "wer": "~5.5%", "status": "ok"},
                {"model": "cohere-transcribe-03-2026",    "label": "Cohere Transcribe 03-2026",   "type": "Local GPU",  "speed_s": 300,  "segments": None, "notes": "Best WER on leaderboard",           "rank": 5,  "wer": "5.42%", "status": "ok"},
                {"model": "canary-qwen-2.5b",             "label": "NVIDIA Canary-Qwen 2.5B",     "type": "Local GPU",  "speed_s": 1500, "segments": None, "notes": "Multilingual, NeMo SALM",           "rank": 6,  "wer": "~6%",   "status": "ok"},
                {"model": "deepgram-nova-2-phonecall",    "label": "Deepgram Nova-2 Phone",       "type": "Cloud API",  "speed_s": 5,    "segments": None, "notes": "Optimised for phone call audio",    "rank": 7,  "wer": "~9%",   "status": "ok"},
                {"model": "deepgram-nova-2-meeting",      "label": "Deepgram Nova-2 Meeting",     "type": "Cloud API",  "speed_s": 5,    "segments": None, "notes": "Multi-speaker meetings",            "rank": 8,  "wer": "~9%",   "status": "ok"},
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

        # Audio files (Range support for seeking)
        audio_exts   = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
        decoded_path = unquote(path)
        if any(decoded_path.lower().endswith(ext) for ext in audio_exts):
            self._serve_audio(decoded_path)
            return

        if path == "/":
            self.path = "/index.html"
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
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            with _status_lock:
                busy = _status["running"]
            if busy:
                self._json({"status": "busy", "message": "Pipeline already running"}, 409)
                return
            query    = parse_qs(parsed.query)
            filename = query.get("filename", ["upload.mp3"])[0]
            model    = query.get("model",    ["whisper-large-v3-turbo"])[0]
            os.makedirs("data/raw_calls", exist_ok=True)
            upload_path = os.path.join("data", "raw_calls", filename)
            n = int(self.headers.get("Content-Length", 0))
            if n > 0:
                with open(upload_path, "wb") as f:
                    f.write(self.rfile.read(n))
            # Clear any stale error from a previous run
            with _status_lock:
                _status.update(running=False, done=False, error=None, result_id=None,
                               stage_num=0, stage="Idle", message="")
            threading.Thread(target=_run_pipeline, args=(upload_path, filename, model),
                             daemon=True).start()
            self._json({"status": "started", "filename": filename, "model": model})
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


# ── Server startup ────────────────────────────────────────────────────────────
socketserver.ThreadingTCPServer.allow_reuse_address = True

with socketserver.ThreadingTCPServer(("", PORT), RequestHandler) as httpd:
    print(f"\n{'='*50}")
    print(f"  UI Dashboard  →  http://localhost:{PORT}")
    print(f"{'='*50}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
