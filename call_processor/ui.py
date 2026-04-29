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

# FFmpeg PATH on Windows (WinGet install). On Linux the system ffmpeg is used.
_FFMPEG_BIN = r"C:\Users\abhis\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
_ENV = os.environ.copy()
if sys.platform == "win32" and os.path.isdir(_FFMPEG_BIN):
    if _FFMPEG_BIN not in _ENV.get("PATH", ""):
        _ENV["PATH"] = _FFMPEG_BIN + os.pathsep + _ENV.get("PATH", "")

# FFmpeg filter chain for call-center audio with background noise.
# silenceremove intentionally removed — it cuts quiet/distant voices near the
# noise floor. dynaudnorm boosts quiet segments locally so Whisper/Parakeet
# can hear them without over-amplifying loud peaks.
AUDIO_FILTER = (
    "aresample=44100,"                        # upsample first — loudnorm needs ≥44.1k
    "highpass=f=80,"                          # strip low-freq HVAC/rumble
    "afftdn=nf=-25:nt=w,"                     # FFmpeg spectral denoiser pass
    "loudnorm=I=-16:TP=-1.5:LRA=11,"          # bring quiet phone audio up to standard level
    "dynaudnorm=p=0.9:m=100:s=5:g=15"         # boost quiet passages locally
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

# ── Agent enrollment status ───────────────────────────────────────────────────
_enroll_status: dict = {"running": False, "done": False, "error": None, "message": ""}
_enroll_lock   = threading.Lock()
# Directory of known-agent recordings used for voice enrollment
AGENT_RECORDINGS_DIR = r"C:\Users\abhis\Desktop\SST-models\Agents-recoding\zak_recodings"


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

def _run_ffmpeg(cmd: list, timeout: int = 180):
    """Run an FFmpeg command with timeout + explicit env so it never hangs silently."""
    result = subprocess.run(
        cmd, env=_ENV, capture_output=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode(errors='replace')[-500:]}")


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
#  Audio trimming — remove gaps where no speech was detected
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  Whisper-only transcription (fallback when HF_TOKEN not set)
# ══════════════════════════════════════════════════════════════════════════════

def _transcribe_inline(audio_path: str, whisper_model: str = "whisper-large-v3-turbo",
                       original_path: str = "") -> str:
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

    # CUDA Parakeet can abort the Python process on this Windows workstation.
    # Parakeet is forced to CPU below, so keep the selected model and surface
    # long-call runtime honestly instead of silently switching to Whisper.
    if whisper_model == "parakeet-tdt-0.6b-v3":
        dur_s = _audio_duration_s(audio_path)
        if dur_s > 600:
            print(
                f"[UI] Parakeet CPU mode selected for long audio ({dur_s:.0f}s).",
                flush=True,
            )
            _set_status(
                2,
                "Transcription",
                f"Parakeet CPU mode selected for {dur_s:.0f}s audio; this can take several minutes...",
            )

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

    # ── Load transcriber (shared for both channels) ───────────────────────────
    _set_status(2, "Transcription", f"Loading {whisper_model}...")
    print(f"[UI] Loading transcriber: {whisper_model}")
    transcriber_device = "cpu" if whisper_model == "parakeet-tdt-0.6b-v3" else "cuda"
    if transcriber_device == "cpu" and whisper_model == "parakeet-tdt-0.6b-v3":
        print("[UI] Parakeet uses CPU mode for stability.", flush=True)
    transcriber = get_transcriber(whisper_model, device=transcriber_device)
    transcriber.load()

    def _retry_empty_transcript(
        current_segments: list,
        wav_path: str,
        reason: str,
    ) -> list:
        nonlocal transcriber, whisper_model
        if current_segments or whisper_model in ("whisper-large-v3-turbo", "distil-whisper-large-v3.5"):
            return current_segments
        fallback_model = "whisper-large-v3-turbo"
        _set_status(3, "Transcription", f"{reason}; retrying with {fallback_model}...")
        print(f"[UI] {reason}; retrying with {fallback_model}", flush=True)
        try:
            transcriber.unload()
        except Exception:
            pass
        whisper_model = fallback_model
        retry_device = "cpu" if whisper_model == "parakeet-tdt-0.6b-v3" else "cuda"
        transcriber = get_transcriber(whisper_model, device=retry_device)
        transcriber.load()
        return transcriber.transcribe(wav_path, language="en")

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

    _set_status(3, "Transcription", f"Transcribing with {whisper_model}...")
    print(f"[UI] Transcribing with {whisper_model}...")
    segments = transcriber.transcribe(norm_wav, language="en")
    segments = _retry_empty_transcript(
        segments,
        norm_wav,
        "No transcript segments returned",
    )

    agent_time = customer_time = 0.0
    agent_turns = customer_turns = 0

    # Free GPU memory before diarization — embeddings run on CPU (force_cpu=True)
    # but leftover VRAM from Parakeet can cause fragmentation issues.
    try:
        import torch as _torch, gc as _gc
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
    except Exception:
        pass

    _set_status(3, "Transcription", "Identifying speakers (voiceprint matching)...")
    try:
        from src.diar_multi import diarize_multi
        print("[UI] Running voiceprint-first multi-speaker diarization...", flush=True)
        diar_result = diarize_multi(segments, norm_wav, force_cpu=True)
        segments     = diar_result["segments"]
        agent_name_id = diar_result.get("agent_name", "Unknown Agent")
        agent_sim     = diar_result.get("agent_similarity", 0.0)
        backend_dim   = diar_result.get("matched_backend_dim")
        print(
            f"[UI] Agent identified: {agent_name_id} "
            f"(cosine={agent_sim:.3f}, dim={backend_dim})",
            flush=True,
        )
        print(
            f"[UI] Speaker mode: {diar_result.get('speaker_mode', 'unknown')}",
            flush=True,
        )
        print(f"[UI] Speakers: {list(diar_result.get('per_speaker', {}).keys())}", flush=True)

        for seg in segments:
            dur = float(seg["end"]) - float(seg["start"])
            if seg.get("identified_speaker") == "AGENT":
                agent_time  += dur;  agent_turns  += 1
            else:
                customer_time += dur; customer_turns += 1

        diarization_applied = True
    except Exception as _diar_err:
        print(f"[UI] Diarization skipped ({repr(_diar_err)}) — single-speaker mode", flush=True)
        for seg in segments:
            seg["speaker"] = seg.get("speaker") or "SPEAKER_99"
            seg["identified_speaker"] = "CUSTOMER"
            seg.pop("agent_name", None)
            seg["display_speaker"] = "Unknown"

    elapsed = round(time.time() - t0, 2)

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
    first_agent_words    = next((s["text"].strip() for s in segments if s.get("identified_speaker") == "AGENT"),    "")
    first_customer_words = next((s["text"].strip() for s in segments if s.get("identified_speaker") == "CUSTOMER"), "")

    if diarization_applied:
        speaker_stats = {
            "AGENT": {
                "time_s": round(agent_time, 1), "turns": agent_turns,
                "first_words": first_agent_words[:120],
            },
            "CUSTOMER": {
                "time_s": round(customer_time, 1), "turns": customer_turns,
                "first_words": first_customer_words[:120],
            },
        }
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
    trim_ok = _trim_to_speech(audio_path, segments, trimmed_path,
                               pad_s=1.0, merge_gap_s=5.0)
    trimmed_audio_file = trimmed_path.replace("\\", "/") if trim_ok else None
    if not trim_ok:
        print("[UI] Trim skipped — using original enhanced audio.", flush=True)

    # Collect identified agent name (set on segments during multi-agent ID)
    _identified_agent = identified_agent_name

    result = {
        "audio_file":               audio_path.replace("\\", "/"),
        "trimmed_audio_file":       trimmed_audio_file,
        "model":                    whisper_model,
        "requested_model":          requested_model,
        "processed_at":             datetime.utcnow().isoformat() + "Z",
        "processing_time_seconds":  elapsed,
        "total_segments":           len(segments),
        "segments":                 segments,
        "transcription_json":       transcription_json,
        "diarization":              "diar_multi_voiceprint" if diarization_applied else "none",
        "speaker_stats":            speaker_stats,
        "identified_agent":         _identified_agent,
        "speaker_id_backend_dim":   diar_result.get("matched_backend_dim"),
        "voiceprint_dims":          diar_result.get("voiceprint_dims", {}),
        "speaker_id_warning":       diar_result.get("warning"),
        "speaker_id_mode":          diar_result.get("speaker_mode"),
        "speaker_id_cluster_report": diar_result.get("cluster_report", {}),
        "note": (
            f"Requested {requested_model}; transcribed with {whisper_model}"
            if requested_model != whisper_model
            else f"Transcribed with {whisper_model}"
        ),
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
            # Replaces noisereduce + angelina + MetricGAN+ chain which was
            # over-processing the audio and causing Whisper hallucinations.
            _set_status(0, "Enhancing Audio", "[2/2] DeepFilterNet3 · neural denoising...")
            print("[UI] Stage 0b: DeepFilterNet3...")
            pipeline_audio = paths["ffmpeg"]   # fallback
            try:
                _enhance_deepfilternet(paths["ffmpeg"], paths["deepfilter"])
                enhancement_paths["deepfilter"] = paths["deepfilter"]
                pipeline_audio = paths["deepfilter"]
                print("[UI] DeepFilterNet3 done.")
            except ImportError:
                print("[UI] deepfilternet not installed — using FFmpeg output.")
            except Exception as e:
                print(f"[UI] DeepFilterNet3 failed: {e} — using FFmpeg output.")

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
            result_id = _transcribe_inline(pipeline_audio, whisper_model,
                                           original_path=upload_path)
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

        with _status_lock:
            _status.update(done=True, running=False, stage_num=4,
                           stage="Complete", message="Done! Results saved.",
                           result_id=result_id)
        print(f"[UI] Pipeline complete. Result: {result_id}")

    except BaseException as e:
        import traceback as _tb
        print(f"[UI] Pipeline error: {e}", flush=True)
        _tb.print_exc()
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
#  Agent Enrollment  (background thread)
# ══════════════════════════════════════════════════════════════════════════════

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
                {"model": "parakeet-tdt-0.6b-v3",      "label": "NVIDIA Parakeet TDT v3",   "type": "Local CPU",  "speed_s": 45,   "segments": 126,  "notes": "CPU mode for stable UI runs; long audio may be slow", "rank": 5, "wer": "~5.5%", "status": "ok"},
                {"model": "cohere-transcribe-03-2026", "label": "Cohere Transcribe 03-2026","type": "Local GPU",  "speed_s": 52,   "segments": 60,   "notes": "Lowest WER on leaderboard",      "rank": 6, "wer": "5.42%", "status": "ok"},
                {"model": "deepgram-nova-2-phonecall", "label": "Deepgram Nova-2 Phone",    "type": "Cloud API",  "speed_s": 6,    "segments": 33,   "notes": "Optimised for phone call audio", "rank": 7, "wer": "~9%",   "status": "ok"},
                {"model": "deepgram-nova-2-meeting",   "label": "Deepgram Nova-2 Meeting",  "type": "Cloud API",  "speed_s": 8,    "segments": 51,   "notes": "Multi-speaker meetings",         "rank": 8, "wer": "~9%",   "status": "ok"},
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
            # Send the response BEFORE starting the pipeline thread so the client
            # gets its confirmation even when the server becomes CPU/GPU-bound.
            self._json({"status": "started", "filename": filename, "model": model})
            threading.Thread(target=_run_pipeline, args=(upload_path, filename, model),
                             daemon=True).start()
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
