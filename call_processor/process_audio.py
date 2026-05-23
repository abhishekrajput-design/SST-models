"""
Best-quality audio processing pipeline for desk recordings.

Stages:
  1. src.audio_cleanup.clean_audio — 10-stage desk-recording pipeline:
       two-pass spectral gating · bandpass 85-6500 Hz · VAD · energy gate ·
       clarity gate · per-segment normalization · DRC · crossfade
  2. faster-whisper large-v3 float16 — transcription (VAD-filtered)
  3. pyannote diarization — who spoke when (optional, requires HF_TOKEN)

Usage:
    python process_audio.py --input path/to/recording.mp3
    python process_audio.py --input path/to/recording.mp3 --hf-token YOUR_TOKEN
    python process_audio.py --input path/to/recording.mp3 --no-diarize
    python process_audio.py --input path/to/recording.mp3 --no-enhance
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows UTF-8 console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

FFMPEG = None  # resolved at startup


def find_ffmpeg() -> str:
    """Locate ffmpeg — check PATH first, then common install locations."""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Auto-discover WinGet installs
    winget_base = os.path.join(
        os.path.expanduser("~"),
        "AppData", "Local", "Microsoft", "WinGet", "Packages",
    )
    if os.path.isdir(winget_base):
        for d in os.listdir(winget_base):
            if "FFmpeg" in d:
                for sub in ("ffmpeg-8.1-full_build", "ffmpeg-7.1-full_build"):
                    candidate = os.path.join(winget_base, d, sub, "bin", "ffmpeg.exe")
                    if os.path.isfile(candidate):
                        return candidate
    candidates = ["ffmpeg"]
    for c in candidates:
        try:
            subprocess.run([c, "-version"], capture_output=True, check=True)
            return c
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("ffmpeg not found. Install it or add to PATH.")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def clear_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1: angelina clean_audio — 10-stage desk-recording pipeline
# ══════════════════════════════════════════════════════════════════════════════

def enhance_angelina(input_path: str, output_wav: str) -> bool:
    """
    Run the angelina 10-stage desk-recording cleanup pipeline.
    Outputs 16 kHz mono WAV ready for Whisper.
    Returns True on success.
    """
    try:
        from src.audio_cleanup import clean_audio
        print("  Running angelina pipeline (two-pass noise gate · bandpass · "
              "VAD · energy gate · clarity gate · per-seg norm · DRC)...")
        t0 = time.time()
        clean_audio(input_path, output_wav)
        print(f"  Done in {time.time()-t0:.1f}s → {output_wav}")
        return True
    except Exception as e:
        print(f"  angelina pipeline failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2: FFmpeg loudnorm — auto-normalize to -23 LUFS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_ffmpeg(input_path: str, output_path: str):
    """
    Two-pass FFmpeg loudnorm:
      - Pass 1: measure actual loudness stats
      - Pass 2: apply correction to -23 LUFS, TP=-2 dBFS, LRA=7 LU
    Also converts to 16kHz mono WAV for maximum Whisper compatibility.
    """
    print("  Pass 1: measuring loudness...")
    probe = subprocess.run(
        [FFMPEG, "-y", "-i", input_path,
         "-af", "loudnorm=I=-23:TP=-2:LRA=7:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    )
    # Parse measured stats from stderr
    stderr = probe.stderr
    try:
        start = stderr.rfind("{")
        end   = stderr.rfind("}") + 1
        stats = json.loads(stderr[start:end])
        il   = stats["input_i"]
        itp  = stats["input_tp"]
        ilra = stats["input_lra"]
        ith  = stats["input_thresh"]
        off  = stats["target_offset"]
        loudnorm_filter = (
            f"loudnorm=I=-23:TP=-2:LRA=7:print_format=summary"
            f":measured_I={il}:measured_TP={itp}:measured_LRA={ilra}"
            f":measured_thresh={ith}:offset={off}:linear=true"
        )
    except Exception:
        # Fall back to single-pass if JSON parse fails
        loudnorm_filter = "loudnorm=I=-23:TP=-2:LRA=7:linear=true"

    print("  Pass 2: applying normalization + converting to 16kHz mono...")
    subprocess.run(
        [FFMPEG, "-y", "-i", input_path,
         "-af", loudnorm_filter,
         "-ac", "1", "-ar", "16000",
         "-acodec", "pcm_s16le",
         output_path],
        check=True, capture_output=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3: Whisper large-v3 transcription
# ══════════════════════════════════════════════════════════════════════════════

def transcribe(audio_path: str, language: str, initial_prompt: str) -> list:
    """
    Transcribe with faster-whisper large-v3 float16.
    Returns list of segment dicts with start, end, text.
    """
    import torch
    from faster_whisper import WhisperModel

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    compute     = "float16" if device == "cuda" else "float32"
    model_dir   = os.path.join(PROJECT_ROOT, "models", "faster-whisper")

    print(f"  Loading Whisper large-v3 ({compute} on {device})...")
    model = WhisperModel(
        "large-v3",
        device=device,
        compute_type=compute,
        download_root=model_dir,
    )

    print("  Transcribing (VAD-filtered)...")
    t0 = time.time()
    segs_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,              # skip silence (important for 30-min desk recordings)
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
        initial_prompt=initial_prompt,
        word_timestamps=True,
        temperature=0,
        no_speech_threshold=0.6,
    )

    segments = []
    for seg in segs_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg.text.strip(),
        })

    elapsed = time.time() - t0
    print(f"  Transcribed {len(segments)} segments in {elapsed:.1f}s")
    print(f"  Detected language: {info.language} ({info.language_probability:.0%})")

    del model
    clear_gpu()
    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4: pyannote diarization (optional)
# ══════════════════════════════════════════════════════════════════════════════

def diarize(audio_path: str, hf_token: str) -> list:
    """
    Run pyannote speaker diarization.
    Returns list of {start, end, speaker}.
    """
    import torch
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading pyannote diarization ({device})...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(device)

    import soundfile as sf
    waveform_np, sr = sf.read(audio_path, always_2d=True)
    waveform = __import__("torch").from_numpy(waveform_np.T).float()

    print("  Running diarization...")
    t0 = time.time()
    result = pipeline({"waveform": waveform, "sample_rate": sr})
    elapsed = time.time() - t0
    print(f"  Diarization done in {elapsed:.1f}s")

    segments = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        segments.append({
            "start":   round(turn.start, 3),
            "end":     round(turn.end, 3),
            "speaker": speaker,
        })

    del pipeline, waveform
    clear_gpu()
    return segments


def assign_speakers(transcript_segs: list, diar_segs: list) -> list:
    """Assign the majority diarization speaker label to each transcript segment."""
    from collections import Counter

    def dominant_speaker(start: float, end: float) -> str:
        overlap = Counter()
        for d in diar_segs:
            ov = min(end, d["end"]) - max(start, d["start"])
            if ov > 0:
                overlap[d["speaker"]] += ov
        return overlap.most_common(1)[0][0] if overlap else "UNKNOWN"

    for seg in transcript_segs:
        seg["speaker"] = dominant_speaker(seg["start"], seg["end"])
    return transcript_segs


# ══════════════════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════════════════

def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def save_outputs(segments: list, audio_path: str, output_dir: str, enhanced_path: str):
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "audio_file":   audio_path,
        "enhanced_file": enhanced_path,
        "total_segments": len(segments),
        "segments": segments,
    }

    json_path = os.path.join(output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(output_dir, "transcript.txt")
    lines = [f"TRANSCRIPT — {os.path.basename(audio_path)}", "=" * 70, ""]
    for seg in segments:
        spk = f"[{seg['speaker']}] " if "speaker" in seg else ""
        lines.append(f"[{fmt_time(seg['start'])} → {fmt_time(seg['end'])}] {spk}{seg['text']}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {txt_path}")
    return txt_path


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global FFMPEG

    parser = argparse.ArgumentParser(
        description="Best-quality desk recording transcription with noise suppression"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to audio file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: data/processed/<name>)",
    )
    parser.add_argument("--language",  default="en",   help="Language code (default: en)")
    parser.add_argument("--hf-token",  default=None,   help="HuggingFace token (for diarization)")
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip speaker diarization (just transcribe)",
    )
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="Skip audio enhancement (transcribe raw audio)",
    )
    parser.add_argument(
        "--initial-prompt", default="CarPlanet car dealership agent speaking with customer.",
        help="Domain context for Whisper",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    FFMPEG = find_ffmpeg()

    audio_name = Path(args.input).stem
    output_dir = args.output or os.path.join(PROJECT_ROOT, "data", "processed", audio_name)
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    print(f"\nInput : {args.input}")
    print(f"Output: {output_dir}")

    enhanced_path = args.input  # fallback: use original if enhancement skipped/fails

    # ── Stage 1: angelina 10-stage desk-recording cleanup ────────────────────
    if not args.no_enhance:
        section("Stage 1/3 — angelina: Desk-Recording Audio Cleanup (10 stages)")
        clean_wav = os.path.join(output_dir, f"clean_{audio_name}.wav")
        if os.path.exists(clean_wav):
            print(f"  Already cleaned: {clean_wav}")
            enhanced_path = clean_wav
        else:
            ok = enhance_angelina(args.input, clean_wav)
            if ok:
                enhanced_path = clean_wav
            else:
                # Fallback: FFmpeg loudnorm only
                print("  Falling back to FFmpeg loudnorm...")
                section("Stage 1b/3 — FFmpeg loudnorm: Auto-Normalize to -23 LUFS")
                norm_wav = os.path.join(output_dir, f"norm_{audio_name}.wav")
                if os.path.exists(norm_wav):
                    print(f"  Already normalized: {norm_wav}")
                    enhanced_path = norm_wav
                else:
                    try:
                        normalize_ffmpeg(args.input, norm_wav)
                        enhanced_path = norm_wav
                    except Exception as e:
                        print(f"  Normalization failed: {e} — using raw audio.")
    else:
        print("\n[Skipping audio enhancement]")

    # ── Stage 2: Transcription ───────────────────────────────────────────────
    section("Stage 2/3 — Whisper large-v3: Transcription")
    segments = transcribe(enhanced_path, args.language, args.initial_prompt)

    # ── Stage 3: Diarization ─────────────────────────────────────────────────
    if not args.no_diarize and hf_token:
        section("Stage 3/3 — pyannote 3.1: Speaker Diarization")
        try:
            diar_segs = diarize(enhanced_path, hf_token)
            segments  = assign_speakers(segments, diar_segs)
            print(f"  Assigned speakers to {len(segments)} transcript segments")
        except Exception as e:
            print(f"  Diarization failed: {e}")
            print("  Continuing without speaker labels.")
    elif not hf_token:
        print("\n[Skipping diarization — no HF_TOKEN set]")
        print("  Set HF_TOKEN env var or pass --hf-token to enable speaker labeling.")
    else:
        print("\n[Diarization skipped (--no-diarize)]")

    # ── Save outputs ─────────────────────────────────────────────────────────
    section("Results")
    txt_path = save_outputs(segments, args.input, output_dir, enhanced_path)

    # Print transcript preview
    print(f"\n  {'─'*58}")
    print(f"  TRANSCRIPT PREVIEW (first 20 segments)")
    print(f"  {'─'*58}")
    for seg in segments[:20]:
        spk = f"[{seg.get('speaker', '')}] " if "speaker" in seg else ""
        print(f"  [{fmt_time(seg['start'])} → {fmt_time(seg['end'])}] {spk}{seg['text']}")
    if len(segments) > 20:
        print(f"\n  ... and {len(segments) - 20} more segments.")

    total = time.time() - total_start
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Full transcript: {txt_path}")


if __name__ == "__main__":
    main()
