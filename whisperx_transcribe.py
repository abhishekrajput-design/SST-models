import gc
import os
import logging
import subprocess
import threading

device = "cuda"
# device = "cpu"
compute_type = "float16"  # float16 reduces hallucination on quiet/telephone audio vs int8

SILENCE_PAD_SECONDS = 0.5  # silence prepended before audio for VAD warmup

# Initial prompt — correct brand names and domain terms fed to Whisper before transcription
# to bias it toward the right spellings from the start.
INITIAL_PROMPT = "CarPlanet, CarPlanet dealership, car dealer, vehicle."

# Hotwords — comma-separated terms that get a log-probability boost during beam search.
# Add any domain-specific words that Whisper keeps mis-spelling.
HOTWORDS = "CarPlanet"

# Switch between diarization backends — "nemo" or "pyannote"
DIARIZER_BACKEND = "nemo"

# ---------------------------------------------------------------------------
# Model cache — ASR and alignment models are loaded once and kept in VRAM.
# Keyed by (model_name, language) for ASR, language for alignment.
# Locks ensure only one thread loads a given model at a time.
# ---------------------------------------------------------------------------
_asr_lock   = threading.Lock()
_asr_cache  = {}   # (model_name, language) → whisperx FasterWhisperPipeline

_align_lock  = threading.Lock()
_align_cache = {}  # language → (model_a, metadata)

NEMO_DEVICE = "cuda"  # NeMo runs on GPU; serialized via _nemo_sem to prevent concurrent OOM
_nemo_sem = threading.Semaphore(1)  # only 1 NeMo instance on GPU at a time


def _diarize_with_nemo(audio_file: str):
    """
    Diarize audio using NeMo MSDD telephonic model.
    Designed for 8kHz telephone recordings — significantly better speaker
    separation than pyannote on narrow-band audio.

    Returns a pandas DataFrame with columns [start, end, speaker],
    identical in shape to what whisperx.diarize.DiarizationPipeline returns,
    so it can be passed directly to whisperx.assign_word_speakers.
    """
    import json
    import tempfile
    import urllib.request
    import pandas as pd
    from omegaconf import OmegaConf
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer

    # Use the official NeMo telephonic diarization config to avoid missing-key errors.
    # Cache it next to this file so it's only downloaded once.
    NEMO_CFG_URL  = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    NEMO_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diar_infer_telephonic.yaml")

    if not os.path.isfile(NEMO_CFG_PATH):
        logging.getLogger(__name__).info("Downloading NeMo telephonic diarization config...")
        urllib.request.urlretrieve(NEMO_CFG_URL, NEMO_CFG_PATH)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # NeMo requires a JSONL manifest pointing to the audio file
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write(json.dumps({
                "audio_filepath": os.path.abspath(audio_file),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }) + "\n")

        import torch
        from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TimeoutError

        NEMO_TIMEOUT_SECONDS = 1200  # 20 min max — covers 30min+ calls; genuine hangs caught after this

        def _run_nemo(run_device):
            cfg = OmegaConf.load(NEMO_CFG_PATH)
            cfg.device                                              = run_device
            cfg.diarizer.manifest_filepath                         = manifest_path
            cfg.diarizer.out_dir                                   = tmp_dir
            cfg.diarizer.clustering.parameters.max_num_speakers    = 4  # agent + customer + up to 2 on transfers/forwarding
            diarizer = NeuralDiarizer(cfg=cfg).to(run_device)
            with _TPE(max_workers=1) as _nemo_pool:
                f = _nemo_pool.submit(diarizer.diarize)
                try:
                    f.result(timeout=NEMO_TIMEOUT_SECONDS)
                except _TimeoutError:
                    raise RuntimeError(f"NeMo diarization timed out after {NEMO_TIMEOUT_SECONDS}s")

        try:
            with _nemo_sem:  # serialize GPU access — prevents OOM when 2 audios run concurrently
                _run_nemo(NEMO_DEVICE)
        except torch.cuda.OutOfMemoryError:
            logging.getLogger(__name__).warning(
                "NeMo GPU OOM for %s — retrying on CPU", os.path.basename(audio_file)
            )
            print(f"[NeMo] GPU OOM — retrying on CPU for {os.path.basename(audio_file)}")
            torch.cuda.empty_cache()
            _run_nemo("cpu")

        # Parse the output RTTM file
        # RTTM format: SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        audio_stem = os.path.splitext(os.path.basename(audio_file))[0]
        rttm_path  = os.path.join(tmp_dir, "pred_rttms", f"{audio_stem}.rttm")
        rows = []
        with open(rttm_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == "SPEAKER":
                    start    = float(parts[3])
                    duration = float(parts[4])
                    speaker  = parts[7]
                    rows.append({"start": start, "end": start + duration, "speaker": speaker})
        rows.sort(key=lambda x: x["start"])
        return pd.DataFrame(rows)


def _ts_from_seconds(secs: float) -> str:
    """Convert float seconds to HH:MM:SS.mmm timestamp string."""
    secs = max(0.0, secs)
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _prepend_silence(audio_file: str) -> str:
    """
    Resample audio to 16kHz and prepend SILENCE_PAD_SECONDS of silence.

    Resampling to 16kHz is critical: WhisperX's VAD (Silero) is trained on 16kHz audio.
    Telephone recordings (8kHz) fed directly cause the VAD to miss speech at the start
    because the frequency characteristics don't match what the model expects.
    Doing the resample here — before WhisperX sees the file — ensures the VAD runs
    against properly-formatted audio from the very first frame.

    Returns path to the padded temp file — caller is responsible for deleting it.
    """
    padded_path = audio_file + "_padded.wav"
    # Pipeline (single input — avoids concat sample-rate mismatch issues):
    #   1. aresample=16000       — upsample 8kHz telephone audio to 16kHz (Silero VAD expects 16kHz)
    #   2. adelay=<ms>:all=1    — prepend silence pad for VAD warmup (applied to all channels)
    #   -ac 1                   — convert to mono
    #   -ar 16000               — force output sample rate
    delay_ms = int(SILENCE_PAD_SECONDS * 1000)
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", audio_file,
        "-af", f"aresample=16000,adelay={delay_ms}:all=1",
        "-ar", "16000",
        "-ac", "1",
        padded_path
    ], capture_output=True)
    if result.returncode != 0:
        logging.getLogger(__name__).error("ffmpeg failed:\n%s", result.stderr.decode())
        raise RuntimeError("ffmpeg preprocessing failed")
    return padded_path


# audio_file must be mp3 or wav
def transcribe(audio_file, model_needed, language=None):
    import whisperx
    import whisperx.diarize
    import torch

    # pyannote diarizer (old) — kept for easy rollback, set DIARIZER_BACKEND = "pyannote" to re-enable
    if DIARIZER_BACKEND == "pyannote":
        diarize_model = whisperx.diarize.DiarizationPipeline(token=os.environ['HF_TOKEN'], device=device)

    logger = logging.getLogger('whisperx.' + __name__)

    batch_size = 8  # 16GB VRAM; kept at 8 to leave headroom for concurrent transcriptions
    asr_options = {
        'beam_size': 5,
        'patience': 1.0,
        'length_penalty': 1.0,
        'temperatures': (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        'compression_ratio_threshold': 2.4,  # default; 1.8 caused too many temperature fallbacks on telephone audio
        'log_prob_threshold': -1.0,          # default; -0.5 was too tight and triggered noisy high-temp retries
        'no_speech_threshold': 0.6,           # lower = less aggressive; let alignment model handle borderline segments
        'condition_on_previous_text': False,
        'initial_prompt': None,
        'suppress_tokens': [-1],
        'suppress_numerals': False,
        'repetition_penalty': 1,
        'prompt_reset_on_temperature': 0.5,
        'no_repeat_ngram_size': 0,
        'multilingual': False,       # bool in faster-whisper 1.x (was int 0 in older versions)
        'max_new_tokens': None,
        'clip_timestamps': "0",      # string in faster-whisper 1.x (was int 0 in older versions)
        'hallucination_silence_threshold': None,
        'hotwords': HOTWORDS,
    }
    vad_options = {
        'vad_onset': 0.2,    # lowered from 0.3 — detects quieter speech mid-call
        'vad_offset': 0.15,  # lowered from 0.2 — VAD stays active through brief pauses
    }

    lang = language or "en"

    # Resample to 16kHz and prepend SILENCE_PAD_SECONDS of silence for VAD warmup.
    # padded_audio_file = _prepend_silence(audio_file)
    padded_audio_file = audio_file  # use original file directly (ffmpeg step skipped for now)

    try:
        # 1. Load (or retrieve cached) ASR model.
        # chunk_size is a transcribe() parameter in whisperx >= 3.8.x.
        asr_key = (model_needed, lang)
        with _asr_lock:
            if asr_key not in _asr_cache:
                logger.info("Loading WhisperX ASR model: %s (lang=%s) — first call only", model_needed, lang)
                _asr_cache[asr_key] = whisperx.load_model(
                    model_needed, device=device, compute_type=compute_type,
                    language=language, asr_options=asr_options, vad_options=vad_options,
                    task='transcribe'
                )
        model = _asr_cache[asr_key]

        audio = whisperx.load_audio(padded_audio_file)
        _audio_duration_secs = len(audio) / 16000  # whisperx loads at 16kHz
        transcribe_args = {}
        if language is not None:
            transcribe_args["language"] = language
        else:
            transcribe_args["language"] = "en"

        torch.cuda.is_available()

        import time as _time
        _t0 = _time.time()
        with torch.no_grad():
            result = model.transcribe(audio, batch_size=batch_size, chunk_size=5, **transcribe_args)
        _transcribe_elapsed = _time.time() - _t0
        _audio_mins = _audio_duration_secs / 60
        print(f"[WhisperX] transcription done in {_transcribe_elapsed:.1f}s  audio={_audio_mins:.1f}min  file={os.path.basename(audio_file)}")
        logger.info("WhisperX transcription completed in %.1fs (audio=%.1fmin, file=%s)", _transcribe_elapsed, _audio_mins, os.path.basename(audio_file))

        # DEBUG — raw segments before alignment (start/end are in padded-file seconds)
        print("\n===== RAW WHISPERX SEGMENTS (before alignment) =====")
        for i, seg in enumerate(result.get("segments", [])):
            adjusted_start = round(seg['start'] - SILENCE_PAD_SECONDS, 3)
            adjusted_end   = round(seg['end']   - SILENCE_PAD_SECONDS, 3)
            print(f"  [{i}] padded({seg['start']:.3f}s → {seg['end']:.3f}s) "
                  f"original({adjusted_start}s → {adjusted_end}s): {seg['text'].strip()[:80]}")
        print("=====================================================\n")

        # 2. Align whisper output
        try:
            with _align_lock:
                if lang not in _align_cache:
                    logger.info("Loading WhisperX alignment model for lang=%s — first call only", lang)
                    _align_cache[lang] = whisperx.load_align_model(language_code=lang, device=device)
            model_a, metadata = _align_cache[lang]
            _t1 = _time.time()
            with torch.no_grad():
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            _align_elapsed = _time.time() - _t1
            print(f"[WhisperX] alignment done in {_align_elapsed:.1f}s  audio={_audio_mins:.1f}min  file={os.path.basename(audio_file)}")
            logger.info("WhisperX alignment completed in %.1fs (audio=%.1fmin, file=%s)", _align_elapsed, _audio_mins, os.path.basename(audio_file))
        except Exception:
            logger.error("Fail to align")

        # 3. Flush transcription intermediates BEFORE loading NeMo to avoid OOM
        del audio
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # 4. Diarize
        if DIARIZER_BACKEND == "nemo":
            _nemo_t0 = _time.time()
            diarize_segments = _diarize_with_nemo(padded_audio_file)
            _nemo_elapsed = _time.time() - _nemo_t0
            print(f"[NeMo] diarization done in {_nemo_elapsed:.1f}s  audio={_audio_mins:.1f}min  device={NEMO_DEVICE}  file={os.path.basename(audio_file)}")
            logger.info("NeMo diarization completed in %.1fs (audio=%.1fmin, device=%s, file=%s)", _nemo_elapsed, _audio_mins, NEMO_DEVICE, os.path.basename(audio_file))
        else:
            # pyannote (fallback)
            diarize_segments = diarize_model(padded_audio_file)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Flush NeMo GPU memory — NeMo leaves ~10GB in PyTorch pool after each call
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Format output — whisperx 3.8.x removed SubtitlesWriter; speaker label is now
        # a direct field on each segment dict after assign_word_speakers.
        items = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue
            # Drop very short segments (< 0.4 s) — typically background noise
            #if (seg["end"] - seg["start"]) < 0.4:
            #    continue
            words = seg.get("words", [])
            word_scores = [w["score"] for w in words if "score" in w]
            avg_score = round(sum(word_scores) / len(word_scores), 3) if word_scores else None
            items.append({
                "start":     _ts_from_seconds(seg["start"] - SILENCE_PAD_SECONDS),
                "end":       _ts_from_seconds(seg["end"]   - SILENCE_PAD_SECONDS),
                "speaker":   seg.get("speaker", "UNKNOWN"),
                "phrase":    text,
                "avg_score": avg_score,
            })

    finally:
        if padded_audio_file != audio_file and os.path.isfile(padded_audio_file):
            os.remove(padded_audio_file)

    return items
