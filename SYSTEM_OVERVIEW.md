# Call Processor — System Overview

> **Last updated:** 2026-04-22  
> **Hardware:** NVIDIA GeForce RTX 4050 (6 GB VRAM) · Windows 11  
> **Python:** 3.11.9 · PyTorch 2.5.1+cu121

---

## What This System Does

Transcribes recorded phone calls / floor recordings from car dealerships.  
The pipeline takes a raw MP3 upload, enhances the audio, transcribes it with an AI model, trims silence, and displays the result in a browser UI with an audio player synced to the transcript.

---

## Architecture

```
Browser (index.html)
     │  upload MP3 + model choice
     ▼
ui.py  (Python HTTP server on :8080)
     │
     ├─ Stage 0a: FFmpeg  — loudnorm + highpass filter
     ├─ Stage 0b: DeepFilterNet3  — neural denoising (GPU)
     │
     ├─ Stage 1-3: Transcription  (_transcribe_inline)
     │     ├─ Normalize to 16 kHz mono WAV
     │     ├─ Load transcriber (whisper / parakeet / …)
     │     ├─ Run transcription → list of segments
     │     └─ Trim audio to speech-only regions
     │
     └─ result.json  →  Browser displays transcript + audio player
```

---

## Starting the Server (Windows Local)

```bat
call_processor\start_ui.bat
```

Opens `http://localhost:8080` in browser.

The bat file sets the FFmpeg PATH and launches `python ui.py` from the `call_processor/` directory.

---

## Supported Transcription Models

| Model | Library | VRAM | Speed (RTF) | Best for |
|-------|---------|------|-------------|----------|
| `whisper-large-v3-turbo` | faster-whisper 1.2.1 | ~3.5 GB | ~8–15× real-time | Best overall quality |
| `whisper-large-v3` | faster-whisper | ~5 GB | ~4–8× | Highest accuracy |
| `distil-large-v3` | faster-whisper | ~1.5 GB | ~20× | Fastest Whisper |
| `distil-large-v3.5` | faster-whisper | ~1.5 GB | ~20× | Fastest Whisper |
| `parakeet-tdt-0.6b-v3` | NeMo 2.7.2 | ~2 GB | Very fast | English, non-autoregressive (no hallucination loops) |

Select the model from the dropdown in the UI before uploading.

---

## Audio Processing Pipeline Detail

### Stage 0a — FFmpeg Enhancement

```python
AUDIO_FILTER = (
    "loudnorm=I=-16:TP=-1:LRA=11,"   # normalize loudness
    "highpass=f=80,"                   # remove rumble below 80 Hz
    "silenceremove=..."                # -60 dB threshold, 3 s stop duration
)
```

Output: `data/raw_calls/enhanced_<filename>.mp3`

### Stage 0b — DeepFilterNet3 (optional)

Neural denoiser trained on the DNS-4 dataset. Runs on GPU if available.  
Falls back gracefully to FFmpeg-only output if not installed or if VRAM is insufficient.

Output: `data/raw_calls/df_<filename>.mp3`

### Transcription (_transcribe_inline)

1. Normalizes audio to 16 kHz mono WAV (cached — skipped if already done)
2. Loads the selected transcriber
3. Runs transcription → list of segment dicts:
   ```json
   {"start": 0.5, "end": 3.2, "text": "Hello sir welcome in.", "speaker": "SPEAKER_00"}
   ```
4. Runs `filter_hallucinations()` — removes repetition loops and silence artefacts
5. Trims audio (see below)
6. Writes `result.json`

### Audio Trimming (_trim_to_speech)

After transcription, the audio is cut to speech-only regions:

- Takes all transcript segments
- Merges segments that are ≤ **5 seconds** apart into one block
- Adds **1 second** padding on each side
- Builds an ffmpeg `aselect` filter via a temp script file (`-filter_complex_script`)
- Output: `data/processed/<id>/trimmed_audio.mp3`
- The UI Enhanced audio player defaults to the trimmed version

This removes long silences from floor recordings (e.g. 30 min → 15–20 min of actual speech).

---

## Whisper VAD Settings (Noisy Audio Tuned)

Key parameters in `src/transcribers/whisper_turbo.py`:

```python
vad_filter=True,
vad_parameters={
    "threshold": 0.3,           # was 0.5 — catches speech in background noise
    "speech_pad_ms": 1000,      # was 400 — keeps 1 s around each speech burst
    "min_silence_duration_ms": 2000,  # was 500 — joins short pauses
},
beam_size=5,
temperature=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # fallback retries
no_speech_threshold=0.6,
compression_ratio_threshold=1.8,  # filters "down down down..." loops
```

**Why threshold=0.3:** Car dealership floor recordings have constant background noise (HVAC, music, ambient voices). The default VAD threshold of 0.5 classifies most noisy speech as "not speech" and skips it. Lowering to 0.3 significantly increases transcription coverage.

---

## Hallucination Filtering

Whisper (being autoregressive) can get stuck in repetition loops on silence or noisy audio. Two-layer defence in `src/transcribers/base.py`:

1. **Before transcription** (model level):
   - `compression_ratio_threshold=1.8` — rejects highly repetitive segments
   - `temperature` fallback chain — retries with higher randomness if quality fails
   - `no_speech_threshold=0.6` — skips near-silent segments

2. **After transcription** (`filter_hallucinations()`):
   - Drops pure-punctuation segments (`.`, `...`, `?` — silence artefacts)
   - Drops segments where one word is >45% of all words
   - Drops segments with 4+ identical consecutive words

Parakeet is CTC/RNNT (non-autoregressive) — it physically cannot produce repetition loops, so the post-filter is a safety net only.

---

## Result Format

Each pipeline run produces `data/processed/<result_id>/result.json`:

```json
{
  "audio_file": "data/raw_calls/enhanced_upload.mp3",
  "trimmed_audio_file": "data/processed/.../trimmed_audio.mp3",
  "model": "whisper-large-v3-turbo",
  "processed_at": "2026-04-22T09:12:00Z",
  "processing_time_seconds": 1469,
  "total_segments": 477,
  "segments": [
    {"start": 0.5, "end": 3.2, "text": "...", "speaker": "SPEAKER_00",
     "identified_speaker": "SPEAKER_00", "confidence": -0.45}
  ],
  "transcription_json": [
    {"start": "00:00:00.500", "end": "00:00:03.200",
     "speaker": "SPEAKER_00", "phrase": "...", "avg_score": -0.45}
  ],
  "enhancements": {
    "ffmpeg": "data/raw_calls/enhanced_upload.mp3",
    "deepfilter": "data/raw_calls/df_upload.mp3"
  }
}
```

---

## Project File Map

```
SST-models/
├── call_processor/
│   ├── ui.py                      ← Main HTTP server (port 8080)
│   ├── index.html                 ← Browser UI
│   ├── start_ui.bat               ← Windows launcher
│   │
│   ├── src/
│   │   ├── transcribers/
│   │   │   ├── base.py            ← BaseTranscriber + filter_hallucinations()
│   │   │   ├── whisper_turbo.py   ← faster-whisper (large-v3-turbo, distil variants)
│   │   │   ├── parakeet_v3.py     ← NVIDIA Parakeet TDT v3 via NeMo
│   │   │   ├── cohere.py          ← Cohere Transcribe (API-based)
│   │   │   └── __init__.py        ← get_transcriber() registry
│   │   ├── diarization.py         ← pyannote speaker diarization
│   │   ├── diar_ecapa.py          ← ECAPA speaker embedding
│   │   └── pipeline.py            ← Full E2E pipeline (run_e2e.py path)
│   │
│   ├── benchmark.py               ← Speed/quality benchmarks
│   ├── download_models.py         ← Pre-download all models
│   ├── test_all_models.py         ← Run all models on a test file
│   └── data/
│       ├── raw_calls/             ← Uploaded + enhanced audio files
│       └── processed/             ← Per-run result directories
│
├── transcribe_only.py             ← Standalone transcription (no UI)
├── test_sample_compare.py         ← WER comparison across models
├── test_dfnet.py                  ← DeepFilterNet3 standalone test
├── compare_all.py                 ← Side-by-side model output comparison
└── restart.sh                     ← AWS server restart script
```

---

## Utility Scripts

### transcribe_only.py
Transcribe a single audio file without the UI server. Useful for batch or debugging.
```bash
python transcribe_only.py
# Edit audio_path inside the script to target a specific file
```

### test_sample_compare.py
Upload `test_sample.mp3` to the local server, run all 3 models, compute WER against a ground truth transcript, and print a ranked comparison table.
```bash
python test_sample_compare.py
```

### benchmark.py
Run speed and quality benchmarks across Whisper variants on the same audio.
```bash
cd call_processor && python benchmark.py
```

### download_models.py
Pre-download all supported models to the local `models/` cache (run once on a new machine).
```bash
cd call_processor && python download_models.py
```

---

## Key Git Commits (Latest → Oldest)

| Hash | Description |
|------|-------------|
| `52b747d` | Fix noisy-audio: lower VAD threshold (0.5→0.3), wider trim merge gap |
| `9e9d18f` | Add audio trimming — cut silence gaps from transcribed calls |
| `49aa470` | Fix hallucination loops + audio over-cutting (silenceremove threshold) |
| `8754ee7` | restart.sh: unbuffered Python stdout + truncate stale log |
| `4e5f40e` | Add diagnose_live.sh — one-command live-server diagnostic dump |
| `cd449e7` | Add compare_all.py + compare_three.py: WER comparison vs ground truth |
| `e09bf9a` | Audio enhancement: loudnorm + silence removal + Linux-safe FFmpeg PATH |
| `8fa30b6` | Stable build: remove diarization, keep all transcribers |

---

## Known Issues & Tuning Notes

| Issue | Cause | Fix Applied |
|-------|-------|-------------|
| Whisper outputs "down, down, down…" loops | Autoregressive hallucination on silence/noise | `compression_ratio_threshold=1.8` + temperature fallback |
| Silence artefacts (`.` segments) | Whisper sees near-silence as speech | `filter_hallucinations()` strips pure-punctuation |
| Low-bitrate audio (32kbps) over-cut | `silenceremove` threshold -45dB cut 97% of audio | Moved `loudnorm` before `silenceremove`; relaxed to -60dB |
| Floor recording: only 9/30 min transcribed | VAD threshold 0.5 too strict for noisy audio | Lowered to 0.3 + speech_pad_ms 1000 |
| Trim creates too many tiny blocks | 1.5 s merge gap too short for burst speech | Increased to 5.0 s merge gap |

---

## Dependencies

```
faster-whisper==1.2.1     # Whisper variants (CTranslate2 backend)
nemo_toolkit[asr]==2.7.2  # NVIDIA Parakeet TDT v3
torch==2.5.1+cu121        # GPU inference
transformers==4.57.6      # HuggingFace model hub
speechbrain==1.1.0        # ECAPA speaker embeddings
ffmpeg (system)           # Audio conversion and trimming
```

Install:
```bash
pip install faster-whisper nemo_toolkit[asr] torch transformers speechbrain
# ffmpeg via winget:
winget install Gyan.FFmpeg
```
