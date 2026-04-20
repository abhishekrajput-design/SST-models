# Call Processor — Architecture & Flow

End-to-end ASR pipeline for call-center audio transcription with 8 interchangeable models.

**Live:** http://13.42.127.218:8080

---

## System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                               │
│                   http://13.42.127.218:8080                        │
│                                                                    │
│   [Upload Audio] ──► [Select Model] ──► [View Transcript]          │
└────────────────────────────┬───────────────────────────────────────┘
                             │ HTTP POST /api/upload
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Python HTTP Server (ui.py)                      │
│                    systemd: callproc.service                       │
│                    Port 8080 · Threading TCP                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────────────────┐    ┌──────────────────────┐             │
│   │ do_POST /api/upload  │───►│  _transcribe_inline  │             │
│   └──────────────────────┘    └──────────┬───────────┘             │
│                                          │                         │
│                                          ▼                         │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │              AUDIO ENHANCEMENT PIPELINE (4 stages)         │   │
│   │                                                            │   │
│   │   [1] FFmpeg        ──► highpass + afftdn + 3× volume      │   │
│   │   [2] noisereduce   ──► spectral gating (5-min windows)    │   │
│   │   [3] angelina      ──► 10-stage desk-recording cleanup    │   │
│   │   [4] MetricGAN+    ──► PESQ-optimised neural denoiser     │   │
│   │                                                            │   │
│   │   Input:  raw_calls/audio.mp3                              │   │
│   │   Output: raw_calls/enhanced_audio.mp3                     │   │
│   └────────────────────────┬───────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │              TRANSCRIBER REGISTRY                          │   │
│   │              src/transcribers/__init__.py                  │   │
│   │                                                            │   │
│   │   get_transcriber(name) ─► routes to one of:               │   │
│   │                                                            │   │
│   │   ☁️  Cloud API       🖥️  Local GPU                         │   │
│   │   ───────────────    ────────────────────                  │   │
│   │   deepgram-nova-3    whisper-large-v3                      │   │
│   │   nova-2-phonecall   whisper-large-v3-turbo                │   │
│   │   nova-2-meeting     distil-whisper-large-v3.5             │   │
│   │                      parakeet-tdt-0.6b-v3                  │   │
│   │                      cohere-transcribe-03-2026             │   │
│   └────────────────────────┬───────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │   Segments: [{start, end, text, speaker, confidence}, ...] │   │
│   └────────────────────────┬───────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │   data/processed/<id>/result.json                          │   │
│   │   ─ audio_file, model, processed_at, segments, ...         │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                             ▲
                             │ GET /api/call/<id>
                             │ GET /api/calls
                             │
                    ┌────────┴────────┐
                    │   UI polls      │
                    │   displays      │
                    │   transcript    │
                    └─────────────────┘
```

---

## Data Flow (Step-by-Step)

1. **User uploads audio** via the web UI (drag-drop or file picker)
2. **POST /api/upload** — server writes the file to `data/raw_calls/`
3. **Enhancement pipeline** runs the 4 cleanup stages sequentially:
   - Each stage writes an intermediate file (e.g. `enhanced_*.mp3`, `nr_*.mp3`, `df_*.mp3`, `mg_*.mp3`)
   - Status updates pushed to `/api/status` as each stage completes
4. **Transcriber dispatch** — the selected model factory is looked up in `TRANSCRIBERS` dict
5. **Inference** — model runs on enhanced audio:
   - Local GPU: loads weights into VRAM, processes chunks
   - Cloud API: uploads audio, polls for result
6. **Output normalisation** — every transcriber returns the same segment schema:
   ```json
   { "start": 0.0, "end": 2.5, "text": "...", "speaker": "SPEAKER_00", "confidence": 0.95 }
   ```
7. **Result saved** to `data/processed/<enhanced_filename>__<model>/result.json`
8. **UI polls `/api/calls`** — shows the new entry in the sidebar (newest first)
9. **User clicks entry** → GET `/api/call/<id>` → transcript rendered in the chat view

---

## Component Breakdown

### Frontend (`call_processor/index.html`)
- Single-page dark-theme dashboard
- Model dropdown grouped by `☁️ Cloud API` / `🖥️ Local GPU`
- Call history sidebar sorted by `processed_at` (latest first)
- Chat view with speaker bubbles + audio scrubber
- Benchmark panel showing ranked models with WER / speed / notes

### Backend (`call_processor/ui.py`)
- Pure stdlib HTTP server (no Flask/FastAPI — minimal deps)
- `_status` dict: in-memory progress state, polled via `/api/status`
- `_transcribe_inline()`: runs the full pipeline for a single upload
- `/api/benchmark`: serves the ranked model table
- `/api/calls`: lists all processed recordings (scans `data/processed/`)

### Transcriber base (`src/transcribers/base.py`)
- `BaseTranscriber` interface:
  - `load()` — download weights / verify API key
  - `transcribe(audio_path, language)` → `List[segment_dict]`
  - `unload()` — free GPU memory / reset state
- Each subclass is isolated — a failure in one doesn't affect others

---

## Model Comparison

Measured on `audio_04_16_2026_15_34_29_kixm5p.mp3` (live server, RTX A-series GPU):

| # | Model | Type | WER | Time | Segments | Best For |
|---|-------|------|-----|------|----------|----------|
| 1 | **Deepgram Nova-3** | Cloud | ~7% | 6s | 60 | Fast + diarization |
| 2 | **Whisper Large-v3** | Local GPU | 8.1% | 78s | 306 | Highest local quality |
| 3 | **Whisper v3-Turbo** | Local GPU | 8.4% | 35s | 307 | Best speed/quality trade-off |
| 4 | **Distil-Whisper v3.5** | Local GPU | 8.6% | 29s | 433 | Fastest Whisper, most granular |
| 5 | **Parakeet TDT v3** | Local GPU | ~5.5% | 22s | 126 | Fastest local + English-only |
| 6 | **Cohere Transcribe** | Local GPU | 5.42% | 52s | 60 | Lowest WER on leaderboard |
| 7 | **Deepgram Nova-2 Phone** | Cloud | ~9% | 6s | 33 | Phone-optimised |
| 8 | **Deepgram Nova-2 Meeting** | Cloud | ~9% | 8s | 51 | Multi-speaker meetings |

---

## Deployment

### Server layout (`/home/ubuntu/projects/SST-models/`)
```
.
├── call_processor/         app code
│   ├── ui.py               HTTP server + pipeline orchestration
│   ├── index.html          dashboard
│   ├── src/
│   │   ├── transcribers/   pluggable ASR backends
│   │   ├── enhance*.py     audio cleanup stages
│   │   └── ...
│   ├── data/
│   │   ├── raw_calls/      uploaded + enhanced audio
│   │   └── processed/      result.json per call × model
│   └── models/             cached model weights
├── restart.sh              git pull + kill + start
├── download_all_models.sh  pre-cache all local models
├── fix_service.sh          update systemd unit
└── .env                    API keys (gitignored)
```

### systemd service (`/etc/systemd/system/callproc.service`)
- Auto-restart on failure
- Logs to `/var/log/callproc/server.log`
- Python: `/opt/miniconda3/envs/callproc/bin/python`

### Deploy workflow
```bash
bash restart.sh            # pull latest + restart server (one command)
tail -f /var/log/callproc/server.log   # watch logs
curl http://localhost:8080/api/status  # health check
```

---

## API Keys (`.env`)

```
HF_TOKEN=...            # HuggingFace — for gated model downloads
DEEPGRAM_API_KEY=...    # Deepgram cloud (Nova-3, Nova-2)
```

Loaded at server startup by `ui.py` via `env_path = Path(__file__).parent.parent / ".env"`.

---

## Error Handling Highlights

- **Hallucination filter (Canary-Qwen)** — drops loops like `"and then X and then X and then X..."` before they reach the UI
- **CUDA probe** — every GPU transcriber does `torch.zeros(1).cuda()` before loading; falls back to CPU if the driver is too old
- **Deepgram Nova-2 fallback** — retries without `diarize=true` if the first call returns 0 utterances (some over-processed audio breaks nova-2)
- **Result directory recreation** — `os.makedirs(out_dir, exist_ok=True)` right before writing `result.json` (guards against user deleting a folder mid-transcription)
- **Stale status reset** — `_status` resets to `Idle` on each fresh server start so crashed runs don't show old errors

---

## Key Design Decisions

1. **Registry pattern** for transcribers — adding a new model = one file + one line in `__init__.py`
2. **Uniform segment schema** — UI doesn't care which model produced the transcript
3. **Single-process HTTP server** — simpler than Flask, zero deps beyond stdlib
4. **`.env` for secrets** — gitignored, loaded once at startup
5. **Enhancement before transcription** — all models get cleaner audio; nova-3 handles heavy processing, some older models (nova-2) need fallback
6. **Model weights cached locally** — `models/faster-whisper/`, `models/nemo/`, `models/hf/` — no re-download between runs
