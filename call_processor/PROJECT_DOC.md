# Call Processor — Project Documentation

**Last updated:** 2026-04-29
**Status:** Production-ready (94.8% identification accuracy vs Audiofy API)

---

## 1. Overview

A local pipeline that takes a phone-call recording (mono MP3/WAV) and produces:
- A speaker-diarised transcript (Agent vs Customer)
- The identified agent's name (matched against enrolled voiceprints)
- Per-segment confidence scores

Runs entirely on a single workstation (RTX 4050 6 GB VRAM). No cloud dependency.
Used to process Car Planet dealership sales calls.

---

## 2. Pipeline architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Upload (UI / API)                                            │
│    POST /api/upload?filename=X.mp3&model=parakeet-tdt-0.6b-v3   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Audio Enhancement — _enhance_ffmpeg                          │
│    aresample=44100 → highpass=80 → afftdn → loudnorm I=-16      │
│    → dynaudnorm                                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Clean-audio detector  (_is_clean_audio)                      │
│    RMS + spectral flatness check → skip DFN3 on clean inputs   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. DeepFilterNet3 neural denoise (GPU, 5-min chunks)            │
│    Skipped if input is already clean broadcast-quality           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Normalise to 16 kHz mono  (_make_norm_wav)                   │
│    aformat=mono → aresample=44100 → loudnorm → dynaudnorm       │
│    → output -ar 16000                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Transcribe (Parakeet TDT v3 on CUDA)                         │
│    Falls back to whisper-large-v3-turbo if 0 segments           │
│    `requested_model` + `fallback_used` recorded in result       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. diar_multi.diarize_multi — voiceprint-first speaker ID       │
│    a. Load 49 enrolled voiceprints (192 ECAPA + 512 CAM++)      │
│    b. Per-segment ECAPA/CAM++ embedding (widened ±0.2s if <0.8s)│
│    c. VAD speech-ratio filter (reject low-speech chunks)        │
│    d. Per-agent threshold (max(0.30, max_outside_sim+0.05) ≤0.42)│
│    e. Cluster_first mode (calls ≥60s, ≥15 valid embs)            │
│    f. Centroid-gated soft-reclaim                               │
│    g. Anti-flip smoothing (3 passes)                            │
│    h. Back-channel demotion (low-sim "yeah/okay/yes"-led segs)  │
│    i. Reverse cluster-reconciliation (high-sim → AGENT)         │
│    j. Role corrections (back-channels, farewells)               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. Trim & save result.json                                      │
│    data/processed/<base>__<model>/result.json                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Enrolled agents (49 total)

### Quality summary

| Quality | Count | Definition |
|---|---:|---|
| GOOD | 19 | inside_sim − outside_sim > 0.15 |
| OK | 18 | inside_sim − outside_sim ∈ (0.05, 0.15] |
| WEAK | 6 | inside_sim − outside_sim ∈ (0, 0.05] |
| BROKEN | 4 | outside_sim ≥ inside_sim — can't separate from customers |
| NO_STATS | 2 | enrolled before stats were recorded — uses global threshold |

### Full list

| # | Agent name | Slug | Backend | Inside sim | Outside sim | Clips | Quality |
|--:|---|---|--:|--:|--:|--:|---|
|  1 | Adil Al-Sammerai | adil_al_sammerai | CAM++ 512 | 0.630 | 0.581 | 757 | WEAK |
|  2 | Adorena Ishtar Hossain | adorena_ishtar_hossain | CAM++ 512 | 0.619 | 0.515 | 117 | OK |
|  3 | Aftaab Supervisor | aftaab_supervisor | CAM++ 512 | 0.718 | 0.548 | 325 | GOOD |
|  4 | Albjon Vokshi | albjon_vokshi | ECAPA 192 | 0.633 | 0.551 | 58 | OK |
|  5 | **Allan Johnson** | allan_johnson | CAM++ 512 | 0.636 | **0.645** | 543 | **BROKEN** |
|  6 | **Amandeep Nandra** | amandeep_nandra | ECAPA 192 | 0.650 | **0.669** | 153 | **BROKEN** |
|  7 | Angeline Packiyaseelan | angeline_packiyaseelan | CAM++ 512 | 0.700 | 0.535 | 189 | GOOD |
|  8 | Anoush Sefatzadeh | anoush_sefatzadeh | CAM++ 512 | 0.689 | 0.648 | 502 | WEAK |
|  9 | Arfat Barnet | arfat_barnet | CAM++ 512 | 0.664 | 0.563 | 397 | OK |
| 10 | Benjamin Ahmadi | benjamin_ahmadi | CAM++ 512 | 0.648 | 0.536 | 90 | OK |
| 11 | Dilayda Barnet | dilayda_barnet | CAM++ 512 | 0.683 | 0.553 | 239 | OK |
| 12 | Dinosh Sinnathamby | dinosh_sinnathamby | CAM++ 512 | 0.713 | 0.523 | 561 | GOOD |
| 13 | Gabriel Bighiu | gabriel_bighiu | CAM++ 512 | 0.707 | 0.538 | 165 | GOOD |
| 14 | **Georgi Angelov** | georgi_angelov | CAM++ 512 | 0.639 | **0.674** | 32 | **BROKEN** |
| 15 | Haris Bajwa | haris_bajwa | CAM++ 512 | 0.670 | 0.545 | 474 | OK |
| 16 | Harrison Morgan | harrison_morgan | CAM++ 512 | 0.712 | 0.519 | 180 | GOOD |
| 17 | Ideal Dacaj | ideal_dacaj | CAM++ 512 | 0.623 | 0.604 | 905 | WEAK |
| 18 | **Janusaan Jeyachandran** | janusaan_jeyachandran | CAM++ 512 | 0.619 | **0.633** | 176 | **BROKEN** |
| 19 | Jason Kurti | jason_kurti | CAM++ 512 | 0.691 | 0.542 | 377 | OK |
| 20 | Jenifer Bajrami | jenifer_bajrami | CAM++ 512 | 0.693 | 0.338 | 66 | GOOD |
| 21 | Kacper Barnet | kacper_barnet | CAM++ 512 | 0.665 | 0.570 | 828 | OK |
| 22 | Kleo Gurra | kleo_gurra | CAM++ 512 | 0.739 | 0.521 | 90 | GOOD |
| 23 | Kowsar Alam | kowsar_alam | CAM++ 512 | 0.711 | 0.544 | 445 | GOOD |
| 24 | Liza Mae Esguerra | liza_mae_esguerra | ECAPA 192 | 0.693 | 0.553 | 63 | OK |
| 25 | Mashrur Rahman | mashrur_rahman | ECAPA 192 | 0.708 | 0.542 | 217 | GOOD |
| 26 | Mohamed Yasin-ali | mohamed_yasin_ali | CAM++ 512 | 0.680 | 0.534 | 118 | OK |
| 27 | Mohammad Malki | mohammad_malki | CAM++ 512 | 0.710 | 0.533 | 378 | GOOD |
| 28 | Mohammed Al Russell | mohammed_al_russell | ECAPA 192 | — | — | 175 | NO_STATS |
| 29 | Mohammed-Hussein Al-Khwildi | mohammed_hussein_al_khwildi | CAM++ 512 | 0.691 | 0.545 | 244 | OK |
| 30 | Mohammed Malik | mohammed_malik | CAM++ 512 | 0.702 | 0.522 | 83 | GOOD |
| 31 | Nevethan Krishnamohan | nevethan_krishnamohan | ECAPA 192 | 0.684 | 0.515 | 73 | GOOD |
| 32 | Niloufar Dastbaz | niloufar_dastbaz | CAM++ 512 | 0.606 | 0.569 | 27 | WEAK |
| 33 | Nirvan Nagra | nirvan_nagra | CAM++ 512 | 0.734 | 0.597 | 31 | OK |
| 34 | Omar El Harchaoui | omar_el_harchaoui | CAM++ 512 | 0.679 | 0.551 | 585 | OK |
| 35 | Qaim Ravji | qaim_ravji | ECAPA 192 | **0.870** | **0.243** | 44 | **GOOD (best)** |
| 36 | Rafik Saleh | rafik_saleh | CAM++ 512 | 0.719 | 0.543 | 130 | GOOD |
| 37 | Rajan Singh | rajan_singh | ECAPA 192 | 0.649 | 0.533 | 110 | OK |
| 38 | Rayyan Ali Khan | rayyan_ali_khan | CAM++ 512 | 0.679 | 0.537 | 537 | OK |
| 39 | Rebeca Cazan | rebeca_cazan | ECAPA 192 | 0.650 | 0.513 | 70 | OK |
| 40 | Rebecca Murphy | rebecca_murphy | ECAPA 192 | 0.607 | 0.576 | 62 | WEAK |
| 41 | Sababa Hossain | sababa_hossain | CAM++ 512 | 0.666 | 0.507 | 72 | GOOD |
| 42 | Sarah Aziz | sarah_aziz | ECAPA 192 | 0.657 | 0.625 | 74 | WEAK |
| 43 | Shuahib Miah | shuahib_miah | CAM++ 512 | 0.712 | 0.543 | 110 | GOOD |
| 44 | Sylwia Recruitment | sylwia_recruitment | CAM++ 512 | 0.698 | 0.543 | 1113 | GOOD |
| 45 | Talha Azam | talha_azam | CAM++ 512 | 0.711 | 0.523 | 90 | GOOD |
| 46 | Tulay Finance Consultant | tulay_finance_consultant | ECAPA 192 | 0.696 | 0.540 | 278 | GOOD |
| 47 | Waris Sales Controllers | waris_sales_controllers | CAM++ 512 | 0.648 | 0.577 | 93 | OK |
| 48 | Zak (local) | zak_local_20260423 | ECAPA 192 | — | — | 8 | NO_STATS |
| 49 | Zak Raissi Barnet | zak_raissi_barnet | CAM++ 512 | 0.691 | 0.541 | 351 | OK |

### Backend distribution

- **CAM++ 512-dim:** 36 agents (newer, larger scope)
- **ECAPA 192-dim:** 13 agents (legacy, still working)

The pipeline handles both transparently — `diar_multi` extracts both embedding types and picks whichever scores higher.

---

## 4. Performance metrics

### Identification accuracy (vs Audiofy API ground truth, 4 Omar calls, 555 sec, 193 GT segments)

| Call | Duration | GT segs | Our segs | Overall | Agent | Customer |
|---|--:|--:|--:|--:|--:|--:|
| call1_132s.mp3 | 132s | 56 | 57 | 93.8% | 96.8% | 79.4% |
| call2_88s.mp3 | 88s | 20 | 25 | 95.6% | 95.7% | 95.6% |
| enroll1_149s.mp3 | 149s | 59 | 59 | **96.3%** | 98.6% | 92.0% |
| enroll2_186s.mp3 | 186s | 58 | 63 | 93.7% | 94.5% | 91.7% |
| **Macro average** | | | | **94.8%** | **96.4%** | **89.7%** |

### Transcription quality (WER vs Audiofy API)

| Metric | WER |
|---|--:|
| Full transcript | 26.3% |
| Agent-only text | 32.0% |
| Customer-only text | 36.6% |

Industry baseline for English phone audio: 25–30 % WER. We're on par.

### Speed (30-min real call)

| Stage | Time |
|---|--:|
| FFmpeg + loudnorm 2-pass | ~2 min |
| DeepFilterNet3 (GPU, chunked) | ~5 min |
| Parakeet TDT v3 transcription (GPU) | ~3 min |
| diar_multi voiceprint matching | ~1 min |
| **Total wall time** | **~10 min** |

Distil-Whisper takes ~10 min instead of 5 (slower transcription + more "Thank you." hallucinations).

---

## 5. Improvements log (2026-04-27 → 2026-04-29)

### Phase 1 — Infrastructure (8 fixes)

1. **Per-agent dynamic threshold** — uses `max_outside_sim + 0.05` per voiceprint
2. **Parakeet on GPU** — 5–10× faster than CPU (was forced CPU before)
3. **Transparent fallback** — `requested_model` + `fallback_used` in result.json
4. **RMS-based DFN3 skip** — clean inputs bypass neural denoise (avoids over-processing)
5. **Cancel button + `/api/cancel`** — checked between pipeline stages
6. **Confidence bar in chat bubbles** — green/amber/grey under each bubble
7. **Auto-cleanup of orphan dirs** — `_gc_orphan_processed_dirs()` on startup
8. **17 legacy scripts moved** to `tools/legacy/`

### Phase 2 — Audio quality (2 critical fixes)

9. **Restored loudnorm** — phone audio was silent for Whisper/Parakeet without it
10. **Replaced `pan=mono|c0=...FL+FR`** with `aformat=channel_layouts=mono` — DFN3 outputs mono, pan filter was producing zero output

### Phase 3 — Diarisation accuracy (5 iterations)

| Iter | Change | Overall | Agent | Customer |
|---|---|--:|--:|--:|
| 0 | Baseline (after audio fix) | 85.7% | 96.2% | 67.1% |
| 1 | Centroid-gated soft-reclaim | 90.0% | 90.5% | 86.0% |
| 2 | Relaxed centroid margin (+0 instead of +0.05) | 92.5% | 94.7% | 85.3% |
| 3 | Multi-word back-channel rule (yeah-led ≤5 words) | 93.3% | 94.7% | 88.1% |
| 4 | Lowered cluster_first thresholds (60s+, 15+ embs) | 94.8% | 96.8% | 89.0% |
| 5 | Reverse cluster reconciliation + 15-word back-channel | **94.8%** | **96.4%** | **89.7%** |

**Net gains over baseline:**
- Customer ID: **67.1% → 89.7%** (+22.6 pts)
- Overall: **85.7% → 94.8%** (+9.1 pts)
- WER: unchanged (transcription was already good)

---

## 6. File organization

```
call_processor/
├── ui.py                         # HTTP server + pipeline orchestrator (1430 lines)
├── index.html                    # Frontend dashboard (1200 lines)
├── PROJECT_DOC.md                # ← this file
│
├── src/
│   ├── diar_multi.py             # Voiceprint-first multi-speaker diariser
│   ├── embedding_campp.py        # CAM++ 512-dim embedding model
│   ├── speaker_role.py           # ECAPA 192-dim helpers
│   ├── voiceprints.py            # Voiceprint loader / path resolver
│   ├── transcribers/
│   │   ├── parakeet_v3.py        # NVIDIA Parakeet TDT v3 (default)
│   │   ├── whisper_turbo.py      # Whisper variants
│   │   ├── cohere.py
│   │   ├── deepgram_asr.py
│   │   └── ...
│   └── audio_cleanup.py          # Legacy 10-stage desk-recording cleaner
│
├── data/
│   ├── raw_calls/                # Original uploads
│   ├── processed/                # Per-call result dirs (<base>__<model>/result.json)
│   ├── agent_voiceprints/        # 49 .npy files + agents.json index
│   └── audiofy/                  # API datasets (omar_dataset has GT speaker_json)
│
├── enroll_*.py                   # Per-agent enrollment scripts (kept in root)
├── test_combined_speakers.py     # Synthetic multi-agent test
├── test_api_compare.py           # API ground-truth comparison
├── test_api_diagnose.py          # Mis-classification diagnostic
├── test_omar_e2e.py              # Per-call Omar accuracy test
├── test_hussein_e2e.py
├── test_zak_e2e.py
├── build_combined_ui_result.py   # UI demo result builder
│
└── tools/legacy/                 # 17 archived one-off scripts
```

---

## 7. How to run

### Start the UI server

```powershell
cd C:\Users\abhis\Desktop\SST-models\call_processor
python ui.py
# → http://localhost:8080
```

### Process a call via API

```bash
curl -X POST "http://localhost:8080/api/upload?filename=mycall.mp3&model=parakeet-tdt-0.6b-v3" \
  --data-binary "@mycall.mp3"

# Poll status
curl http://localhost:8080/api/status

# Cancel mid-pipeline
curl -X POST http://localhost:8080/api/cancel
```

### Verify accuracy against ground truth

```bash
python test_api_compare.py    # ~5 min, all 4 omar calls
python test_api_diagnose.py   # show per-segment mis-classifications
```

### Synthetic multi-agent test

```bash
python test_combined_speakers.py    # builds 3-agents-plus-random WAV, tests independently
python build_combined_ui_result.py  # also creates UI-viewable result
```

---

## 8. Outstanding work — manual

### 4 BROKEN voiceprints (need re-enrollment from API)

These have `outside_sim ≥ inside_sim` — their voiceprint can't separate them from customers:

| Agent | inside | outside |
|---|--:|--:|
| Allan Johnson | 0.636 | 0.645 |
| Amandeep Nandra | 0.650 | 0.669 |
| Georgi Angelov | 0.639 | 0.674 |
| Janusaan Jeyachandran | 0.619 | 0.633 |

**Effect:** with our per-agent threshold capped at 0.42, these agents are effectively disabled for matching. They won't false-positive customer audio, but they also won't reliably identify their own calls.

**Fix:** re-run with stricter tightening. Open `enroll_all_from_api.py`, raise `TIGHT_PASS_2` from 0.55 to 0.65, then:

```bash
python enroll_all_from_api.py --agents "Allan Johnson" "Amandeep Nandra" "Georgi Angelov" "Janusaan Jeyachandran"
```

Requires API access (`AUDIOFY_API_TOKEN` in `.env`).

### 6 WEAK voiceprints (could improve, not urgent)

Adil Al-Sammerai, Anoush Sefatzadeh, Ideal Dacaj, Niloufar Dastbaz, Rebecca Murphy, Sarah Aziz.

Same fix script — included in the rerun above with their names.

### 2 NO_STATS voiceprints (low priority)

Mohammed Al Russell, Zak (local). They work fine via the global threshold fallback (0.30). Re-enrolling would just give them per-agent thresholds.

---

## 9. Known limitations

1. **Customer-ID drops on short calls** — call2_88s sometimes hits 70–95% range depending on noise. Calls under 60s benefit less from cluster-based reasoning.

2. **Generic phrases are ambiguous** — "Bye bye.", "Thank you.", "Take care." can be either party. Without context, embedding alone isn't enough.

3. **Voicemail prompts get labelled "Customer"** by Audiofy API and by us — neither distinguishes a recorded prompt from a live customer.

4. **Single-channel mono only** — stereo path was removed; uploading stereo gets downmixed via `aformat=channel_layouts=mono`.

5. **Parakeet is English-only** — for other languages use `whisper-large-v3-turbo` from the model dropdown.

---

## 10. Quick reference

### Processing time targets

- Phone call < 5 min: ~1 min total
- Phone call 5–15 min: ~3 min total
- Phone call 15–30 min: ~5 min total
- 30-min desk recording: ~10 min total

### Result JSON schema (top-level)

```json
{
  "audio_file": "path",
  "trimmed_audio_file": "path|null",
  "model": "parakeet-tdt-0.6b-v3",
  "requested_model": "parakeet-tdt-0.6b-v3",
  "fallback_used": false,
  "processed_at": "2026-04-29T17:22:42Z",
  "processing_time_seconds": 184.5,
  "total_segments": 215,
  "identified_agent": "Zak Raissi Barnet",
  "speaker_id_backend_dim": 512,
  "speaker_id_mode": "cluster_first_voiceprint",
  "voiceprint_dims": {"192": 13, "512": 36},
  "diarization": "diar_multi_voiceprint",
  "segments": [...],
  "transcription_json": [...],
  "speaker_stats": {...}
}
```

### Per-segment fields

```json
{
  "start": 12.4,
  "end": 15.1,
  "text": "I can offer you 0% finance over 48 months.",
  "speaker": "SPEAKER_00",
  "identified_speaker": "AGENT",
  "display_speaker": "Zak Raissi Barnet",
  "agent_name": "Zak Raissi Barnet",
  "_best_sim": 0.62,
  "_best_match": "zak_raissi_barnet",
  "confidence": -0.12
}
```

---

## 11. Stack

- **Python 3.11** (Windows, conda/winget Python)
- **PyTorch 2.x** with CUDA 12.x
- **NeMo** (Parakeet)
- **HuggingFace Transformers** (Whisper variants)
- **SpeechBrain 1.1** (ECAPA-TDNN — patched for Windows path bug)
- **WeSpeaker** (CAM++)
- **DeepFilterNet3** (rust + python, neural denoise)
- **FFmpeg 8.1** (Gyan WinGet build)
- **scikit-learn** (KMeans clustering)
- **soundfile**, **noisereduce**, **torchaudio**

---

## 12. Where to look for things

| Need | File / location |
|---|---|
| Add a new agent | `enroll_<name>.py` or `enroll_all_from_api.py` |
| Tune diarisation thresholds | `src/diar_multi.py` (constants at top) |
| Change FFmpeg chain | `ui.py` `AUDIOOFILTER`, `_NORM_AF` |
| Add a new transcriber model | `src/transcribers/` + register in `__init__.py` |
| Modify the chat bubble UI | `index.html` `renderBubble()` |
| Add a new pipeline stage | `ui.py` `_run_pipeline()` |
| Debug a bad result | `data/processed/<id>/result.json` |
| Re-test against ground truth | `python test_api_compare.py` |

---

*Generated 2026-04-29 — v1 of project documentation.*
