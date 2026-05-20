# Call Processor — Project Documentation

**Last updated:** 2026-05-20 (branch `exp-overlap-role-fix`)
**Status:** Production-ready. 94.8% identification accuracy vs Audiofy API (measured 2026-04-29, see §4).

> **For voiceprint-file handover** (paths, formats, copy-list) see `VOICEPRINTS_HANDOVER.md` at the repo root.

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
│ 6. Transcribe (default Parakeet TDT v3 on CUDA)                 │
│    Pluggable backends in src/transcribers/ (see §6 for list)    │
│    Falls back to whisper-large-v3-turbo if 0 segments           │
│    `requested_model` + `fallback_used` recorded in result       │
│    Word-timestamp output drives turn splitting in step 7        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. diar_multi.diarize_multi — voiceprint-first speaker ID       │
│    a. Load 52 enrolled voiceprints (192 ECAPA + 512 CAM++)      │
│    b. Per-segment ECAPA/CAM++ embedding (widened ±0.2s if <0.8s)│
│    c. VAD speech-ratio filter (reject low-speech chunks)        │
│    d. Per-agent threshold (max(0.34, max_outside_sim+0.06) ≤0.42)│
│    e. Cluster_first mode (calls ≥30s, ≥10 valid embs)            │
│    f. Centroid-gated soft-reclaim                               │
│    g. Anti-flip smoothing (3 passes)                            │
│    h. Back-channel demotion (low-sim "yeah/okay/yes"-led segs)  │
│    i. Reverse cluster-reconciliation (high-sim → AGENT)         │
│    j. Word-timestamp turn splitting (commit d5c31d2)            │
│    k. Role corrections (back-channels, farewells)               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. ui.py post-processing                                        │
│    _repair_agent_roles_from_voiceprint_clusters                 │
│    _repair_agent_roles_from_segment_voiceprints                 │
│    _repair_agent_roles_from_text_cues                           │
│    _fallback_unknown_text_to_customer                           │
│    _smooth_short_unknown_segments                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. Trim & save result.json                                      │
│    data/processed/<base>__<model>/result.json                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10. User feedback loop (optional, in UI)                        │
│     Click a word → "Set role: AGENT / CUSTOMER" → corrections    │
│     can be promoted into voiceprint training (commit f342d3f+)  │
└─────────────────────────────────────────────────────────────────┘
```

**Verified live thresholds** (`call_processor/src/diar_multi.py`):
`PER_SEG_THRESHOLD=0.34`, `PER_AGENT_MARGIN=0.06`, `AGENT_MIN_MATCHED=3`,
`CLUSTER_FIRST_MIN_DUR=30.0`, `CLUSTER_FIRST_MIN_SEGMENTS=10`,
`CLUSTER_FIRST_AGENT_RATIO=0.16`.

---

## 3. Enrolled agents (52 total, as of 2026-05-20)

### Tier summary (current state)

| Tier | Count | Description |
|---|---:|---|
| **Tier 1** — CAM++ 512-dim, multi-VP (pure) | 6 | Omar, Amandeep, Zak Raissi Barnet, Hussein, Aayush, Anil |
| **Tier 2** — ECAPA/CAM++ multi-VP, SNR-bucketed | 21 | Haris, Allan, Talha, Mohammad Malki, Sylwia, Ideal, Anoush, Harrison, Adil, Angeline, Jason, Kowsar, Georgi, Janusaan, Aftaab, Yasin-ali, Rayyan, Adorena, Waris, Kacper, Dinosh |
| **Tier 3** — Legacy single-vector ECAPA | 25 | Zak local, Mohammed Al Russell, Sarah Aziz, Rebeca, Rajan, Rebecca, Nevethan, Shuahib, Albjon, Qaim, Tulay, Mashrur, Mohammed Malik, Mohammed-Hussein, Sababa, Liza Mae, Benjamin, Niloufar, Rafik, Gabriel, Nirvan, Arfat, Dilayda, Kleo, Jenifer |

### Quality summary (from 2026-04-29 snapshot, when count was 49)

| Quality | Count | Definition |
|---|---:|---|
| GOOD | 19 | inside_sim − outside_sim > 0.15 |
| OK | 18 | inside_sim − outside_sim ∈ (0.05, 0.15] |
| WEAK | 6 | inside_sim − outside_sim ∈ (0, 0.05] |
| BROKEN | 4 | outside_sim ≥ inside_sim — can't separate from customers |
| NO_STATS | 2 | enrolled before stats were recorded — uses global threshold |

> **Note:** 3 new agents have been enrolled since this snapshot was taken. For a current, file-level inventory of every voiceprint (paths, dims, sources), see `VOICEPRINTS_HANDOVER.md` at the repo root. The per-agent quality table below is the last full audit (2026-04-29).

### Full list (2026-04-29 audit)

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

## 5. Improvements log

### Phase 4 — Role-correction feedback + ASR expansion (2026-04-30 → 2026-05-20)

Most recent work, post the 94.8 % accuracy baseline.

16. **Manual role-correction in UI** (commits `f342d3f`, `93ada7a`, `df2c7e6`, `921c865`, `30f159e`, `c322735`, `924464b`)
    — Per-word click → "Set role: AGENT / CUSTOMER". Corrections feed back into voiceprint training when the user marks the correction as verified.
17. **Multi-agent role windows refinement** (commits `720f829`, `08b56fc`, `42aed56`, `7d675b6`)
    — Cleaner voiceprint role-matching when multiple enrolled agents appear in one call; tuned window over-splitting.
18. **Word-timestamp turn splitting** (commit `d5c31d2`)
    — Speaker turns now split using word-level timestamps, not just segment boundaries. Active on `exp-overlap-role-fix` branch.
19. **New ASR backends added** (registry in `src/transcribers/__init__.py`):
    - `granite-speech-4.1-2b` — IBM Granite Speech, research model for noisy English (no native word timestamps)
    - `parakeet-tdt-0.6b-v2` — Parakeet V2 benchmark (commit `8561fdb`)
    - `canary-qwen-2.5b`, `qwen3-asr-1.7b`, `vibevoice-asr` — experimental local
    - `groq-whisper-large-v3` / `-turbo` — Groq cloud Whisper (ultra-fast, no GPU)
    - `assemblyai-universal-2` — AssemblyAI cloud, tuned for call-centre audio
20. **Shareable call links** (commit `d7611ce`) — UI generates a stable URL per call result.
21. **Tightened diarisation thresholds** — `PER_SEG_THRESHOLD` 0.30→0.34, `PER_AGENT_MARGIN` 0.05→0.06, `AGENT_MIN_MATCHED` 5→3, `CLUSTER_FIRST_MIN_DUR` 60→30s, `CLUSTER_FIRST_MIN_SEGMENTS` 15→10. Engages on shorter calls and is less permissive on weak matches.

### Phases 1–3 (2026-04-27 → 2026-04-29) — baseline 94.8 %

#### Phase 1 — Infrastructure (8 fixes)

1. **Per-agent dynamic threshold** — uses `max_outside_sim + 0.05` per voiceprint
2. **Parakeet on GPU** — 5–10× faster than CPU (was forced CPU before)
3. **Transparent fallback** — `requested_model` + `fallback_used` in result.json
4. **RMS-based DFN3 skip** — clean inputs bypass neural denoise (avoids over-processing)
5. **Cancel button + `/api/cancel`** — checked between pipeline stages
6. **Confidence bar in chat bubbles** — green/amber/grey under each bubble
7. **Auto-cleanup of orphan dirs** — `_gc_orphan_processed_dirs()` on startup
8. **17 legacy scripts moved** to `tools/legacy/`

#### Phase 2 — Audio quality (2 critical fixes)

9. **Restored loudnorm** — phone audio was silent for Whisper/Parakeet without it
10. **Replaced `pan=mono|c0=...FL+FR`** with `aformat=channel_layouts=mono` — DFN3 outputs mono, pan filter was producing zero output

#### Phase 3 — Diarisation accuracy (5 iterations)

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

## 6. File organisation (verified 2026-05-20)

```
SST-models/                              # repo root
├── README.md
├── CLAUDE.md                            # AgentMemory + project rules for AI agents
├── VOICEPRINTS_HANDOVER.md              # voiceprint file handover doc (root-level)
├── traning_data/                        # source training audio (note typo "traning")
│   └── <agent>/call_NN/{audio.mp3,data.json}
├── Agents-recoding/                     # clean agent enrollment recordings
│
└── call_processor/                      # main app
    ├── ui.py                            # HTTP server + pipeline orchestrator (~4930 lines)
    ├── index.html                       # Frontend dashboard (~3420 lines)
    ├── PROJECT_DOC.md                   # ← this file
    ├── README.md
    │
    ├── download_models.py               # one-shot model downloader (Parakeet, ECAPA, CAM++)
    ├── process_audio.py                 # CLI: single-call pipeline runner
    ├── run_e2e.py                       # CLI: batch end-to-end
    ├── transcribe_job.py                # CLI: transcribe only (no diarisation)
    │
    ├── src/
    │   │  ── Diarisation / speaker ID ──
    │   ├── diar_multi.py                # Voiceprint-first multi-speaker diariser (PRIMARY)
    │   ├── diar_voiceprint.py           # Voiceprint-only variant
    │   ├── diar_campp.py                # CAM++ flavoured diariser
    │   ├── diar_ecapa.py                # ECAPA flavoured diariser
    │   ├── diar_clean.py                # Clean-mode diariser (for high-SNR inputs)
    │   ├── diarization.py               # Common dispatch / shared types
    │   ├── target_speaker_vad.py        # TS-VAD with enrolled voiceprint
    │   ├── speaker_matcher.py           # Multi-VP cosine matching utilities
    │   ├── speaker_role.py              # ECAPA helpers + enrollment (192-dim)
    │   ├── embedding.py                 # ECAPA wrapper
    │   ├── embedding_campp.py           # CAM++ 512-dim (wespeaker, CPU-forced)
    │   ├── embedding_titanet.py         # TitaNet (NeMo) wrapper
    │   ├── boundary_refinement.py       # Word-timestamp turn splitting (commit d5c31d2)
    │   ├── conversation_roles.py        # AGENT/CUSTOMER role assignment logic
    │   ├── supervised_labels.py         # Map user role corrections → training labels
    │   ├── voiceprints.py               # Voiceprint loader / path resolver
    │   │
    │   │  ── Audio + ASR ──
    │   ├── audio_cleanup.py             # Legacy 10-stage desk-recording cleaner
    │   ├── transcription.py             # Transcriber dispatch helper
    │   ├── pipeline.py                  # Shared pipeline glue (legacy)
    │   ├── utils.py                     # Misc helpers
    │   ├── config.py                    # embedding_model = "cam++", thresholds, paths
    │   │
    │   └── transcribers/                # ASR backends (registered in __init__.py)
    │       ├── base.py                  # BaseTranscriber abstract
    │       ├── parakeet_v3.py           # Parakeet TDT v2 + v3 (default v3)
    │       ├── whisper_turbo.py         # Whisper family via faster-whisper
    │       ├── granite_speech.py        # IBM Granite Speech 4.1 2B (added 929992a)
    │       ├── cohere.py                # Cohere Transcribe
    │       ├── canary_qwen.py           # NVIDIA Canary-Qwen 2.5B
    │       ├── qwen3_asr.py             # Qwen3 ASR 1.7B
    │       ├── vibevoice_asr.py         # VibeVoice
    │       ├── deepgram_asr.py          # Deepgram cloud (Nova-3, Nova-2)
    │       ├── groq_whisper.py          # Groq cloud Whisper
    │       └── assemblyai_asr.py        # AssemblyAI Universal-2 cloud
    │
    ├── scripts/                         # Enrollment, training, evaluation, diagnostics
    │   │  ── Enrollment (the gated entry point is auto_enroll_agent.py) ──
    │   ├── auto_enroll_agent.py         # PRIMARY: leave-one-call-out + 95-96% gates
    │   ├── bulk_reenroll_from_api.py    # Bulk ECAPA re-enrolment
    │   ├── smart_reenroll_phase2.py     # Phase 2 re-enrolment
    │   ├── enroll_clean_agent.py        # Clean single-speaker recordings
    │   ├── enroll_omar_from_gt.py       # Omar from GT speaker_json
    │   ├── extract_and_reenroll.py
    │   ├── add_voiceprint_from_result.py # Promote a result.json segment → voiceprint
    │   ├── build_voiceprint_from_transcript.py
    │   ├── audiofy_profile_scrape.py
    │   ├── prepare_hussein_labels_from_local_results.py
    │   │
    │   │  ── CAM++ pure training (Tier 1) ──
    │   ├── train_omar_pure_embeddings.py
    │   ├── train_zak_pure_embeddings.py
    │   ├── train_ds_voiceprints.py
    │   ├── enhanced_train.py            # Re-train with inline+npy support (see memory: VP loader bug)
    │   ├── combine_and_retrain.py
    │   ├── train_agent_from_api_labels.py
    │   ├── train_from_gemini_labels.py  # Legacy: Gemini-labelled training
    │   ├── train_zak_from_gemini.py
    │   ├── train_titanet.py
    │   │
    │   │  ── Evaluation / diagnostics ──
    │   ├── evaluate_agent_loco.py       # leave-one-call-out evaluation
    │   ├── evaluate_zak_cluster_first.py
    │   ├── evaluate_zak_hybrid.py
    │   ├── evaluate_zak_voiceprints.py
    │   ├── verify_identification_e2e.py
    │   ├── test_accuracy_on_api_calls.py
    │   ├── test_prod_upload.py
    │   ├── test_deep_enhance.py
    │   ├── test_enhancement_variants.py
    │   ├── time_pipeline.py
    │   ├── compare_cloud_asr_on_call.py
    │   ├── ab_test_amandeep.py
    │   ├── check_agent_intro_whisper.py
    │   ├── deepgram_transcribe_folder.py
    │   ├── deepgram_zak_compare.py
    │   ├── fetch_gt_for_file.py
    │   ├── fix_mojibake.py
    │   ├── daily_training_daemon.py     # Auto re-train daemon
    │   ├── gemini_api_train.py          # Legacy Gemini browser path
    │   ├── gemini_auto_train.py
    │   └── gemini_browser_auto.py
    │
    ├── data/
    │   ├── raw_calls/                   # Original uploads + enhanced_<file>.mp3 + df_<file>.mp3
    │   ├── processed/                   # Per-(call×model) result dirs → result.json
    │   ├── agent_voiceprints/           # 219 .npy + agents.json (see VOICEPRINTS_HANDOVER.md)
    │   │   ├── agents.json              # ← INDEX, source of truth
    │   │   ├── *.npy                    # 52 agents (some multi-VP)
    │   │   ├── _candidates/             # In-progress experiments (do not ship)
    │   │   ├── backup_20260504T151730/  # Old backup
    │   │   ├── daily_reports/           # Per-day training audit
    │   │   └── agents.backup.*.json     # ~30 timestamped backups (inert)
    │   ├── agent_clean_clips/           # Pure single-speaker clips by agent/source
    │   ├── agent_samples/               # Reference samples for QA
    │   ├── training/                    # Curated training data (copied from traning_data/)
    │   ├── ground_truth/                # GT speaker_json files
    │   ├── audiofy/                     # Audiofy API datasets
    │   ├── api_compare/                 # API-vs-local comparison outputs
    │   ├── desk_recordings_cache/       # Desk-recording cache
    │   ├── deepgram_zak_eval/           # Deepgram baseline for Zak
    │   ├── provider_compare/            # Cross-provider comparison
    │   └── logs/                        # Server / pipeline logs
    │
    └── tools/legacy/                    # Archived one-off scripts (including scrape_dataset_api.py)
```

### What moved since the 2026-04-29 version of this doc

- `enroll_*.py`, `test_*.py`, `build_*.py` **moved from `call_processor/` root → `call_processor/scripts/`**.
- New top-level CLIs: `process_audio.py`, `run_e2e.py`, `transcribe_job.py`, `download_models.py`.
- New `src/` modules: `boundary_refinement.py`, `conversation_roles.py`, `diar_campp.py`, `diar_clean.py`, `diar_ecapa.py`, `diar_voiceprint.py`, `diarization.py`, `pipeline.py`, `speaker_matcher.py`, `supervised_labels.py`, `target_speaker_vad.py`, `transcription.py`, `utils.py`.
- New transcribers: `granite_speech.py`, `canary_qwen.py`, `qwen3_asr.py`, `vibevoice_asr.py`, `groq_whisper.py`, `assemblyai_asr.py`, plus `parakeet_v3.py` now exposes V2 too.
- New data subdirs: `api_compare/`, `desk_recordings_cache/`, `deepgram_zak_eval/`, `ground_truth/`, `logs/`, `provider_compare/`, `training/`.

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

- **Python 3.11.9** (Windows, conda/winget Python)
- **PyTorch 2.5.1 + CUDA 12.1**
- **NeMo** (Parakeet TDT v2/v3, TitaNet, Canary-Qwen)
- **HuggingFace Transformers** (Whisper variants via faster-whisper, IBM Granite Speech, Qwen3-ASR, VibeVoice)
- **SpeechBrain 1.1** (ECAPA-TDNN — patched for Windows path bug)
- **WeSpeaker** (CAM++, forced to CPU due to GPU device mismatch)
- **DeepFilterNet3** (rust + python, neural denoise, DNS-4 trained)
- **FFmpeg 8.1** (Gyan WinGet build)
- **scikit-learn** (KMeans clustering)
- **soundfile**, **noisereduce**, **torchaudio**, **librosa**
- **Cloud ASR (optional, API-keyed):** Deepgram (Nova-3 / Nova-2), Groq (Whisper Large v3 / turbo), AssemblyAI (Universal-2), Cohere Transcribe

**Env:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set at startup to prevent VRAM fragmentation after Parakeet/ECAPA runs.

---

## 12. Where to look for things

| Need | File / location |
|---|---|
| Add a new agent (gated, recommended) | `scripts/auto_enroll_agent.py` |
| CAM++ pure training (Tier 1) | `scripts/train_omar_pure_embeddings.py`, `train_zak_pure_embeddings.py` |
| Tune diarisation thresholds | `src/diar_multi.py` (constants at top of file) |
| Change FFmpeg DSP chain | `ui.py` — `AUDIOOFILTER`, `_NORM_AF` |
| Add a new transcriber model | `src/transcribers/` + register in `__init__.py` |
| Word-timestamp turn splitting | `src/boundary_refinement.py` |
| Map UI role corrections → training data | `src/supervised_labels.py` |
| Modify chat bubble / role-correction UI | `index.html` — `renderBubble()` |
| Add a new pipeline stage | `ui.py` — `_run_pipeline()` |
| Debug a bad result | `data/processed/<id>/result.json` |
| Re-test against ground truth | `scripts/test_accuracy_on_api_calls.py` |
| Run leave-one-call-out eval | `scripts/evaluate_agent_loco.py` |
| Voiceprint file handover (sharing with another dev) | `../VOICEPRINTS_HANDOVER.md` (repo root) |

---

*Generated 2026-04-29 — v1 of project documentation.*
*Updated 2026-05-20 — refresh for branch `exp-overlap-role-fix`: Phase 4 log, expanded transcribers, scripts/ migration, 52-agent count, current `diar_multi` thresholds.*
