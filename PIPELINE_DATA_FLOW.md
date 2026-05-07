# Call Processor Pipeline — Data Flow Documentation

End-to-end trace of what happens when a user uploads an audio file via the UI.

---

## High-Level Flow

```
[Browser]
   |  POST /api/upload?filename=...&model=parakeet-tdt-0.6b-v3  (binary audio body)
   v
[ui.py do_POST]
   |
   |--(1) Save raw bytes to: data/raw_calls/<filename>
   |--(2) Spawn thread: _run_pipeline(upload_path, filename, model)
   |--(3) Return JSON {"status": "started"} (instant)
   v
[_run_pipeline THREAD]
   |
   |--Stage 0a: FFmpeg light enhance       -> data/processed/<base>/<enhanced_base>.wav
   |--Stage 0b: DeepFilterNet3 denoise     -> data/processed/<base>/<deepfilter>.wav   (skipped if clean)
   |--Stage 1:  Normalize to 16 kHz mono   -> data/processed/<base>/norm_<base>.wav
   |--Stage 2:  Transcriber (Parakeet/etc) -> List[{start, end, text}]
   |--Stage 3:  diar_multi.diarize_multi   -> List[{start, end, text, speaker, identified_speaker, _best_sim}]
   |--Stage 4:  Build result.json          -> data/processed/<base>__<model>/result.json
   v
[Browser polls GET /api/status]
   |
   |  -> sees {running: false, done: true, result_id: "<base>__<model>"}
   |
[Browser GET /api/call/<result_id>]
   |
   |  -> reads data/processed/<result_id>/result.json
   v
[UI renders transcript]
```

---

## Stage-by-Stage Data Flow

### Stage 0 — Upload & Storage (`ui.py:1590`)

**Input:** HTTP POST `/api/upload?filename=<name>&model=<m>` with binary audio body.

**Output:**
- `data/raw_calls/<filename>` — raw uploaded bytes (mp3/wav)
- HTTP response: `{"status": "started", "filename": "...", "model": "..."}`
- Thread spawned: `_run_pipeline(upload_path, filename, whisper_model)`

**Side effects:** `_status` global dict updated to `running=True`. UI polls `/api/status` to track progress.

---

### Stage 0a — FFmpeg Light Normalisation (`ui.py:_enhance_ffmpeg`)

**Input:** `data/raw_calls/<filename>` (any format/sample rate)

**Process:** highpass filter + EBU R128 loudnorm — cleans format issues without altering speech content.

**Output:** `data/processed/<base>/<enhanced_base>.wav` — normalized but NOT denoised yet.

**Variables in `paths` dict:**
```python
paths["ffmpeg"]      # = data/processed/<base>/<enhanced_base>.wav  (this stage's output)
paths["deepfilter"]  # = data/processed/<base>/<dfn_output>.wav     (Stage 0b output)
```

**Skip rule:** If filename starts with `enhanced_`, this stage is skipped (already enhanced).

---

### Stage 0b — DeepFilterNet3 Neural Denoising (`ui.py:_enhance_deepfilternet`)

**Input:** `paths["ffmpeg"]` (output from Stage 0a)

**Process:** PyTorch neural denoiser — removes background noise, line hiss, room reverb.

**Output:** `paths["deepfilter"]` = denoised WAV.

**Skip rule:** If `_is_clean_audio(paths["ffmpeg"])` returns True (synthetic/broadcast audio), skipped — DFN3 over-processes clean audio into silence.

**`pipeline_audio` variable** is set to whichever stage actually ran:
- Skipped DFN3 → `pipeline_audio = paths["ffmpeg"]`
- DFN3 ran    → `pipeline_audio = paths["deepfilter"]`

This is what the transcriber sees.

---

### Stage 1 — Audio Normalisation for Transcription (`ui.py:_make_norm_wav`)

**Input:** `pipeline_audio` (from Stage 0b/0a output)

**Process:** ffmpeg with filter chain:
```
aformat=channel_layouts=mono            (force mono)
aresample=44100                         (upsample to 44.1k for loudnorm — needed because single-pass loudnorm on 8 kHz produces silent output)
loudnorm=I=-16:TP=-1.5:LRA=11           (EBU R128)
dynaudnorm=p=0.9:m=100:s=5              (dynamic compression)
-ar 16000                               (output at 16 kHz mono)
```

**Output:** `norm_wav = data/processed/<base>/norm_<base>.wav` — 16 kHz mono, peak-normalized.

**THIS is the file used for both transcription AND speaker diarization downstream.**

---

### Stage 2 — Transcription (`ui.py:_transcribe_inline`, line 542+)

**Input:** `norm_wav` (16 kHz mono WAV)

**Process:** Selected transcriber model is loaded:
- `parakeet-tdt-0.6b-v3` — runs in **isolated subprocess** to prevent CUDA crashes from killing UI
- `whisper-large-v3*`, `distil-whisper*` — in-process via SpeechBrain
- `deepgram-nova-*`, `cohere-transcribe-*` — REST API calls

**Output:** `segments` = `List[{start: float, end: float, text: str}]`

Example:
```python
[
  {"start": 0.0,  "end": 1.2, "text": "Hello."},
  {"start": 1.5,  "end": 4.1, "text": "Hello, good afternoon, my friend Samuel."},
  ...
]
```

**No speaker labels yet — only text + timestamps.**

---

### Stage 3 — Speaker Diarization & Identification (`call_processor/src/diar_multi.py:diarize_multi`)

**Input:**
- `segments` from Stage 2 (list of `{start, end, text}`)
- `norm_wav` from Stage 1

**Process (this is the complex part):**

1. **Load enrolled voiceprints** from `call_processor/data/agent_voiceprints/agents.json`:
   - For each agent slug, loads `.npy` file paths → stack of `(N_voiceprints, dim)` arrays
   - **NOTE:** Inline arrays in agents.json are silently dropped (loader bug — see memory).

2. **Compute per-segment embedding** via CAM++ (or whichever model is configured):
   - Cut window from `norm_wav` at each `[start, end]`
   - Pad to ≥1.5s if shorter, reject if <0.3s
   - Run through CAM++ → 512-dim L2-normalized vector

3. **Per-segment cosine similarity** to every agent voiceprint → `sims[i, j_agent]`.

4. **Initial labelling** based on similarity vs threshold (per-agent `max_outside_sim + 0.04`, capped at 0.36).

5. **Optional: cluster_first_voiceprint mode** kicks in if:
   - Total duration ≥ `CLUSTER_FIRST_MIN_DUR` (60s default, 30s after this session)
   - Valid embeddings ≥ `CLUSTER_FIRST_MIN_SEGMENTS` (15)
   - Initial agent ratio ≥ `CLUSTER_FIRST_AGENT_RATIO` (0.55 default, 0.10 after this session)

   When triggered: K-means clusters segments into 2 groups, assigns AGENT role to the cluster closer to enrolled voiceprint.

6. **Anti-flip passes 1-3** correct isolated mislabels (short utterances embedded in clear context).

7. **Result:** Each segment gets:
   ```python
   {
     "start": float, "end": float, "text": str,
     "speaker": "SPEAKER_00" | "SPEAKER_01" | ...,    # raw cluster
     "identified_speaker": "AGENT" | "CUSTOMER" | "UNKNOWN",
     "agent_name": "Zak Raissi" (only if AGENT),
     "_best_sim": 0.682,                              # max cosine across enrolled VPs
     "_emb_failed": True/False,                       # set if embedding extraction failed
   }
   ```

8. **Optional overlay: Gemini supervised labels** (if `SST_USE_GEMINI_SUPERVISED_LABELS=1` env var):
   - Calls `src/supervised_labels.apply_supervised_labels()`
   - Looks up matching call_id in `data/training/gemini_labels_*.json`
   - Replaces voiceprint labels with Gemini's perfect labels for matching segments

**Output:** `diar_result` dict with:
```python
{
  "segments": [...],                       # labelled segments
  "agent_name": "Zak Raissi",              # winning agent slug
  "agent_similarity": 0.682,               # mean similarity of agent-cluster segments
  "matched_backend_dim": 512,              # embedding dim used
  "speaker_mode": "voiceprint" | "cluster_first_voiceprint",
  "per_speaker": {"AGENT": {...}, "CUSTOMER": {...}},
  "match_counts": {"zak_local_20260423": 30},
  "cluster_report": {...},                 # if cluster_first ran
}
```

---

### Stage 4 — Build & Persist `result.json` (`ui.py:_transcribe_inline` end)

**Output file:** `data/processed/<base>__<model>/result.json`

**Schema:**
```json
{
  "result_id": "<base>__<model>",
  "audio_file": "norm_<base>.wav",
  "orig_file": "<original_filename>",
  "model_name": "parakeet-tdt-0.6b-v3",
  "processing_time_seconds": 168.5,
  "model_processing_time_seconds": 42.1,
  "audio_duration_seconds": 608.6,
  "transcriber": "parakeet-tdt-0.6b-v3",

  "segments": [
    {
      "start": 0.0, "end": 1.2,
      "text": "Hello.",
      "speaker": "SPEAKER_01",
      "identified_speaker": "CUSTOMER",
      "_best_sim": 0.306,
      "display_speaker": "Customer"
    },
    ...
  ],

  "agent_name": "Zak Raissi",
  "agent_similarity": 0.682,
  "speaker_mode": "voiceprint",
  "speaker_stats": {
    "AGENT":    {"time_s": 245.3, "turns": 25, "first_words": "Hello, good afternoon..."},
    "CUSTOMER": {"time_s": 189.2, "turns": 25, "first_words": "Hello."}
  },
  "transcription_json": [
    {"start": "00:00:00.000", "end": "00:00:01.200", "speaker": "CUSTOMER", "phrase": "Hello.", "avg_score": null},
    ...
  ],

  "enhancement_paths": {
    "ffmpeg": "data/processed/<base>/<base>.wav",
    "deepfilter": "data/processed/<base>/<base>_deepfilter.wav"
  }
}
```

---

### Stage 5 — UI Polling & Render

Browser polls `GET /api/status` every ~1s:
```json
{
  "running": false, "done": true,
  "result_id": "<base>__<model>",
  "stage": "Finalizing",
  "elapsed_seconds": 168.5
}
```

When `done: true`, browser fetches `GET /api/call/<result_id>` → reads `result.json` → renders.

---

## Key File Locations Summary

| Path | Created when | Read when |
|------|--------------|-----------|
| `data/raw_calls/<filename>` | Stage 0 (upload) | Stage 0a |
| `data/processed/<base>/<enhanced>.wav` | Stage 0a | Stage 0b, 1 |
| `data/processed/<base>/<deepfilter>.wav` | Stage 0b | Stage 1 |
| `data/processed/<base>/norm_<base>.wav` | Stage 1 | Stage 2, 3 |
| `data/processed/<base>__<model>/result.json` | Stage 4 | UI render, all subsequent reads |
| `data/agent_voiceprints/agents.json` | enrollment time | Stage 3 |
| `data/agent_voiceprints/<slug>_v<N>.npy` | enrollment time | Stage 3 (loaded by `_load_voiceprints`) |
| `data/training/gemini_labels_*.json` | training scripts | Stage 3 (only if `SST_USE_GEMINI_SUPERVISED_LABELS=1`) |

---

## Where Each Stage Spends Time (8-min call)

| Stage | Time | Bottleneck |
|-------|------|-----------|
| 0a — FFmpeg | ~3-5s | I/O |
| 0b — DeepFilterNet3 | ~30-60s | GPU/CPU compute |
| 1 — Normalise | ~5-10s | ffmpeg loudnorm |
| 2 — Transcription (Parakeet) | ~40-90s | GPU inference (in subprocess) |
| 3 — Diarization (CAM++ embed × N segments) | ~30-60s | CPU embedding extraction |
| 4 — Write result.json | <1s | I/O |

Total: **~2-4 minutes** for an 8-minute call.

---

## Critical Hooks Where We Could Inject Things

1. **Before Stage 2 (transcription):** swap in a different transcriber via `whisper_model` param.
2. **After Stage 2, before Stage 3:** segments exist with text but no speaker labels — **prime injection point for Gemini-based labeling**.
3. **In Stage 3:** `SST_USE_GEMINI_SUPERVISED_LABELS=1` already overlays Gemini labels post-diarization (`src/supervised_labels.py`) — but this lookup matches by call_id against pre-saved labels, doesn't call Gemini live.
4. **After Stage 4:** result.json is final — POST `/api/call/<id>/swap-roles` can flip AGENT↔CUSTOMER manually.

---

## What's NOT in the Pipeline (Today)

- No live Gemini call during inference (pre-saved labels only)
- No stereo channel split (pipeline forces mono — `aformat=channel_layouts=mono`)
- No second-pass model fusion (CAM++ only; ECAPA loaded for Stage 0c MetricGAN+ enhancement only, not for embeddings)
- No active learning feedback loop (corrections via swap-roles aren't fed back to voiceprints)

---

**Now tell me your approach** — I'll know exactly where to plug it in.
