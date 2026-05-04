# Multi-Voiceprint Training & Testing Flow

End-to-end explanation of how we go from "raw recordings on the Audiofy API"
to "a matcher that can identify the right agent in a noisy desk recording."

## Why we changed the design

Old approach: one ECAPA centroid per agent (single 192-dim vector). Worked
fine on clean phone calls (the kind of audio we trained on), but degraded on
**desk recordings** — laptop microphone, room reverb, fans, keyboards, etc.
A single averaged centroid is the agent's "average voice in clean
conditions"; it doesn't know what the agent sounds like through noise.

New approach: **multiple voiceprints per agent**, organised by audio quality
bucket (low / mid / high SNR). At inference time, an unknown segment is
matched against **every** centroid for an agent, and we take the closest one
(`max-cosine`). A noisy desk window will naturally land closer to a centroid
trained from the agent's noisy phone calls than to one trained from clean
calls — so it still matches.

```
            ┌─────────────────────────────────────────────────┐
            │  Old: 1 centroid/agent  →  fragile on noise     │
            │                                                 │
            │  agent ─→  [192-d vector]  ←─  unknown segment  │
            │                       cosine                    │
            └─────────────────────────────────────────────────┘

            ┌─────────────────────────────────────────────────┐
            │  New: N centroids/agent  →  robust on noise     │
            │                                                 │
            │  agent ─→  [c_high_0]                           │
            │         →  [c_high_1]                           │
            │         →  [c_mid_0 ]   ←─  unknown segment     │
            │         →  [c_low_0 ]   max-cosine              │
            └─────────────────────────────────────────────────┘
```

---

## Pipeline overview

```
   1. SCRAPE                2. ENROLL                  3. INFERENCE / TEST
 ┌──────────────┐         ┌──────────────────┐        ┌────────────────────┐
 │  Audiofy API │  POST   │ enroll_multi_    │        │ diar_voiceprint    │
 │   payload    │ ─────►  │ from_api.py      │ ─────► │ (used in           │
 │ {start,end,  │         │                  │        │  process_audio.py) │
 │  limit,skip} │         │  • slice agent   │        │                    │
 └──────┬───────┘         │    phrases       │        │  + test_voiceprints│
        │                 │  • SNR bucket    │        │     _api.py        │
        ▼                 │  • k-means/bucket│        │  + test_voiceprints│
 data/audiofy/_dataset/   │  • save N vps    │        │     _desk.py       │
   audio/<id>.mp3         │                  │        │                    │
   index.json             └────────┬─────────┘        └─────────┬──────────┘
   (ground-truth                   │                            │
    speaker_json)                  ▼                            ▼
                            data/agent_voiceprints/      result.json with
                              <slug>.npy        (legacy)   identified agent
                              <slug>__high_0.npy           + per-segment
                              <slug>__high_1.npy             AGENT/CUSTOMER
                              <slug>__mid_0.npy              labels
                              <slug>__low_0.npy
                              agents.json
```

---

## 1. Scraping recordings from the API

**Script:** `tools/legacy/scrape_dataset_api.py`

Hits `POST /api/desk-streamer/get-recording-for-dataset` on
`https://cp.audiofy.co.uk` with a payload like:

```json
{
  "start_time": "2026-04-01T00:00:00.000Z",
  "end_time":   "2026-05-01T00:00:00.000Z",
  "limit":      100,
  "skip":       0
}
```

Auth comes from `.env` → `AUDIOFY_API_TOKEN` (already populated).

For every recording the API returns a `speaker_json` array — phrase-level
ground truth labelled by Audiofy's own diarizer:

```json
[
  {"start":"00:00:01.99","end":"00:00:02.31","speaker":"Omar El Harchaoui","phrase":"Hello.","avg_score":0.91},
  {"start":"00:00:04.31","end":"00:00:04.97","speaker":"Customer","phrase":"Speaking, yes.","avg_score":0.75},
  ...
]
```

That's the gold dataset that lets us trim audio to **agent-only** segments
deterministically — we don't need to run our own diarization on the training
calls.

Outputs:
- `data/audiofy/_dataset/audio/<_id>.mp3` — downloaded MP3 per call.
- `data/audiofy/_dataset/index.json` — flat per-call list with the
  `speaker_json` and metadata.

Run: `python tools/legacy/scrape_dataset_api.py --days 30 --max-calls 300`

---

## 2. Enrollment — building multiple voiceprints per agent

**Script:** `enroll_multi_from_api.py`

For each agent who has at least `--min-calls` (default 5) recordings in the
index, we use up to `--max-calls-per-agent` (default 5) calls and run:

### 2a. Trim to agent-only audio

For each call:
- Load the MP3 once → 16 kHz mono numpy array (one ffmpeg call).
- Walk `speaker_json`. Keep only phrases where
  `speaker is not None and speaker != "Customer"` — these are the agent's
  phrases. Customer phrases and any unlabelled segment are **dropped**.
- Concatenate the kept slices in time order. We never include even a
  half-second of customer audio in the training data — that has been the
  single biggest source of mis-labelled calls historically.

### 2b. SNR bucketing

The same agent recorded over a clean phone line vs a poor cell connection
sounds different to ECAPA. We don't want a single centroid that averages
both — we want **separate** centroids so each one is tight. Bucketing is
how we keep them apart.

For each call's agent-only audio we compute:

```
snr_db = 20 · log10( P90(rms_50ms) / P10(rms_50ms) )
```

(90th-percentile frame RMS sits inside speech; 10th-percentile sits in the
inter-phrase noise floor — robust to outlier clicks, no model dependency).

Bucket assignment:

| bucket | snr_db        |
|--------|---------------|
| high   | ≥ 15          |
| mid    | 8 – 15        |
| low    | < 8           |

The whole call goes into one bucket — the noise floor is uniform within a
single recording.

### 2c. Sliding-window embeddings

For each call we compute ECAPA embeddings on a 2-second sliding window with
1-second stride (same as the legacy enroller). Each embedding is tagged with
the call's bucket.

### 2d. Per-bucket clustering

For each bucket that has at least `MIN_BUCKET_EMBS` (30) embeddings:

1. **L2-normalise** every embedding (cosine space).
2. **Iterative tightening** (`iterative_tighten`, reused from
   `enroll_all_from_api.py`): compute the bucket centroid; drop windows
   whose cosine to the centroid is < 0.45; recompute; drop again at < 0.55.
   This kills any residual customer leakage that snuck through the
   speaker_json filter (e.g. brief customer back-channels at phrase edges).
3. **K-means** on the kept embeddings, `k = min(2, n_kept // 30)`. So a
   bucket with ≥ 60 kept windows produces 2 cluster centroids; a smaller
   bucket produces 1. We use cosine k-means (sklearn KMeans + L2-normalise
   each cluster's mean).
4. Save each cluster centroid as its own `.npy`:

```
data/agent_voiceprints/
   omar_el_harchaoui.npy             # legacy mean (HIGH bucket only)
   omar_el_harchaoui__high_0.npy     # multi-VP entries
   omar_el_harchaoui__high_1.npy
   omar_el_harchaoui__mid_0.npy
   omar_el_harchaoui__low_0.npy
   agents.json
```

### 2e. agents.json schema

```json
{
  "omar_el_harchaoui": {
    "agent_name":      "Omar El Harchaoui",
    "voiceprint_path": ".../omar_el_harchaoui.npy",
    "voiceprints": [
      {"path": ".../omar_el_harchaoui__high_0.npy", "bucket": "high", "n_clips": 87, "snr_db": 19.4},
      {"path": ".../omar_el_harchaoui__high_1.npy", "bucket": "high", "n_clips": 64, "snr_db": 18.1},
      {"path": ".../omar_el_harchaoui__mid_0.npy",  "bucket": "mid",  "n_clips": 51, "snr_db": 11.8},
      {"path": ".../omar_el_harchaoui__low_0.npy",  "bucket": "low",  "n_clips": 32, "snr_db":  6.2}
    ],
    "n_voiceprints":   4,
    "total_seconds":   735.1,
    "used_calls":      5,
    "source":          "multi_vp_v1",
    "per_call_snr": [
      {"_id":"...","snr_db":18.7,"bucket":"high","embs":62},
      {"_id":"...","snr_db":11.2,"bucket":"mid", "embs":48},
      ...
    ]
  }
}
```

`voiceprint_path` (legacy single-vector field) is still written so old code
keeps working — it's set to the **mean of the HIGH bucket only** so the
single-VP path doesn't get diluted by low-quality samples.

`per_call_snr` is the audit trail. The test scripts use it to identify
held-out calls (anything in `index.json` whose `_id` is **not** in any
agent's `per_call_snr`).

Run: `python enroll_multi_from_api.py --min-calls 5 --max-calls-per-agent 5`

---

## 3. Inference / matching

**Files:** `src/diar_voiceprint.py`, `src/speaker_matcher.py`

Both load every centroid for every agent into a single `(N, 192)` stack per
agent, then for an unknown embedding `e` they compute:

```python
similarity = max(stack @ e)        # one cosine per centroid → take the largest
```

The rest of the pipeline (top-30 % mean to pick the call's agent, threshold
to label AGENT vs CUSTOMER per segment, neighbour-vote smoothing for short
segments) is unchanged. `process_audio.py` and `run_e2e.py` go through this
matcher transparently — no caller-side changes.

Backwards compatibility: if an agent has only `voiceprint_path` (no
`voiceprints` list), the loader returns a (1, 192) stack — `max-cosine` over
a length-1 stack equals plain cosine, so legacy entries behave exactly as
before.

---

## 4. Testing

### 4a. API held-out accuracy — `test_voiceprints_api.py`

For each held-out call:
- For each phrase in `speaker_json`: embed the audio slice, max-cosine
  against every agent stack, predict AGENT (if best score ≥ threshold) or
  CUSTOMER (otherwise), and check whether the predicted agent matches the
  call's `agent_name`.
- Aggregate to: per-segment AGENT P/R/F1, per-call "right agent picked?"
  rate, per-bucket breakdown (was the held-out call high / mid / low SNR).
- Run the **same** evaluation again with `multi=False` (legacy single-VP)
  and print a head-to-head: where multi-VP fixed a single-VP miss, where
  multi-VP regressed.

This isolates the speaker-ID part of the system — a regression here is
unambiguously the multi-VP change's fault.

Run: `python test_voiceprints_api.py --top 30`

### 4b. Desk-recording sanity — `test_voiceprints_desk.py`

No phrase-level ground truth, so we can't compute precision/recall. Instead:
- Sliding 1.5 s window with 0.75 s stride over each MP3 in
  `testing-audio/{low,mid,high}/`.
- Percentile-RMS VAD drops silent windows.
- Each surviving window is matched against every agent stack (max-cosine),
  the best agent and best score recorded.
- Per call: identified agent (top-30 % mean across windows), top-3
  candidates with scores, and AGENT/CUSTOMER time-share. A real 1-on-1 call
  should be roughly 50/50 — a wildly skewed split is a red flag.
- Per call: write `<filename>.result.json` next to the MP3 for spot-check.

Run: `python test_voiceprints_desk.py`

---

## File map

| Path                                              | Role                                        |
|---------------------------------------------------|---------------------------------------------|
| `tools/legacy/scrape_dataset_api.py`              | API scrape (unchanged)                      |
| `enroll_all_from_api.py`                          | Old single-VP enroller — helpers reused     |
| **`enroll_multi_from_api.py`**                    | **New: multi-VP enroller**                  |
| `src/embedding_campp.py`                          | ECAPA / CAM++ embedding model (unchanged)   |
| `src/diar_voiceprint.py`                          | Modified: max-cosine over stack             |
| `src/speaker_matcher.py`                          | Modified: max-cosine over stack             |
| `src/voiceprints.py`                              | Modified: inventory aware of new list       |
| `process_audio.py` / `run_e2e.py`                 | Unchanged — call the matcher transparently  |
| **`test_voiceprints_api.py`**                     | **New: held-out API accuracy harness**      |
| **`test_voiceprints_desk.py`**                    | **New: desk recording sanity check**        |
| `data/audiofy/_dataset/`                          | Scraped MP3s + ground-truth `index.json`    |
| `data/agent_voiceprints/`                         | Centroid `.npy` files + `agents.json`       |
| `testing-audio/{low,mid,high}/*.mp3`              | Desk recordings                             |

---

## Run order (cold start)

```bash
cd call_processor

# 1. Refresh API recordings (skip if data/audiofy/_dataset/index.json is current)
python tools/legacy/scrape_dataset_api.py --days 30 --max-calls 300

# 2. Train multi-voiceprints
python enroll_multi_from_api.py --min-calls 5 --max-calls-per-agent 5

# 3. Verify we beat (or at least match) single-VP on held-out API calls
python test_voiceprints_api.py --top 30

# 4. Confirm correct agent identified on noisy desk recordings
python test_voiceprints_desk.py
```

Step 3 must show `multi-VP fixed single-VP miss` ≥ `multi-VP regressed`. If
that flips, the most likely cause is that the legacy mean (used by
`voiceprint_path`) is being computed from too few HIGH-bucket calls — fix by
re-running enrollment with more `--max-calls-per-agent`.

Step 4 doesn't have a numeric pass/fail — read each `*.result.json` and
sanity-check that the identified agent matches the person who actually
recorded that desk audio, and that the AGENT/CUSTOMER time share isn't
wildly skewed.

---

## Key decisions, recorded

- **Bucketing first, then clustering.** Clustering raw windows alone splits
  by phonetic content (vowels vs consonants) — useless for our problem.
  Bucketing by SNR forces at least one centroid into the noisy region of
  embedding space, which is what we need at inference for desk recordings.
- **k ≤ 2 per bucket, not arbitrary.** We could let k grow with bucket
  size, but each extra centroid increases the false-positive risk that some
  *other* speaker happens to land near it. Capping at 2 keeps the matcher
  tight while still giving us up to 6 centroids per agent (high×2 +
  mid×2 + low×2) — empirically enough variation to cover phone vs cell vs
  desk.
- **Legacy mean = HIGH bucket only.** When the single-VP code path is used,
  we want that one vector to represent the agent's *clean* voice. Mixing in
  low-bucket samples would degrade single-VP accuracy on clean calls.
- **Customer phrases never enter the training pool.** Even with a 0.5
  `avg_score` threshold the API's labels are imperfect; we lean on the
  speaker_json filter + iterative tightening to keep voiceprints pure.
- **`per_call_snr` is the held-out audit trail.** Without it we'd have no
  way to tell which calls were used for training vs which are safe to test
  on. Every test script trusts this field.
