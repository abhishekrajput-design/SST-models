# V2 Live Pipeline Results

Date: 2026-04-29
Live server: `http://13.42.127.218:8080`
Current V2 code commit before this document: `1eb5473 Use cluster-first speaker roles on long calls`

## Current V2 Pipeline

| Stage | V2 choice | Purpose | Notes |
|---|---|---|---|
| Audio preparation | FFmpeg normalization, DeepFilterNet if installed | Produce clean 16 kHz mono audio for ASR and speaker embeddings | Live logs show FFmpeg path active; DeepFilterNet may fall back if not installed |
| Transcription option 1 | `parakeet-tdt-0.6b-v3` through NeMo, CPU mode | Stable Parakeet transcription on this machine | Better average role accuracy in live long-call tests |
| Transcription option 2 | `distil-whisper-large-v3.5` | Faster local Whisper transcription | Better average word accuracy and speed in live long-call tests |
| Speaker embedding | WeSpeaker CAM++ 512-dim embeddings | Agent voiceprint match and role separation | Active voiceprint backend for enrolled agents |
| Long-call role assignment | `cluster_first_voiceprint` | Separate speaker clusters first, then map the closest cluster to the enrolled agent | Fixes long-call over-detection of customer speech as agent |
| Short-call role assignment | `per_segment_similarity` plus short-reply correction | Preserve high short-call role accuracy | Short Omar test reached 97.05% role accuracy |
| Pyannote | Diagnostic only, not default | Alternative diarization test | Slower and not better on the 7.90 min diagnostic call |

## Live Model Comparison

These runs were executed on the live server through the real `/api/upload` UI path and compared against Audiofy API `speaker_json` / transcript references.

| Call | Agent | Dur | Model | Mode | Segs | Proc | Speed | Word Acc | WER | Role Acc | Agent Rec | Cust Rec |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `69efc3acf91ac02559f83a89` | Ideal Dacaj | 7.90m | Parakeet | `cluster_first_voiceprint` | 163 | 127.6s | 3.71x | 69.36% | 30.64% | 92.12% | 66.35% | 79.13% |
| `69efc3acf91ac02559f83a89` | Ideal Dacaj | 7.90m | Distil v3.5 | `cluster_first_voiceprint` | 143 | 82.9s | 5.72x | 75.07% | 24.93% | 84.81% | 77.22% | 83.86% |
| `69efb362f91ac02559f813fe` | Omar El Harchaoui | 7.89m | Parakeet | `per_segment_similarity` | 167 | 171.0s | 2.77x | 48.45% | 51.55% | 88.41% | 80.76% | 55.44% |
| `69efb362f91ac02559f813fe` | Omar El Harchaoui | 7.89m | Distil v3.5 | `per_segment_similarity` | 217 | 91.8s | 5.16x | 54.20% | 45.80% | 79.27% | 86.84% | 62.01% |
| `69efa1a8f91ac02559f7e4ee` | Anoush Sefatzadeh | 8.37m | Parakeet | `per_segment_similarity` | 142 | 183.9s | 2.73x | 55.57% | 44.43% | 89.57% | 76.54% | 81.23% |
| `69efa1a8f91ac02559f7e4ee` | Anoush Sefatzadeh | 8.37m | Distil v3.5 | `per_segment_similarity` | 128 | 104.6s | 4.80x | 65.85% | 34.15% | 89.83% | 91.15% | 80.81% |

## Average Results

| Model | Avg Proc | Avg Speed | Avg Word Acc | Avg WER | Avg Role Acc | Avg Agent Rec | Avg Cust Rec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parakeet | 160.8s | 3.07x | 57.80% | 42.20% | 90.03% | 74.55% | 71.93% |
| Distil v3.5 | 93.1s | 5.23x | 65.04% | 34.96% | 84.64% | 85.07% | 75.56% |

## Recommendation

Keep V2 speaker identification as CAM++ voiceprints with cluster-first protection for long calls. It fixed the long-call customer-to-agent over-detection problem while keeping the live path fast.

For transcription model choice:

| Need | Recommended model | Reason |
|---|---|---|
| Best average word accuracy and speed | `distil-whisper-large-v3.5` | 5.23x average speed and 65.04% average word accuracy on the three long live calls |
| Best average speaker-role accuracy | `parakeet-tdt-0.6b-v3` | 90.03% average role accuracy on the three long live calls |
| Stable Parakeet on this live machine | Parakeet CPU mode only | CUDA Parakeet previously crashed the UI process |

## Live UI Visibility

The individual comparison runs are visible in the live UI call list because each run has a `data/processed/<result_id>/result.json` file.

Visible result IDs:

```text
enhanced_api_69efc3acf91ac02559f83a89_long_clusterfix__parakeet-tdt-0.6b-v3
enhanced_api_69efc3acf91ac02559f83a89_cmp_distil_v35__distil-whisper-large-v3.5
enhanced_api_69efb362f91ac02559f813fe_cmp_parakeet__parakeet-tdt-0.6b-v3
enhanced_api_69efb362f91ac02559f813fe_cmp_distil_v35__distil-whisper-large-v3.5
enhanced_api_69efa1a8f91ac02559f7e4ee_cmp_parakeet2__parakeet-tdt-0.6b-v3
enhanced_api_69efa1a8f91ac02559f7e4ee_cmp_distil_v35__distil-whisper-large-v3.5
```

The combined comparison table is not currently rendered as a dashboard view. Local artifacts:

```text
call_processor/data/api_compare/model_compare_long_live/comparison.md
call_processor/data/api_compare/model_compare_long_live/comparison.json
```

## Agent Memory Note

Use this note for future memory updates:

```text
V2 live call_processor pipeline is deployed on main. The speaker stack uses WeSpeaker CAM++ 512-dim voiceprints. Long calls use cluster_first_voiceprint in src/diar_multi.py to cluster speaker embeddings first and then map the closest cluster to the enrolled agent; short calls keep per_segment_similarity plus short-reply correction. On three 7.9-8.4 minute live API-backed calls, Parakeet averaged 3.07x speed, 57.80% word accuracy, 42.20% WER, and 90.03% role accuracy. Distil-Whisper v3.5 averaged 5.23x speed, 65.04% word accuracy, 34.96% WER, and 84.64% role accuracy. Keep Parakeet CPU-only on this machine; CUDA Parakeet previously crashed the UI process. Individual result runs are visible in the live UI; combined comparison tables are saved under call_processor/data/api_compare/model_compare_long_live/.
```
