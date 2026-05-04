# SST-Models: Function Index

**Quick lookup: "What function does X?" Reference**

---

## INDEX BY FUNCTION NAME

### A
| Function | File | Purpose |
|----------|------|---------|
| `clean_audio()` | src/audio_cleanup.py | Apply FFmpeg filters to audio |
| `classify_agent_vs_customer()` | src/diar_multi.py | Classify segment as AGENT or CUSTOMER |
| `classify_role()` | src/conversation_roles.py | Classify speaker role |
| `cluster_per_bucket()` | enroll_multi_advanced.py | Run K-means clustering per SNR bucket |
| `compute_f1_score()` | test_api_vs_ours.py | Calculate segment-level F1 accuracy |
| `compute_snr()` | src/voiceprints.py | Estimate Signal-to-Noise Ratio |
| `compute_snr_per_clip()` | enroll_multi_advanced.py | Calculate SNR for specific segment |
| `create_wav_chunk()` | src/embedding_campp.py | Create audio chunk for embedding |

---

### D
| Function | File | Purpose |
|----------|------|---------|
| `diarize()` | src/diar_ecapa.py | Diarization using ECAPA (legacy) |
| `diarize_multi()` | src/diar_multi.py | ⭐ Main diarization using multi-VP |
| `download_batch()` | enroll_multi_advanced.py | Download API calls for training |
| `download_call()` | tools/legacy/scrape_dataset_api.py | Download single API call |

---

### E
| Function | File | Purpose |
|----------|------|---------|
| `embed()` | src/embedding_campp.py | Extract speaker embedding for segment |
| `extract_agent_samples()` | enroll_multi_advanced.py | Get agent-only speech from call |
| `extract_segment_embedding()` | src/diar_multi.py | Extract embedding for specific time window |

---

### G
| Function | File | Purpose |
|----------|------|---------|
| `get_agent_voiceprints()` | src/speaker_matcher.py | Get all VP paths for agent |
| `get_agent_voiceprints()` | src/voiceprints.py | Get specific agent VP metadata |
| `get_role_labels()` | src/conversation_roles.py | Get standard speaker roles |
| `get_vad()` | src/target_speaker_vad.py | Load VAD model |

---

### I
| Function | File | Purpose |
|----------|------|---------|
| `iterative_tighten()` | enroll_multi_advanced.py | Remove outlier embeddings |

---

### L
| Function | File | Purpose |
|----------|------|---------|
| `load_and_validate_audio()` | process_audio.py | Load and validate input audio |
| `load_mp3_mono_16k()` | enroll_multi_advanced.py | Load MP3 as 16kHz mono |
| `load_voiceprints()` | src/diar_multi.py | Load multi-VP stacks |
| `load_voiceprints()` | src/voiceprints.py | Load agents.json + VP files |

---

### M
| Function | File | Purpose |
|----------|------|---------|
| `main()` | enroll_multi_advanced.py | Train multi-voiceprints (entry point) |
| `main()` | test_api_vs_ours.py | Test against API ground truth |
| `main()` | test_voiceprints_api.py | Test held-out API calls |
| `main()` | test_top_5_agents_ui.py | Test UI with top 5 agents |
| `main()` | test_long_calls.py | Test 3+ minute long calls |
| `main()` | tools/legacy/scrape_dataset_api.py | Download API data |
| `match()` | src/speaker_matcher.py | Find best matching agents for embedding |
| `match_agent()` | process_audio.py | Identify agent from segments |

---

### P
| Function | File | Purpose |
|----------|------|---------|
| `parse_speaker_json()` | tools/legacy/scrape_dataset_api.py | Extract speaker labels from API |
| `process_audio()` | process_audio.py | ⭐ Main pipeline orchestrator |

---

### S
| Function | File | Purpose |
|----------|------|---------|
| `save_voiceprints()` | enroll_multi_advanced.py | Write agents.json with multi-VP metadata |
| `select_held_out_calls()` | test_voiceprints_api.py | Find calls not in training set |

---

### T
| Function | File | Purpose |
|----------|------|---------|
| `test_call()` | test_api_vs_ours.py | Test single API call |
| `transcribe()` | src/transcription.py | Speech-to-text (abstract) |
| `transcribe()` | src/transcribers/whisper_turbo.py | Whisper transcription via Groq API |
| `transcribe()` | src/transcribers/parakeet_v3.py | Parakeet transcription |

---

### U
| Function | File | Purpose |
|----------|------|---------|
| `upload_and_wait()` | test_top_5_agents_ui.py | Upload audio and wait for result |

---

### V
| Function | File | Purpose |
|----------|------|---------|
| `voiceprint_inventory()` | src/voiceprints.py | Get VP statistics (count by agent, bucket) |

---

## INDEX BY PURPOSE

### Audio Preparation
| Purpose | Function | File |
|---------|----------|------|
| Clean/filter audio | `clean_audio()` | src/audio_cleanup.py |
| Load MP3 file | `load_mp3_mono_16k()` | enroll_multi_advanced.py |
| Load and validate | `load_and_validate_audio()` | process_audio.py |

---

### Embedding & Feature Extraction
| Purpose | Function | File |
|---------|----------|------|
| Extract speaker embedding | `embed()` | src/embedding_campp.py |
| Extract segment embedding | `extract_segment_embedding()` | src/diar_multi.py |
| Estimate SNR | `compute_snr()` | src/voiceprints.py |
| Compute SNR per clip | `compute_snr_per_clip()` | enroll_multi_advanced.py |

---

### Diarization (Speaker ID)
| Purpose | Function | File |
|---------|----------|------|
| ⭐ Main diarization | `diarize_multi()` | src/diar_multi.py |
| Legacy ECAPA | `diarize()` | src/diar_ecapa.py |
| Classify AGENT vs CUSTOMER | `classify_agent_vs_customer()` | src/diar_multi.py |

---

### Agent Matching
| Purpose | Function | File |
|---------|----------|------|
| Find best agent | `match()` | src/speaker_matcher.py |
| Match agent (pipeline) | `match_agent()` | process_audio.py |
| Load agent VPs | `get_agent_voiceprints()` | src/speaker_matcher.py |

---

### Voiceprint Management
| Purpose | Function | File |
|---------|----------|------|
| Load all VPs | `load_voiceprints()` | src/voiceprints.py |
| Get agent VPs | `get_agent_voiceprints()` | src/voiceprints.py |
| VP statistics | `voiceprint_inventory()` | src/voiceprints.py |

---

### Enrollment & Training
| Purpose | Function | File |
|---------|----------|------|
| ⭐ Train all agents | `main()` | enroll_multi_advanced.py |
| Extract agent speech | `extract_agent_samples()` | enroll_multi_advanced.py |
| Bucket by SNR | `bucket_by_snr()` | enroll_multi_advanced.py |
| K-means clustering | `cluster_per_bucket()` | enroll_multi_advanced.py |
| Remove outliers | `iterative_tighten()` | enroll_multi_advanced.py |
| Save VPs | `save_voiceprints()` | enroll_multi_advanced.py |
| Download API data | `download_batch()` | enroll_multi_advanced.py |

---

### Transcription
| Purpose | Function | File |
|---------|----------|------|
| Speech-to-text | `transcribe()` | src/transcription.py |
| Whisper (via Groq) | `transcribe()` | src/transcribers/whisper_turbo.py |
| Parakeet | `transcribe()` | src/transcribers/parakeet_v3.py |

---

### Pipeline & Processing
| Purpose | Function | File |
|---------|----------|------|
| ⭐ Main pipeline | `process_audio()` | process_audio.py |

---

### Testing & Validation
| Purpose | Function | File |
|---------|----------|------|
| Test vs ground truth | `main()` | test_api_vs_ours.py |
| Test single call | `test_call()` | test_api_vs_ours.py |
| Calculate F1 score | `compute_f1_score()` | test_api_vs_ours.py |
| Test held-out data | `main()` | test_voiceprints_api.py |
| Test via UI | `main()` | test_top_5_agents_ui.py |
| Upload to UI | `upload_and_wait()` | test_top_5_agents_ui.py |
| Test long calls | `main()` | test_long_calls.py |

---

### Data Handling
| Purpose | Function | File |
|---------|----------|------|
| Download API calls | `main()` | tools/legacy/scrape_dataset_api.py |
| Download single call | `download_call()` | tools/legacy/scrape_dataset_api.py |
| Parse API labels | `parse_speaker_json()` | tools/legacy/scrape_dataset_api.py |
| Find held-out calls | `select_held_out_calls()` | test_voiceprints_api.py |

---

## FUNCTION SIGNATURES

### Core Pipeline

```python
def process_audio(audio_path, model=None, upload_id=None, status_callback=None):
    """Main audio processing orchestrator"""
    # Returns: dict with result_id, identified_agent, similarity_score, segments

def diarize_multi(audio_path, model, segments, output_path=None):
    """Multi-voiceprint diarization"""
    # Returns: List of segments with speaker labels

def embed(audio_path, start_s, end_s):
    """Extract speaker embedding"""
    # Returns: np.array of shape (512,)

def match(embedding, top_n=5):
    """Find best matching agents"""
    # Returns: [('Agent Name', score), ...]
```

---

### Enrollment

```python
def main(max_calls_per_agent=150, min_agent_calls=5):
    """Train multi-voiceprints"""
    # Modifies: agents.json and creates .npy files

def extract_agent_samples(call_data, agent_name):
    """Get agent-only speech"""
    # Returns: List of (audio_clip, start_s, end_s)

def cluster_per_bucket(bucket_embeddings, agent_name, bucket_name):
    """K-means clustering per SNR bucket"""
    # Saves: agent_name__bucket_idx.npy files

def iterative_tighten(embeddings):
    """Remove outliers"""
    # Returns: Filtered embeddings list
```

---

### Testing

```python
def test_call(call_data, model, voiceprints):
    """Test single API call"""
    # Returns: dict with accuracy metrics

def compute_f1_score(predicted_labels, ground_truth_labels):
    """Calculate F1 score"""
    # Returns: float (0-1)

def upload_and_wait(audio_path, api_url="http://localhost:8080"):
    """Upload to UI and wait"""
    # Returns: dict with identified_agent, similarity, segments
```

---

## COMMON WORKFLOWS

### "I want to train agents"
1. `main()` → enroll_multi_advanced.py
   - Calls: `download_batch()`
   - Calls: `extract_agent_samples()`
   - Calls: `bucket_by_snr()`
   - Calls: `cluster_per_bucket()`
   - Calls: `save_voiceprints()`

### "I want to identify a speaker"
1. `process_audio()` → process_audio.py (main)
   - Calls: `clean_audio()` → src/audio_cleanup.py
   - Calls: `transcribe()` → src/transcription.py
   - Calls: `diarize_multi()` → src/diar_multi.py
     - Calls: `embed()` → src/embedding_campp.py
     - Calls: `load_voiceprints()` → src/voiceprints.py
     - Calls: `match()` → src/speaker_matcher.py

### "I want to test accuracy"
1. `main()` → test_api_vs_ours.py
   - Calls: `test_call()`
   - Calls: `compute_f1_score()`

### "I want to validate on UI"
1. `main()` → test_top_5_agents_ui.py
   - Calls: `upload_and_wait()`

---

## ALPHABETICAL FUNCTION LOOKUP

```
A: clean_audio, classify_agent_vs_customer, classify_role, cluster_per_bucket
   compute_f1_score, compute_snr, compute_snr_per_clip, create_wav_chunk

D: diarize, diarize_multi, download_batch, download_call

E: embed, extract_agent_samples, extract_segment_embedding

G: get_agent_voiceprints, get_role_labels, get_vad

I: iterative_tighten

L: load_and_validate_audio, load_mp3_mono_16k, load_voiceprints

M: main, match, match_agent

P: parse_speaker_json, process_audio

S: save_voiceprints, select_held_out_calls

T: test_call, transcribe

U: upload_and_wait

V: voiceprint_inventory
```

---

## FUNCTION DEPENDENCIES

```
process_audio()
├── clean_audio()
├── transcribe()
└── diarize_multi()
    ├── embed()
    ├── load_voiceprints()
    └── match()

enroll_multi_advanced.main()
├── download_batch()
├── extract_agent_samples()
├── compute_snr_per_clip()
├── bucket_by_snr()
├── cluster_per_bucket()
├── iterative_tighten()
└── save_voiceprints()

test_api_vs_ours.main()
├── test_call()
└── compute_f1_score()

test_top_5_agents_ui.main()
└── upload_and_wait()
```

---

## PARAMETER GUIDE

### Common Parameters

| Parameter | Type | Example | Purpose |
|-----------|------|---------|---------|
| `audio_path` | str | "data/raw_calls/upload.mp3" | Input audio file path |
| `model` | str or obj | "whisper" or EmbeddingModel() | AI model to use |
| `embedding` | np.ndarray | shape (512,) | Speaker embedding |
| `segments` | list | [{'text': '...', 'start': 0.0, 'end': 1.5}] | Transcribed segments |
| `agent_name` | str | "Omar El Harchaoui" | Speaker name |
| `bucket_name` | str | "high", "mid", "low" | SNR quality bucket |
| `threshold` | float | 0.55 | Confidence threshold (0-1) |
| `top_n` | int | 5 | Number of results to return |
| `max_calls` | int | 150 | Maximum training calls per agent |

---

## RETURN VALUE GUIDE

### Common Return Types

```python
# Embeddings
np.ndarray of shape (512,)  # Speaker embedding from CAM++
np.ndarray of shape (n, 512) # Stack of embeddings

# Results
{
    'identified_agent': 'Agent Name' or 'CUSTOMER',
    'agent_similarity': 0.681,
    'segments': [{...}],
    'result_id': 'unique_id'
}

# Accuracy Metrics
{
    'accuracy': 0.70,
    'f1_score': 0.815,
    'correct': 7,
    'total': 10
}

# Voiceprints Dict
{
    'agent_name': {
        'voiceprints': [{...}],
        'paths': np.ndarray
    }
}
```

---

**Last Updated**: 2026-05-04  
**Version**: 1.0  
**Use**: For quick function lookup and understanding relationships
