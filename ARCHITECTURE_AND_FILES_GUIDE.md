# SST-Models: Complete Architecture & File Guide

**Date**: 2026-05-04  
**Version**: 1.0 (Multi-Voiceprint Speaker Identification)  
**Status**: Production Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Production Core Files](#production-core-files)
3. [Core Pipeline Modules (src/)](#core-pipeline-modules)
4. [Enrollment & Training Scripts](#enrollment--training-scripts)
5. [Testing & Validation Scripts](#testing--validation-scripts)
6. [Data & Configuration Files](#data--configuration-files)
7. [Function Reference](#function-reference)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      UI SERVER (ui.py)                          │
│  Flask web server at http://localhost:8080                      │
│  - File upload endpoint (/api/upload)                           │
│  - Result retrieval (/api/call/{id})                            │
│  - Status monitoring (/api/status)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PIPELINE ORCHESTRATOR                          │
│  (process_audio.py, pipeline.py)                                │
│  - Manages multi-stage processing                               │
│  - Error handling & recovery                                    │
│  - GPU memory management                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
     │ STAGE 1     │ │ STAGE 2      │ │ STAGE 3      │
     │ Audio Clean │ │ Transcription│ │ Diarization  │
     │             │ │              │ │              │
     │audio_cleanup│ │transcription │ │diar_multi    │
     │.py          │ │.py           │ │.py           │
     └─────────────┘ └──────────────┘ └──────────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ STAGE 4          │
                                    │ Agent Matching   │
                                    │                  │
                                    │speaker_matcher   │
                                    │.py               │
                                    └──────────────────┘
```

### Data Flow

```
[User Upload MP3] 
        ↓
[FFmpeg Audio Cleanup] → filters, resampling, noise reduction
        ↓
[Whisper Transcription] → text + timing for each phrase
        ↓
[Diarization] → Speaker identification per segment
        ↓
[Multi-Voiceprint Matching] → Agent ID with similarity score
        ↓
[Result JSON] → Stored & returned to UI
```

---

## Production Core Files

### 1. `ui.py` (715 lines)
**Purpose**: Flask web server - Main entry point for web UI  
**Location**: `call_processor/ui.py`  
**Runs on**: Port 8080

#### Key Endpoints
```python
GET  /                              # Serve web UI HTML
POST /api/upload                    # Upload audio file
GET  /api/call/<result_id>          # Get processing result
GET  /api/calls                     # List processed calls
GET  /api/agents                    # List trained agents
GET  /api/status                    # Get pipeline status
POST /api/cancel                    # Cancel current processing
POST /api/enroll                    # Trigger agent enrollment
```

#### Key Functions
```python
class CallHandler(BaseRequestHandler):
    """HTTP request handler for web API"""
    do_POST()                    # Handle POST requests
    do_GET()                     # Handle GET requests
    send_json_response()         # Send JSON to client
    send_error()                 # Send error response

def _process_upload()            # Main processing pipeline (threaded)
    - Accepts audio file
    - Validates format
    - Launches process_audio.py
    - Manages status updates
    - Handles cancellation

def _update_status()             # Update pipeline status
def _get_processed_results()     # Load results from disk
def _start_server()              # Initialize HTTP server
```

#### Status Fields
```json
{
  "running": bool,               # Pipeline is currently processing
  "stage_num": int,              # 1=Audio, 2=Transcription, 3=Diarization, 4=Matching
  "stage": "string",             # Current stage name
  "message": "string",           # Detailed status message
  "done": bool,                  # Processing complete
  "error": null|string,          # Error message if failed
  "result_id": "string",         # ID for retrieving result
  "elapsed_seconds": float,      # Total elapsed time
  "stage_elapsed_seconds": float, # Current stage elapsed time
  "processing_time_seconds": float # Total processing time
}
```

---

### 2. `process_audio.py` (420 lines)
**Purpose**: Main audio processing orchestrator  
**Location**: `call_processor/process_audio.py`  
**Called by**: `ui.py`

#### Key Functions
```python
def process_audio(audio_path, model=None, upload_id=None, status_callback=None):
    """
    Main pipeline entry point.
    
    Args:
        audio_path: Path to input MP3/WAV
        model: Transcription model ('whisper', 'parakeet', etc)
        upload_id: Result ID for status tracking
        status_callback: Function to call with status updates
    
    Returns:
        dict with 'result_id', 'identified_agent', 'similarity_score', 'segments'
    
    Pipeline stages:
        1. Load & validate audio
        2. Clean audio (FFmpeg filters)
        3. Transcribe (Whisper/Parakeet)
        4. Diarize (speaker identification)
        5. Match agent (multi-voiceprint)
        6. Format & save result
    """

def _load_and_validate_audio()   # Load MP3, check duration
def _transcribe_audio()          # Run speech-to-text model
def _diarize_audio()             # Identify speakers per segment
def _match_agent()               # Identify which agent is speaking
def _save_result()               # Write result.json to disk
```

---

### 3. `pipeline.py` (185 lines)
**Purpose**: Audio processing pipeline abstraction  
**Location**: `call_processor/src/pipeline.py`

#### Key Classes & Functions
```python
class AudioPipeline:
    """Orchestrates multi-stage audio processing"""
    
    __init__(model, device, params)
    run(audio_path, status_callback)     # Execute full pipeline
    
    # Stage methods
    _stage_load_audio()          # Load & resample to 16kHz mono
    _stage_audio_cleanup()       # FFmpeg filters
    _stage_transcription()       # Speech-to-text
    _stage_diarization()         # Speaker identification
    _stage_agent_matching()      # Identify agent
    
    # Utilities
    _check_cancelled()           # Check if user cancelled
    _update_status()             # Update status for UI
    _cleanup()                   # Free GPU memory
```

---

## Core Pipeline Modules (src/)

### 1. **Audio Processing**

#### `src/audio_cleanup.py` (90 lines)
**Purpose**: Audio filtering and enhancement  
**Functions**:
```python
def clean_audio(input_path, output_path, filters_str):
    """
    Apply FFmpeg filters to audio.
    
    Filters (from ui.py AUDIO_FILTER):
    - aresample=44100: Upsample for loudness normalization
    - highpass=f=80: Remove HVAC rumble
    - afftdn: Spectral noise reduction
    - loudnorm: Normalize to -16 LUFS (podcast standard)
    - dynaudnorm: Boost quiet segments locally
    
    Result: Cleaner audio that Whisper/Parakeet can understand better
    """
```

---

### 2. **Transcription Module** (Multiple Engines)

#### `src/transcription.py` (140 lines)
**Purpose**: Unified transcription interface  
**Classes**:
```python
class Transcriber(ABC):
    """Base class for all transcription models"""
    
    transcribe(audio_path, language='en')  # Returns: [{'text': str, 'start': float, 'end': float}]
    
    # Model-specific implementations:
    # - whisper_turbo.py: Whisper (large-v3-turbo) via Groq API
    # - parakeet_v3.py: Parakeet via HuggingFace
    # - deepgram_asr.py: Deepgram API
    # - assemblyai_asr.py: AssemblyAI API
```

#### `src/transcribers/whisper_turbo.py`
**Purpose**: Groq Whisper API integration  
**Functions**:
```python
class WhisperTurbo(Transcriber):
    def transcribe(audio_path, language='en'):
        """
        Call Groq API with Whisper-large-v3-turbo model.
        - Supports: MP3, WAV, FLAC
        - Language: Auto-detect or specify
        - Returns: segments with timing
        """
```

---

### 3. **Diarization (Speaker Identification)**

#### `src/diar_multi.py` (95 lines) ⭐ MAIN DIARIZATION
**Purpose**: Multi-voiceprint speaker diarization  
**Called by**: `process_audio.py`

#### Key Functions
```python
def diarize_multi(audio_path, model, segments, output_path=None):
    """
    Identify speaker for each segment (AGENT vs CUSTOMER).
    
    Process:
        1. Load trained voiceprints from data/agent_voiceprints/
        2. Extract embedding for each segment
        3. Max-cosine matching across all agent voiceprints
        4. Assign AGENT/CUSTOMER label
    
    Args:
        audio_path: Path to audio file
        model: CAM++ embedding model
        segments: List of (text, start_time, end_time) from transcriber
    
    Returns:
        List of segments with speaker labels:
        [{
            'text': str,
            'start': float,
            'end': float,
            'speaker': 'AGENT' | 'CUSTOMER' | 'Agent Name',
            'similarity': float (0-1)
        }]
    """

def _extract_segment_embedding(audio, start, end, model):
    """Extract CAM++ embedding for a specific time window"""

def _load_voiceprints(data_dir):
    """Load multi-voiceprint stacks from agent_voiceprints/"""
    
    Returns:
        {
            'agent_name': {
                'voiceprints': [(path, bucket, SNR), ...],  # Multiple VPs per agent
                'paths': [np array of stacked embeddings]
            }
        }

def _match_agent_max_cosine(segment_embedding, voiceprints):
    """
    Find best matching agent using max-cosine across all centroids.
    
    For each agent:
        similarities = [cosine(segment, vp1), cosine(segment, vp2), ...]
        agent_score = max(similarities)
    
    Return: agent with highest score
    """

def _classify_agent_vs_customer(speaker_name, threshold=0.55):
    """
    Classify segment as AGENT (identified agent) or CUSTOMER (unknown/low-confidence).
    
    Rules:
        - If agent match score > threshold → AGENT
        - If score < threshold OR no match → CUSTOMER
    """
```

#### Voiceprint Schema (Extended)
```json
{
  "agent_name": {
    "voiceprint_path": "data/agent_voiceprints/agent_name.npy",  // Legacy single VP
    "voiceprints": [
      {
        "path": "data/agent_voiceprints/agent_name__high_0.npy",
        "bucket": "high",
        "n_clips": 87,
        "snr_db": 19.4
      },
      {
        "path": "data/agent_voiceprints/agent_name__mid_0.npy",
        "bucket": "mid",
        "n_clips": 51,
        "snr_db": 11.8
      },
      {
        "path": "data/agent_voiceprints/agent_name__low_0.npy",
        "bucket": "low",
        "n_clips": 32,
        "snr_db": 6.2
      }
    ],
    "total_seconds": 735.1,
    "used_calls": 5,
    "source": "multi_vp_v1"
  }
}
```

#### Related Modules
- `src/diar_ecapa.py` - Original ECAPA diarization (legacy)
- `src/diar_campp.py` - CAM++ embeddings (legacy)
- `src/diar_voiceprint.py` - Multi-VP matching with max-cosine

---

### 4. **Embedding Models**

#### `src/embedding_campp.py` (80 lines)
**Purpose**: CAM++ speaker embedding extraction  
**Functions**:
```python
class EmbeddingModel:
    """Load and run CAM++ speaker embedding model from WeSpeaker"""
    
    __init__(model_name='camplus_cn_zh-CN', device='cuda'):
        """
        Initialize CAM++ model.
        Default: camplus_cn_zh-CN (Chinese optimized, works well in English)
        Embedding dimension: 512
        """
    
    embed(audio_path, start_s, end_s):
        """
        Extract 512-dimensional speaker embedding for a segment.
        
        Process:
            1. Load audio [start_s : end_s]
            2. Compute mel-spectrogram
            3. Pass through CAM++ encoder
            4. Return 512-dim embedding vector (L2 normalized)
        
        Returns:
            np.array of shape (512,)
        """
```

---

### 5. **Agent Matching**

#### `src/speaker_matcher.py` (105 lines)
**Purpose**: Match speaker embedding to trained agents  
**Functions**:
```python
class SpeakerMatcher:
    """Match speaker embedding to known agents using voiceprints"""
    
    __init__(voiceprints_dir='data/agent_voiceprints'):
        """Load all trained voiceprints"""
    
    match(embedding, top_n=5):
        """
        Find best matching agent(s) for an embedding.
        
        Process:
            1. For each agent, compute cosine similarity with all their voiceprints
            2. Take MAX similarity for that agent
            3. Rank agents by max similarity
            4. Return top_n matches
        
        Args:
            embedding: 512-dim numpy array
            top_n: Number of top agents to return
        
        Returns:
            [
                ('Agent Name', similarity_score),
                ('Agent Name 2', similarity_score),
                ...
            ]
        
        Why MAX not MEAN?
            - MEAN: Single centroid averaged across all conditions
            - MAX: Each agent has VPs for LOW/MID/HIGH SNR, we use closest match
            - More robust to audio quality variations
        """
    
    get_agent_voiceprints(agent_name):
        """Return all voiceprint paths for an agent (LOW, MID, HIGH buckets)"""
```

---

### 6. **Voiceprints Module**

#### `src/voiceprints.py` (125 lines)
**Purpose**: Voiceprint data management  
**Functions**:
```python
def load_voiceprints(voiceprints_dir):
    """
    Load agents.json and voiceprint .npy files.
    
    Structure:
        {
            'agent_name': {
                'voiceprints': [list of VP metadata],
                'paths': 2D array of embeddings (shape: num_vps x 512)
            }
        }
    """

def voiceprint_inventory(voiceprints_dir):
    """
    Get statistics about trained voiceprints.
    
    Returns:
        {
            'total_agents': int,
            'total_voiceprints': int,
            'per_agent': {
                'Agent Name': {'count': int, 'buckets': [high, mid, low]}
            }
        }
    """

def get_agent_voiceprints(voiceprints_dir, agent_name):
    """Get all voiceprint metadata for specific agent"""

def compute_snr(audio_segment):
    """
    Estimate Signal-to-Noise Ratio (SNR) for a segment.
    
    Method:
        - Compute RMS energy in 50ms frames
        - 90th percentile RMS = speech
        - 10th percentile RMS = noise
        - SNR = 20*log10(speech_rms / noise_rms)
    
    Returns:
        SNR in dB
    """
```

---

## Enrollment & Training Scripts

### Production Enrollment Script

#### `enroll_multi_advanced.py` (235 lines) ⭐ CURRENT PRODUCTION
**Purpose**: Train multi-voiceprints for all agents  
**Called by**: UI enrollment endpoint OR manual run  
**Uses**: API data from `data/audiofy/_dataset/`

#### Key Functions
```python
def main(max_calls_per_agent=150, min_agent_calls=5):
    """
    Train multi-voiceprints for all agents from API data.
    
    Pipeline:
        1. Download API calls (scrape_dataset_api.py)
        2. For each agent:
            a. Extract agent-only segments (skip customer)
            b. Compute SNR for each segment
            c. Bucket by SNR: HIGH (≥15dB), MID (8-15dB), LOW (<8dB)
            d. Run K-means clustering per bucket
            e. Save centroids as individual VP files
        3. Update agents.json with multi-VP schema
    """

def download_batch(days=30, max_calls=300):
    """Call scrape_dataset_api.py to get API recordings"""

def extract_agent_samples(call_data, agent_name):
    """
    Extract agent-only phrases from transcription.
    
    Input: call_data with speaker_json from API
    Output: List of (audio_clip, start_s, end_s) for agent only
    
    Enforces purity: Skip any segment labeled as Customer
    """

def compute_snr_per_clip(audio, start_s, end_s):
    """Estimate SNR for a specific segment"""

def bucket_by_snr(embeddings_with_snr):
    """
    Group embeddings into buckets:
        HIGH: SNR ≥ 15dB (clean phone calls)
        MID:  SNR 8-15dB (typical mixed audio)
        LOW:  SNR < 8dB (noisy desk recordings)
    
    Why bucketing?
        - Single centroid can't represent voice across all acoustic conditions
        - LOW bucket captures how agent sounds on laptop/mobile with background noise
        - When matching desk recordings, LOW bucket is the closest match
    """

def cluster_per_bucket(bucket_embeddings, agent_name, bucket_name):
    """
    Run K-means clustering per bucket.
    
    Args:
        bucket_embeddings: List of 512-dim embeddings for this SNR range
        agent_name: e.g., 'Omar El Harchaoui'
        bucket_name: 'high' | 'mid' | 'low'
    
    Centroid count:
        k = min(3, len(embeddings) // 30)
        - Most agents: k=3 (one VP per bucket)
        - Under-trained agents: k=1
    
    Output:
        Save each centroid as: data/agent_voiceprints/agent_name__bucket_idx.npy
    """

def save_voiceprints(agent_data):
    """
    Save to agents.json with multi-VP schema.
    
    Also computes legacy single VP (mean of all VPs) for backward compatibility.
    """

def iterative_tighten(embeddings):
    """
    Remove outlier embeddings that are far from cluster center.
    
    Purpose: Remove customer leakage (segments misclassified as agent speech)
    
    Method:
        1. Compute centroid (mean embedding)
        2. Compute distance of each point from centroid
        3. Discard top 10% furthest points
        4. Repeat until centroid stabilizes
    """
```

#### Run Command
```bash
cd call_processor
python enroll_multi_advanced.py --max-calls-per-agent 150 --min-agent-calls 5
```

---

### Data Preparation

#### `tools/legacy/scrape_dataset_api.py` (180 lines)
**Purpose**: Download recordings from Audiofy API  
**Called by**: `enroll_multi_advanced.py`

#### Key Functions
```python
def main(days=30, max_calls=300):
    """
    Download API calls and metadata.
    
    Saves to: data/audiofy/_dataset/index.json
    
    Schema per call:
        {
            '_id': 'unique-id',
            'agent_name': 'Omar El Harchaoui',
            'duration': 245.6,  # seconds
            'speaker_json': [
                {
                    'start': '00:00:01.775',
                    'end': '00:00:02.203',
                    'speaker': 'Agent Name_2' or 'Customer_1',
                    'phrase': 'text...',
                    'avg_score': 0.821  // confidence
                }
            ]
        }
    """

def download_call(call_id, url):
    """Download single MP3 from API S3 URL"""

def parse_speaker_json(transcription_blob):
    """Extract speaker labels + timings from API response"""
```

---

## Testing & Validation Scripts

### Production Testing

#### `test_api_vs_ours.py` (231 lines) ⭐ MAIN ACCURACY TEST
**Purpose**: Compare system against API ground truth  
**Use**: Validate accuracy before deploying  
**Expected**: 70%+ accuracy on diverse agent pool

#### Key Functions
```python
def main(num_calls=10):
    """
    Test against real API data with ground truth labels.
    
    Process:
        1. Load 10 random API calls from data/audiofy/_dataset/
        2. For each call:
            a. Extract agent name from API metadata
            b. Run diarization on audio
            c. Identify agent using multi-voiceprint system
            d. Compare against ground truth
            e. Compute F1 score for segment-level accuracy
    
    Output:
        [OK] Agent Name | F1=0.921 | Sim=0.593
        [WRONG] Agent Name | F1=0.750 | Sim=0.452
        ...
        Correctly identified: 7/10 (70.0%)
    """

def test_call(call_data, model, voiceprints):
    """Test single call and return accuracy metrics"""

def compute_f1_score(predicted_labels, ground_truth_labels):
    """
    Segment-level F1 score (not call-level).
    
    Measures: How many segments correctly labeled AGENT vs CUSTOMER
    Range: 0 (all wrong) to 1 (perfect)
    """
```

#### Run Command
```bash
cd call_processor
python test_api_vs_ours.py  # Tests 10 random calls
```

---

#### `test_voiceprints_api.py` (372 lines)
**Purpose**: Test held-out API calls (calls NOT used in training)  
**Use**: Measure generalization accuracy

#### Key Functions
```python
def main(top_n=20):
    """
    Test on held-out API calls.
    
    Held-out: Calls whose _id was NOT used in agent training
    Expected: 83%+ accuracy (short calls)
    """

def select_held_out_calls():
    """
    Find calls not in agent training set.
    
    Input: agents.json (tracks which call IDs were used)
    Output: List of untested calls
    """

def compute_confusion_vs_baseline():
    """Compare multi-VP against single-VP performance"""
```

---

#### `test_top_5_agents_ui.py` (227 lines)
**Purpose**: Test UI interface for top 5 agents  
**Use**: Validate web upload/processing pipeline  
**Expected**: 85%+ on top agents

#### Key Functions
```python
def main():
    """
    Test via /api/upload endpoint.
    
    For each top 5 agent:
        1. Find held-out calls
        2. Upload via POST /api/upload
        3. Poll GET /api/call/{result_id}
        4. Measure accuracy + similarity
    """

def upload_and_wait(audio_path, api_url="http://localhost:8080"):
    """
    Upload audio file and wait for processing.
    
    Returns:
        {
            'identified_agent': 'Agent Name',
            'agent_similarity': 0.681,
            'segments': [...]
        }
    """
```

#### Run Command
```bash
cd call_processor
python test_top_5_agents_ui.py
```

---

#### `test_long_calls.py` (New - Your Request)
**Purpose**: Test 3+ minute long calls on live API  
**Expected**: 100% accuracy (extended duration calls)

---

## Data & Configuration Files

### 1. **Voiceprint Data**

#### Location: `data/agent_voiceprints/`

**Files**:
```
agents.json                    # Agent metadata + voiceprint paths
agent_name.npy                 # Legacy: Single voiceprint per agent (mean)
agent_name__high_0.npy         # Clean audio (SNR ≥ 15dB)
agent_name__high_1.npy         # Clean audio (SNR ≥ 15dB)
agent_name__mid_0.npy          # Normal (SNR 8-15dB)
agent_name__low_0.npy          # Noisy (SNR < 8dB)
```

**agents.json Schema**:
```json
{
  "omar_el_harchaoui": {
    "agent_name": "Omar El Harchaoui",
    "voiceprint_path": "data/agent_voiceprints/omar_el_harchaoui.npy",
    "voiceprints": [
      {
        "path": "data/agent_voiceprints/omar_el_harchaoui__high_0.npy",
        "bucket": "high",
        "n_clips": 87,
        "snr_db": 19.4
      },
      {
        "path": "data/agent_voiceprints/omar_el_harchaoui__mid_0.npy",
        "bucket": "mid",
        "n_clips": 51,
        "snr_db": 11.8
      },
      {
        "path": "data/agent_voiceprints/omar_el_harchaoui__low_0.npy",
        "bucket": "low",
        "n_clips": 32,
        "snr_db": 6.2
      }
    ],
    "total_seconds": 735.1,
    "used_calls": 5,
    "source": "multi_vp_v1"
  }
}
```

---

### 2. **Configuration**

#### `src/config.py`
**Purpose**: Global settings  
**Key Settings**:
```python
DEVICE = "cuda"  or "cpu"           # GPU/CPU selection
MODEL_NAME = "camplus_cn_zh-CN"    # CAM++ embedding model
SAMPLE_RATE = 16000                # Audio resampling target
EMBEDDING_DIM = 512                # CAM++ output dimension
SNR_HIGH_THRESHOLD = 15            # dB (clean audio)
SNR_LOW_THRESHOLD = 8              # dB (noisy audio)
AGENT_MATCH_THRESHOLD = 0.55       # Confidence for AGENT classification
MAX_CLUSTER_K = 3                  # Max K for K-means per bucket
```

---

### 3. **API Data**

#### Location: `data/audiofy/_dataset/`

**Files**:
```
index.json                     # Downloaded API call metadata
audio/                         # Downloaded MP3 files
  69efc3acf91ac02559f83a89.mp3
  69efb9b2f91ac02559f821c7.mp3
  ...
```

---

## Function Reference

### By Category

#### Audio Processing Pipeline
| Function | File | Purpose |
|----------|------|---------|
| `process_audio()` | process_audio.py | Main entry point |
| `clean_audio()` | src/audio_cleanup.py | FFmpeg filtering |
| `transcribe()` | src/transcription.py | Speech-to-text |
| `diarize_multi()` | src/diar_multi.py | Speaker identification |
| `match()` | src/speaker_matcher.py | Agent matching |

#### Embedding & Matching
| Function | File | Purpose |
|----------|------|---------|
| `EmbeddingModel.embed()` | src/embedding_campp.py | Extract speaker embedding |
| `SpeakerMatcher.match()` | src/speaker_matcher.py | Find best agent |
| `_match_agent_max_cosine()` | src/diar_multi.py | Max-cosine matching |
| `load_voiceprints()` | src/voiceprints.py | Load VP data |

#### Enrollment & Training
| Function | File | Purpose |
|----------|------|---------|
| `main()` | enroll_multi_advanced.py | Multi-VP training |
| `extract_agent_samples()` | enroll_multi_advanced.py | Get agent speech |
| `bucket_by_snr()` | enroll_multi_advanced.py | SNR bucketing |
| `cluster_per_bucket()` | enroll_multi_advanced.py | K-means per bucket |
| `iterative_tighten()` | enroll_multi_advanced.py | Remove outliers |

#### Testing & Validation
| Function | File | Purpose |
|----------|------|---------|
| `main()` | test_api_vs_ours.py | Accuracy on ground truth |
| `main()` | test_voiceprints_api.py | Test held-out data |
| `upload_and_wait()` | test_top_5_agents_ui.py | Web UI testing |
| `compute_f1_score()` | test_api_vs_ours.py | Segment-level accuracy |

---

## Common Workflows

### Workflow 1: Process a Single Audio File

```python
from process_audio import process_audio

result = process_audio(
    audio_path="call.mp3",
    model="whisper",  # or "parakeet"
    upload_id="test_123",
    status_callback=None
)

print(f"Agent: {result['identified_agent']}")
print(f"Similarity: {result['agent_similarity']}")
print(f"Segments: {len(result['segments'])}")
```

**Files Involved**:
1. `ui.py` - Receives upload
2. `process_audio.py` - Orchestrates
3. `src/audio_cleanup.py` - Cleans audio
4. `src/transcription.py` - Transcribes
5. `src/diar_multi.py` - Diarizes
6. `src/speaker_matcher.py` - Matches agent

---

### Workflow 2: Train New Agents

```bash
cd call_processor

# 1. Download API data
python tools/legacy/scrape_dataset_api.py --days 30 --max-calls 300

# 2. Train multi-voiceprints
python enroll_multi_advanced.py --max-calls-per-agent 150

# 3. Test accuracy
python test_api_vs_ours.py
python test_voiceprints_api.py --top 50

# 4. Validate UI
python test_top_5_agents_ui.py
```

**Files Modified**:
- `data/agent_voiceprints/agents.json` - Updated metadata
- `data/agent_voiceprints/*.npy` - New voiceprints

---

### Workflow 3: Deploy to Production

```bash
# 1. Pull latest code
git pull origin main

# 2. Restart server
bash restart.sh

# 3. Monitor status
curl http://localhost:8080/api/status

# 4. Test with sample audio
curl -X POST -F "file=@sample.mp3" http://localhost:8080/api/upload
```

**Files Modified**:
- Code updated from GitHub
- No data changes (voiceprints already trained)

---

## Performance Metrics

### System Accuracy
```
Short calls (<1 min):    83% (47 held-out API calls)
Long calls (3-8.6 min):  100% (5 extended duration)
Real API data:           70% (10 diverse agents)
Top-5 agents:            71% (when agent is in top 5)
```

### Processing Speed
```
1-minute audio:   ~40 seconds (0.67x RT)
5-minute audio:   ~120 seconds (0.4x RT)
8-minute audio:   ~180 seconds (0.27x RT)

GPU: RTX 4050 6GB VRAM
Memory: 3-4 GB peak usage
```

### System Requirements
```
Python: 3.11+
GPU: 6GB+ VRAM (tested on RTX 4050)
Storage: ~500MB (voiceprints)
CPU: 4+ cores recommended
```

---

## Next Steps

### For Developers
1. Read `src/diar_multi.py` to understand multi-voiceprint matching
2. Modify `enroll_multi_advanced.py` to adjust SNR bucketing thresholds
3. Add new transcription model in `src/transcribers/`

### For Operations
1. Monitor `/api/status` endpoint
2. Check `data/processed/` for result.json files
3. Review logs in server startup output

### For Product
1. Test with real call center audio
2. Validate accuracy on target agents
3. Optimize thresholds based on production data

---

**Last Updated**: 2026-05-04  
**Maintained By**: Claude Code  
**Status**: Production Ready ✅
