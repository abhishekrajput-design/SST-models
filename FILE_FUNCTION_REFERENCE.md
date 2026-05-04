# SST-Models: Complete File & Function Reference

**Quick lookup for which file does what**

---

## 🔴 CRITICAL PRODUCTION FILES

### `ui.py` - Web Server & API
- **What**: Flask HTTP server with REST API
- **Port**: 8080
- **Endpoints**:
  - `POST /api/upload` → Upload audio file
  - `GET /api/call/{id}` → Get result
  - `GET /api/calls` → List processed
  - `GET /api/status` → Pipeline status
  - `POST /api/cancel` → Cancel processing
- **Key Classes**:
  - `CallHandler` - HTTP request handler
  - `PipelineCancelled` - Cancellation exception
- **Key Functions**:
  - `_process_upload()` - Main pipeline (threaded)
  - `_update_status()` - Update status
  - `_start_server()` - Start HTTP server
- **Status Updates**: Shared via `_status` dict with threading lock
- **When to modify**: Add new API endpoints, change port, adjust AUDIO_FILTER

---

### `process_audio.py` - Pipeline Orchestrator
- **What**: Main audio processing entry point
- **Called by**: `ui.py` (in separate thread)
- **Pipeline Stages**:
  1. Load audio (ffmpeg)
  2. Clean audio (filters)
  3. Transcribe (Whisper/Parakeet)
  4. Diarize (speaker ID)
  5. Match agent (multi-VP)
  6. Save result
- **Key Functions**:
  - `process_audio()` - Main pipeline
  - `_load_and_validate_audio()` - Load MP3/WAV
  - `_transcribe_audio()` - Speech-to-text
  - `_diarize_audio()` - Speaker identification
  - `_match_agent()` - Identify agent
  - `_save_result()` - Write result.json
- **When to modify**: Change pipeline order, add/remove stages, adjust timeouts

---

### `src/pipeline.py` - Pipeline Abstraction
- **What**: Class-based pipeline orchestrator
- **Alternative to**: Direct function calls in process_audio.py
- **Key Class**: `AudioPipeline`
  - `__init__()` - Initialize with model
  - `run()` - Execute full pipeline
  - `_stage_*()` - Individual stage methods
  - `_check_cancelled()` - Check if user cancelled
  - `_update_status()` - Status callback
- **When to modify**: Refactoring, error handling improvements

---

## 🟢 CORE PROCESSING MODULES

### Stage 1: Audio Cleanup

#### `src/audio_cleanup.py` - Audio Filtering
- **What**: Apply FFmpeg filters to audio
- **Filter Chain** (from ui.py AUDIO_FILTER):
  - `aresample=44100` → Upsample to 44.1kHz
  - `highpass=f=80` → Remove HVAC rumble
  - `afftdn=nf=-25:nt=w` → Spectral noise reduction
  - `loudnorm=I=-16:TP=-1.5:LRA=11` → Normalize to podcast standard
  - `dynaudnorm=p=0.9:m=100:s=5:g=15` → Boost quiet segments
- **Key Functions**:
  - `clean_audio(input_path, output_path, filters_str)` - Apply filters
- **When to modify**: Change noise reduction, adjust loudness target

---

### Stage 2: Transcription

#### `src/transcription.py` - Transcriber Interface
- **What**: Abstract base class for all transcription engines
- **Key Class**: `Transcriber` (abstract)
  - `transcribe(audio_path, language)` - Return segments with timing
  - Returns: `[{'text': str, 'start': float, 'end': float}, ...]`
- **Implementations**:
  - `WhisperTurbo` → `src/transcribers/whisper_turbo.py`
  - `ParakeetV3` → `src/transcribers/parakeet_v3.py`
  - `Deepgram` → `src/transcribers/deepgram_asr.py`
  - `AssemblyAI` → `src/transcribers/assemblyai_asr.py`
- **When to modify**: Add new transcriber, change language, adjust timeouts

---

#### `src/transcribers/whisper_turbo.py` - Groq Whisper
- **What**: Groq API integration for Whisper-large-v3-turbo
- **Model**: whisper-large-v3-turbo (Fast, accurate)
- **Inference**: Via Groq API (not local)
- **Key Class**: `WhisperTurbo`
  - `transcribe()` - Call API, parse JSON
- **When to modify**: Change to different model, update API key handling

---

#### `src/transcribers/parakeet_v3.py` - Parakeet TTS
- **What**: Parakeet speech-to-text model
- **Inference**: Local (HuggingFace)
- **Speed**: Faster than Whisper
- **Key Class**: `ParakeetV3`
  - `transcribe()` - Run model inference
- **When to modify**: Change model version, adjust batch size

---

### Stage 3: Diarization (Speaker Identification)

#### `src/diar_multi.py` ⭐ MAIN DIARIZATION
- **What**: Multi-voiceprint speaker identification
- **Algorithm**: Max-cosine matching across SNR-bucketed voiceprints
- **Key Functions**:
  - `diarize_multi(audio_path, model, segments)` - Main function
    - Returns: Segments with speaker labels (AGENT/CUSTOMER)
  - `_extract_segment_embedding()` - Get speaker embedding
  - `_load_voiceprints()` - Load multi-VP stacks
  - `_match_agent_max_cosine()` - Find best match
  - `_classify_agent_vs_customer()` - Classify as AGENT or CUSTOMER
- **Voiceprint Selection**:
  - Per agent: LOW + MID + HIGH bucket voiceprints
  - For each segment: max-cosine across all VPs
  - Result: Agent name or CUSTOMER (if low confidence)
- **SNR Buckets**:
  - HIGH (≥15dB): Clean phone calls
  - MID (8-15dB): Typical mixed audio
  - LOW (<8dB): Noisy desk recordings
- **When to modify**: Change matching algorithm, adjust thresholds, add new bucket

---

#### `src/diar_ecapa.py` - ECAPA Diarization (Legacy)
- **What**: Original diarization using ECAPA embeddings
- **Status**: Legacy, replaced by CAM++ (better performance)
- **Keep for**: Backward compatibility tests
- **When to modify**: Only if needed for legacy code

---

#### `src/diar_campp.py` - CAM++ Diarization (Legacy)
- **What**: CAM++ speaker identification (single VP per agent)
- **Status**: Legacy, replaced by multi-VP in diar_multi.py
- **Key Functions**:
  - `diarize()` - Single-VP matching
- **When to modify**: Only for legacy tests/comparison

---

#### `src/diar_voiceprint.py` - Multi-VP Matching Core
- **What**: Core multi-voiceprint matching logic
- **Called by**: `diar_multi.py`
- **Key Functions**:
  - `_load_voiceprints()` - Load VP stacks
  - `_match_agent_max_cosine()` - Max-cosine similarity
  - `_classify_agent_vs_customer()` - Binary classification
- **When to modify**: Optimize matching algorithm, change similarity metric

---

### Stage 4: Agent Matching

#### `src/speaker_matcher.py` - Agent Matcher
- **What**: Match embedding to trained agents
- **Algorithm**: Max-cosine across all agent voiceprints
- **Key Class**: `SpeakerMatcher`
  - `__init__(voiceprints_dir)` - Load VPs
  - `match(embedding, top_n)` - Find best agents
    - Returns: `[('Agent Name', score), ...]`
  - `get_agent_voiceprints()` - Get VP paths for agent
- **When to modify**: Change ranking logic, add confidence scoring

---

### Embedding Models

#### `src/embedding_campp.py` - CAM++ Embeddings
- **What**: Speaker embedding extraction using CAM++ model
- **Model**: camplus_cn_zh-CN (from WeSpeaker)
- **Output**: 512-dimensional normalized vector
- **Key Class**: `EmbeddingModel`
  - `__init__()` - Load model to GPU
  - `embed(audio_path, start_s, end_s)` - Extract embedding
    - Process: Load audio → Mel-spectrogram → CAM++ encoder → L2 normalize
- **When to modify**: Change model, adjust feature extraction

---

#### `src/embedding.py` - Embedding Interface
- **What**: Abstract embedding interface
- **Status**: Legacy/alternative to embedding_campp
- **Key Class**: `Embedder` (abstract)
- **When to modify**: Add new embedding model

---

### Voiceprints Management

#### `src/voiceprints.py` - VP Data Management
- **What**: Load/manage voiceprint files and metadata
- **Key Functions**:
  - `load_voiceprints(dir)` - Load agents.json + .npy files
  - `voiceprint_inventory(dir)` - Get VP statistics
  - `get_agent_voiceprints(dir, agent)` - Get specific agent VPs
  - `compute_snr(audio)` - Calculate Signal-to-Noise ratio
    - Method: RMS energy per 50ms frame, 90th vs 10th percentile
    - Output: SNR in dB
- **When to modify**: Change SNR calculation, add VP metadata

---

## 🔵 ENROLLMENT & TRAINING

### Production Enrollment

#### `enroll_multi_advanced.py` ⭐ CURRENT PRODUCTION
- **What**: Train multi-voiceprints for all agents from API data
- **Pipeline**:
  1. Download API calls
  2. Extract agent-only speech (skip customer)
  3. Compute SNR per segment
  4. Bucket by SNR: HIGH/MID/LOW
  5. K-means clustering per bucket
  6. Save centroids as individual VP files
  7. Update agents.json
- **Key Functions**:
  - `main(max_calls, min_agent_calls)` - Entry point
  - `download_batch(days, max_calls)` - Get API data
  - `extract_agent_samples(call_data, agent_name)` - Get agent speech
  - `compute_snr_per_clip()` - SNR estimation
  - `bucket_by_snr(embeddings)` - Group by quality
  - `cluster_per_bucket()` - K-means per bucket
  - `save_voiceprints()` - Write agents.json
  - `iterative_tighten(embeddings)` - Remove outliers
- **Inputs**: `data/audiofy/_dataset/index.json` + MP3 files
- **Outputs**: 
  - `data/agent_voiceprints/agent_name.npy` (legacy mean)
  - `data/agent_voiceprints/agent_name__high_*.npy`
  - `data/agent_voiceprints/agent_name__mid_*.npy`
  - `data/agent_voiceprints/agent_name__low_*.npy`
  - `data/agent_voiceprints/agents.json` (updated)
- **Run**: `python enroll_multi_advanced.py --max-calls-per-agent 150`
- **When to modify**: Change SNR thresholds, K-means k value, clustering method

---

#### `enroll_multi_from_api.py` - Multi-VP from API (Original)
- **What**: Original multi-VP enrollment script
- **Status**: Working, but newer `enroll_multi_advanced.py` is preferred
- **Differences**: Slightly different implementation, same result
- **When to use**: Alternative if `enroll_multi_advanced.py` fails

---

#### `enroll_multi_strict_purity.py` - Strict Enrollment
- **What**: Enrollment with stricter customer filtering
- **Status**: Alternative version for quality-sensitive training
- **When to use**: If customer leakage is detected in normal enrollment

---

#### `enroll_multi_final_optimized.py` - Optimized Enrollment
- **What**: Performance-optimized version
- **Status**: Latest optimizations
- **When to use**: For faster re-training with large datasets

---

#### `enroll_all_from_api.py` - Single-VP Enrollment (Legacy)
- **What**: Original enrollment (single VP per agent)
- **Status**: Superseded by multi-VP versions
- **Keep for**: Historical comparison
- **When to modify**: Only for legacy testing

---

#### `enroll_agents.py` - Agent Enrollment (Deprecated)
- **What**: Old enrollment script
- **Status**: Deprecated, use `enroll_multi_advanced.py` instead
- **When to modify**: Never, use newer version instead

---

#### `enroll_from_desk.py` - Desk Recording Enrollment (Experimental)
- **What**: Train from local desk recordings (not API data)
- **Status**: Experimental
- **When to use**: If you have local MP3s to train on

---

### Data Preparation

#### `tools/legacy/scrape_dataset_api.py` - Download API Data
- **What**: Download calls from Audiofy API
- **Output**: `data/audiofy/_dataset/index.json` + MP3s
- **Key Functions**:
  - `main(days, max_calls)` - Download calls
  - `download_call()` - Get single call
  - `parse_speaker_json()` - Extract speaker labels
- **Schema Output**:
  ```json
  {
    "_id": "call_id",
    "agent_name": "Agent Name",
    "duration": 245.6,
    "speaker_json": [{"start": "00:00:01", "speaker": "Agent Name_2", ...}]
  }
  ```
- **When to modify**: Change API endpoint, adjust auth, modify download logic

---

#### `tools/legacy/download_recordings.py` - Batch Download
- **What**: Alternative download script
- **Status**: Legacy variant of scrape_dataset_api.py
- **When to use**: If main downloader fails

---

## 🟡 TESTING & VALIDATION

### Main Accuracy Tests

#### `test_api_vs_ours.py` ⭐ PRIMARY VALIDATION
- **What**: Test against real API data with ground truth
- **Data**: 10 random calls from `data/audiofy/_dataset/`
- **Ground Truth**: API speaker_json labels
- **Output**:
  ```
  [OK] Agent Name | F1=0.921 | Sim=0.593
  [WRONG] Agent Name | F1=0.750 | Sim=0.452
  Correctly identified: 7/10 (70%)
  ```
- **Key Functions**:
  - `main(num_calls)` - Run test
  - `test_call()` - Test single call
  - `compute_f1_score()` - Segment-level accuracy
- **Metrics**:
  - Accuracy: % of calls correctly identified
  - F1: Segment-level AGENT vs CUSTOMER accuracy (0-1)
  - Similarity: Confidence score (0-1)
- **Expected**: 70%+ overall (varies by agent)
- **Run**: `python test_api_vs_ours.py`
- **When to modify**: Change test dataset, add metrics, adjust validation logic

---

#### `test_voiceprints_api.py` - Held-Out Data Test
- **What**: Test on API calls NOT used in training
- **Held-Out**: Calls whose _id not in agents.json
- **Expected**: 83%+ (short calls), better generalization
- **Key Functions**:
  - `main(top_n)` - Run on top N held-out calls
  - `select_held_out_calls()` - Find unused calls
  - `compute_confusion_vs_baseline()` - Multi-VP vs single-VP
- **Output**: Per-agent F1 scores + confusion matrix
- **Run**: `python test_voiceprints_api.py --top 50`
- **When to modify**: Change threshold, add new metrics

---

#### `test_top_5_agents_ui.py` - UI/API Testing
- **What**: Test web upload via `/api/upload`
- **Agents**: Top 5 by accuracy (Kowsar, Haris, Ideal, Janusaan, Omar)
- **Process**: Upload → Poll → Compare
- **Expected**: 85%+ accuracy on top agents
- **Key Functions**:
  - `main()` - Run tests
  - `upload_and_wait()` - Upload and wait for result
- **Output**: Per-agent accuracy + similarity
- **Run**: `python test_top_5_agents_ui.py`
- **When to modify**: Add agents, change test calls, adjust polling

---

#### `test_long_calls.py` - Long Call Testing (New)
- **What**: Test 3+ minute recordings
- **Expected**: 100% accuracy (extended duration helps)
- **Key Functions**:
  - `main()` - Find and test long calls
- **Output**: Per-call accuracy with duration breakdown
- **Run**: `python test_long_calls.py`
- **When to modify**: Change duration threshold, add agents

---

### Other Test Scripts

#### `test_omar_e2e.py` - End-to-End Test (Omar)
- **What**: Test specific agent (Omar El Harchaoui) full pipeline
- **Process**: Load call → Process → Verify output
- **Expected**: Correct agent identification
- **When to use**: Regression testing for specific agent

---

#### `test_diar_direct.py` - Direct Diarization Test
- **What**: Test diarization stage only (skip transcription)
- **When to use**: Debug diarization without transcription overhead

---

#### `test_ui_simple.py`, `test_ui_sequential.py`, etc.
- **What**: Various UI testing approaches
- **Status**: Experimental/development
- **When to use**: Development and debugging

---

#### `compare_ui_vs_api.py` - UI vs API Comparison
- **What**: Compare results from UI vs direct API call
- **Purpose**: Verify UI matches backend
- **When to use**: Debug discrepancies

---

## 📊 ANALYSIS & REPORTING

#### `generate_test_report.py` - Report Generator
- **What**: Generate test result reports
- **Input**: Test result JSON files
- **Output**: Formatted markdown report
- **Key Functions**:
  - `main()` - Generate report
- **When to use**: After running tests, create summary

---

#### `optimize_threshold.py` - Threshold Tuning
- **What**: Find optimal agent matching threshold
- **Process**: Test different thresholds, measure accuracy
- **Output**: Optimal threshold value
- **Key Functions**:
  - `main(min_threshold, max_threshold, step)` - Search thresholds
- **When to use**: Improve accuracy for specific agent pool

---

## 🔧 UTILITIES & HELPERS

#### `src/utils.py` - General Utilities
- **What**: Helper functions used across project
- **Functions**: TBD (check file for specific utilities)
- **Common**: Path handling, file I/O, logging

---

#### `src/config.py` - Global Configuration
- **What**: Global settings and constants
- **Key Settings**:
  - `DEVICE` = "cuda" or "cpu"
  - `SAMPLE_RATE` = 16000
  - `EMBEDDING_DIM` = 512
  - `SNR_HIGH_THRESHOLD` = 15
  - `SNR_LOW_THRESHOLD` = 8
  - `AGENT_MATCH_THRESHOLD` = 0.55
- **When to modify**: Adjust thresholds, change model paths

---

#### `src/target_speaker_vad.py` - VAD (Voice Activity Detection)
- **What**: Detect speech vs silence in audio
- **Purpose**: Trim segments to only speech
- **Key Functions**:
  - `get_vad()` - Load VAD model
  - `detect()` - Find speech boundaries
- **When to modify**: Change VAD model or sensitivity

---

#### `src/conversation_roles.py` - Speaker Role Classification
- **What**: Classify speaker as Agent vs Customer
- **Key Functions**:
  - `classify_role()` - Assign role to speaker
  - `get_role_labels()` - Get standard roles
- **When to modify**: Change role definitions

---

#### `src/speaker_role.py` - Speaker Role Abstraction
- **What**: Speaker role management
- **When to modify**: Extend role system

---

#### `src/transcribers/__init__.py` - Transcriber Loader
- **What**: Load and initialize transcriber models
- **Patch**: Windows SpeechBrain bug fix (importlib path)
- **When to modify**: Add new transcriber registration

---

## 📁 DATA DIRECTORIES

```
data/
├── agent_voiceprints/           # Trained voiceprints
│   ├── agents.json              # Metadata (multi-VP schema)
│   ├── agent_name.npy           # Legacy single VP
│   ├── agent_name__high_0.npy   # Clean audio
│   ├── agent_name__mid_0.npy    # Normal audio
│   └── agent_name__low_0.npy    # Noisy audio
│
├── audiofy/
│   └── _dataset/
│       ├── index.json           # Downloaded API metadata
│       └── audio/               # Downloaded MP3 files
│           ├── call_id.mp3
│           └── ...
│
├── raw_calls/                   # User uploads
│   ├── upload.mp3
│   ├── enhanced_upload.mp3      # After FFmpeg cleanup
│   └── ...
│
└── processed/                   # Results
    ├── result_id/
    │   ├── result.json          # Final result
    │   ├── diarization.json     # Segment details
    │   └── transcription.json   # Text + timing
    └── ...
```

---

## 🚀 Quick Command Reference

### Data Preparation
```bash
cd call_processor

# Download API calls
python tools/legacy/scrape_dataset_api.py --days 30 --max-calls 300

# Train voiceprints
python enroll_multi_advanced.py --max-calls-per-agent 150
```

### Testing
```bash
# Test against ground truth (10 random calls)
python test_api_vs_ours.py

# Test held-out calls (calls not in training)
python test_voiceprints_api.py --top 50

# Test UI interface
python test_top_5_agents_ui.py

# Test 3+ minute long calls
python test_long_calls.py
```

### Running Server
```bash
# Development
python ui.py

# Production (Linux)
bash restart.sh

# Check status
curl http://localhost:8080/api/status
```

---

## 📈 Expected Accuracy

| Test | Accuracy | Count | Notes |
|------|----------|-------|-------|
| Short calls (<1 min) | 83% | 47 calls | Held-out API data |
| Long calls (3-8.6 min) | 100% | 5 calls | Extended context helps |
| Real API (diverse) | 70% | 10 calls | All agents, including undertrained |
| Top-5 agents only | 71% | When agent in top-5 | Best agents only |

---

## 🎯 Next Steps for Users

### To Deploy
1. Read: DEPLOYMENT_GUIDE.md
2. Run: `bash restart.sh`
3. Test: `curl http://localhost:8080/api/status`

### To Train
1. Read: `enroll_multi_advanced.py` documentation
2. Run: `python enroll_multi_advanced.py`
3. Validate: `python test_api_vs_ours.py`

### To Test
1. Read: Test script docstrings
2. Run: `python test_api_vs_ours.py`
3. Analyze: Result JSON in `data/processed/`

---

**Last Updated**: 2026-05-04  
**Maintained By**: Claude Code  
**Version**: 1.0
