# Development Summary & Roadmap

## What We Have Done So Far

### 1. Built the Core Architecture
We designed a production-ready, modular pipeline for multi-speaker call processing that operates within a strict 4GB VRAM constraint constraints:
- **`src/diarization.py`**: Uses Pyannote Audio 3.1 to detect "who spoke when" in the recording. 
- **`src/embedding.py`**: Uses SpeechBrain ECAPA-TDNN to turn extracted voices into 192-dimensional "voiceprints" (embeddings).
- **`src/speaker_matcher.py`**: Compares voiceprints against known agents using Cosine Similarity to identify the speaker (or label as "Customer").
- **`src/transcription.py`**: Uses Faster-Whisper (Int8 quantized) to transcribe the individual segments.
- **`src/pipeline.py` & `main.py`**: Orchestrates sequential model loading (load model → process → clear VRAM → next model) to prevent Out of Memory errors.

### 2. Built the Agent Voice Extractor (`extract_agent_voice.py`)
We analyzed the S3 files downloaded to `data/agent_samples/` and realized they are **30-minute desk recordings** with multiple people talking. Since we can't train the agent's voice on a multi-speaker file, we built an automated extractor:
- It processes the long recording.
- Uses diarization to figure out who is speaking the most (which is logically the agent at the desk).
- Slices out multiple clean 3–15 second clips of *just the agent* and saves them to `data/agent_clean_clips/`.

### 3. Built the Unified End-to-End Runner (`run_e2e.py`)
A single script that sequentially executes the entire workflow:
1. Exacts the clean clips from the 30 min recordings.
2. Enrolls the clean clips to create known agent embeddings.
3. Takes one full test recording, diarizes it, identifies the agent throughout the call, and transcribes the conversation. 

### 4. Windows/PyTorch Troubleshooting
- Downgraded to `numpy<2` to fix `onnxruntime` conflicts.
- Removed deprecated `use_auth_token` Pyannote arguments and migrated to native `huggingface_hub.login()`.
- Bypassed the failing `libtorchcodec_core4.dll` PyTorch dependency gracefully on Windows by overriding audio loading to use `soundfile` (`sf.read`) in `diarization.py`.
- Fixed cp1252 Windows unicode limitations.

---

## What We Are Working On Now

### 1. Finalizing testing for the E2E Workflow
We are evaluating the test run powered by `run_e2e.py`. We needed to bypass the HuggingFace gated model restrictions, which is now resolved. Now we are verifying the pipeline connects seamlessly by taking the raw S3 agent data and outputting a perfectly aligned, speaker-named transcript test sample.

### 2. Tuning and Refinement
Once the test pipeline finishes successfully, we will move to tuning the AI stack:
- **Similarity Threshold**: We need to see if `0.25` is too strict or too loose for accepting an embedding as the Agent instead of "Customer".
- **Whisper Optimizations**: We switched the default config to `whisper_model: medium` and `device: cpu` for safety/compatibility. We will push this to `GPU / Int8 / Large-v3` if the VRAM holds up to get state-of-the-art transcriptions.

### 3. Processing the Full Dataset
Upon successful tuning, we will fire the pipeline against the rest of the raw call data to deliver clean `.json` and `.txt` transcriptions of every call.
