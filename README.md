# STT-models

Multi-speaker call processing pipeline with audio enhancement and web UI.

## Features
- Speaker diarization (pyannote.audio)
- Speaker identification (SpeechBrain ECAPA-TDNN)
- Transcription (Whisper Large v3 via faster-whisper)
- Audio enhancement (FFmpeg, noisereduce, DeepFilterNet3, SpeechBrain MetricGAN+)
- Web dashboard for viewing and comparing results

## Setup
```bash
pip install -r call_processor/requirements.txt
```

## Usage
```bash
# Start web UI
python call_processor/ui.py

# Run pipeline directly
python call_processor/run_e2e.py --hf-token YOUR_TOKEN --device cuda --test-audio path/to/audio.mp3
```
