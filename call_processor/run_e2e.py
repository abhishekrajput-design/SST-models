"""
End-to-End Pipeline Runner

Runs the complete flow:
  1. Extract clean agent voice clips from desk recordings (diarization)
  2. Enroll agent voice embeddings (SpeechBrain ECAPA-TDNN)
  3. Process a test call recording (diarize + identify + transcribe)

Usage:
    python run_e2e.py --hf-token YOUR_TOKEN
    python run_e2e.py --skip-extraction  (if clips already extracted)
    python run_e2e.py --skip-enrollment  (if embeddings already exist)
"""

import os
import sys
import gc
import time
import argparse
import logging
from pathlib import Path

# Fix Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def clear_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def step_extract(hf_token: str, device: str, agents_dir: str, clean_dir: str,
                 max_recordings: int = 2, max_clips: int = 5):
    """Step 1: Extract clean voice clips from desk recordings."""
    print("\n" + "=" * 60)
    print("  STEP 1: Extract Agent Voice Clips")
    print("=" * 60)

    import torch
    from pyannote.audio import Pipeline
    from extract_agent_voice import process_agent

    import huggingface_hub
    huggingface_hub.login(token=hf_token)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading pyannote diarization on {dev}...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )
    pipeline.to(dev)

    agents_path = Path(agents_dir)
    agent_dirs = sorted([
        d for d in agents_path.iterdir()
        if d.is_dir() and d.name.startswith("agent_")
    ])

    total = 0
    for agent_dir in agent_dirs:
        clips = process_agent(
            str(agent_dir), clean_dir, pipeline,
            min_duration=3.0, max_duration=15.0,
            max_clips_per_recording=max_clips,
            max_recordings=max_recordings,
        )
        total += (clips or 0)

    del pipeline
    clear_gpu()
    print(f"\n  >> Extracted {total} clean clips for {len(agent_dirs)} agents")
    return total


def step_enroll(device: str, clean_dir: str, embeddings_path: str):
    """Step 2: Create agent voice embeddings."""
    print("\n" + "=" * 60)
    print("  STEP 2: Enroll Agent Embeddings")
    print("=" * 60)

    os.makedirs(clean_dir, exist_ok=True)

    # Check if we actually have clips
    from pathlib import Path
    clip_dirs = [d for d in Path(clean_dir).iterdir() if d.is_dir()] if Path(clean_dir).exists() else []
    has_clips = any(
        any(f.suffix in ('.wav', '.mp3', '.flac') for f in d.iterdir())
        for d in clip_dirs
    ) if clip_dirs else False

    if not has_clips:
        print("\n  [WARN] No clean clips found. Falling back to raw agent_samples for enrollment.")
        clean_dir = "data/agent_samples"

    from src.embedding import EmbeddingExtractor

    extractor = EmbeddingExtractor(
        device=device,
        model_dir="models/spkrec-ecapa",
    )

    try:
        agent_embeddings = extractor.enroll_all_agents(
            agents_dir=clean_dir,
            output_path=embeddings_path,
        )
    finally:
        extractor.unload_model()

    clear_gpu()
    print(f"\n  >> Enrolled {len(agent_embeddings)} agents: {', '.join(agent_embeddings.keys())}")
    return agent_embeddings


def step_test_call(hf_token: str, device: str, test_audio: str,
                   embeddings_path: str, whisper_model: str = "base",
                   language: str = "en"):
    """Step 3: Process a test call through the full pipeline."""
    print("\n" + "=" * 60)
    print("  STEP 3: Process Test Call")
    print("=" * 60)

    from src.pipeline import CallProcessor

    processor = CallProcessor(
        hf_token=hf_token,
        embeddings_path=embeddings_path,
        model_cache_dir="models",
        whisper_model=whisper_model,
        whisper_compute_type="int8" if device == "cuda" else "float32",
        language=language,
        device=device,
        similarity_threshold=0.25,
        min_segment_duration=1.0,
        output_dir="data/processed",
    )

    result = processor.process(test_audio)
    clear_gpu()

    # Print transcript
    print(f"\n  >> Processed {result['total_segments']} segments in {result['processing_time_seconds']}s")
    print(f"\n{'='*60}")
    print("TRANSCRIPT")
    print(f"{'='*60}")
    for seg in result["segments"]:
        speaker = seg.get("identified_speaker", seg.get("speaker", "Unknown"))
        text = seg.get("text", "")
        conf = seg.get("confidence", 0)
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker} ({conf:.2f}): {text}")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Pipeline: Extract + Enroll + Process"
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--agents-dir", default="data/agent_samples",
                        help="Raw agent recordings directory")
    parser.add_argument("--clean-dir", default="data/agent_clean_clips",
                        help="Output for extracted clean clips")
    parser.add_argument("--embeddings", default="embeddings/agent_embeddings.pkl")
    parser.add_argument("--test-audio", default=None,
                        help="Audio file to test (if not provided, uses first agent recording)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
                        help="Whisper model (use 'base' for fast testing, 'large-v3' for production)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip step 1 if clips already extracted")
    parser.add_argument("--skip-enrollment", action="store_true",
                        help="Skip step 2 if embeddings already exist")
    parser.add_argument("--max-recordings", type=int, default=2,
                        help="Max recordings per agent for extraction")

    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: Set HF_TOKEN or pass --hf-token")
        sys.exit(1)

    start = time.time()
    print("=" * 60)
    print("  MULTI-SPEAKER CALL PROCESSOR - E2E PIPELINE")
    print("=" * 60)

    # Step 1: Extract
    if not args.skip_extraction:
        step_extract(
            hf_token, args.device, args.agents_dir, args.clean_dir,
            max_recordings=args.max_recordings,
        )
    else:
        print("\n  >> Skipping extraction (--skip-extraction)")

    # Step 2: Enroll
    if not args.skip_enrollment:
        step_enroll(args.device, args.clean_dir, args.embeddings)
    else:
        print("\n  >> Skipping enrollment (--skip-enrollment)")

    # Step 3: Test call
    test_audio = args.test_audio
    if not test_audio:
        # Use first recording from first agent as test
        agents_path = Path(args.agents_dir)
        for agent_dir in sorted(agents_path.iterdir()):
            if agent_dir.is_dir() and agent_dir.name.startswith("agent_"):
                files = sorted([
                    f for f in agent_dir.iterdir()
                    if f.suffix.lower() in ('.mp3', '.wav', '.flac')
                ])
                if files:
                    test_audio = str(files[0])
                    break

    if test_audio and os.path.exists(test_audio):
        step_test_call(
            hf_token, args.device, test_audio, args.embeddings,
            whisper_model=args.whisper_model, language=args.language,
        )
    else:
        print("\n  >> No test audio found. Place a call recording in data/raw_calls/")
        print("     Or pass --test-audio path/to/file.wav")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE -- Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
