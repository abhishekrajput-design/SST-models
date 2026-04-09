"""
Multi-Speaker Call Processor — CLI Entry Point

Usage:
    python main.py --input path/to/audio.wav
    python main.py --input path/to/audio.wav --threshold 0.3 --whisper-model large-v2
    python main.py --input path/to/audio.wav --device cpu

Requirements:
    1. Set HF_TOKEN environment variable (HuggingFace token for pyannote)
    2. Enroll agents first: python enroll_agents.py
"""

import os
import sys
import argparse
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Speaker Call Processor — Diarize, Identify, Transcribe"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the call audio file (WAV, MP3, FLAC, etc.)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--embeddings",
        default="embeddings/agent_embeddings.pkl",
        help="Path to agent embeddings file (default: embeddings/agent_embeddings.pkl)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.25,
        help="Speaker matching similarity threshold (default: 0.25)",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        choices=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Whisper compute type — int8 for 4GB VRAM (default: int8)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for transcription (default: en)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=1.0,
        help="Minimum segment duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Resolve HuggingFace token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HuggingFace token required for pyannote.audio")
        print("Set it via: export HF_TOKEN=your_token")
        print("Or pass: --hf-token your_token")
        print("Get token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Check if agent embeddings exist
    if not os.path.exists(args.embeddings):
        print(f"WARNING: No agent embeddings found at {args.embeddings}")
        print("Run 'python enroll_agents.py' first to enroll agent voices.")
        print("Proceeding without speaker identification...\n")

    # Run pipeline
    from src.pipeline import CallProcessor

    processor = CallProcessor(
        hf_token=hf_token,
        embeddings_path=args.embeddings,
        model_cache_dir="models",
        whisper_model=args.whisper_model,
        whisper_compute_type=args.compute_type,
        language=args.language,
        device=args.device,
        similarity_threshold=args.threshold,
        min_segment_duration=args.min_segment,
        output_dir=args.output,
    )

    result = processor.process(args.input)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Segments: {result['total_segments']}")
    print(f"Time: {result['processing_time_seconds']}s")
    print(f"\nTranscript:")
    print(f"{'-'*60}")
    for seg in result["segments"]:
        speaker = seg.get("identified_speaker", seg.get("speaker", "Unknown"))
        text = seg.get("text", "")
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker}: {text}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
