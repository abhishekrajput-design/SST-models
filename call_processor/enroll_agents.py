"""
Agent Voice Enrollment Script

Processes audio samples from agent directories and creates
speaker embedding database for voice identification.

Usage:
    python enroll_agents.py
    python enroll_agents.py --agents-dir data/agent_samples --output embeddings/agent_embeddings.pkl

Directory structure expected:
    data/agent_samples/
    ├── agent_abhishek/
    │   ├── sample1.wav
    │   ├── sample2.wav
    │   └── ...
    ├── agent_rahul/
    │   ├── sample1.wav
    │   └── ...
    └── ...

Tips for best results:
    - Use 3-10 clear audio samples per agent (5-30 seconds each)
    - Samples should contain ONLY the agent's voice (no background speakers)
    - Include variety: different phrases, slight volume differences
    - Clean audio without heavy background noise works best
"""

import os
import sys
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Enroll agent voices — create speaker embedding database"
    )
    parser.add_argument(
        "--agents-dir",
        default="data/agent_samples",
        help="Directory containing agent_NAME/ subdirectories with audio samples",
    )
    parser.add_argument(
        "--output",
        default="embeddings/agent_embeddings.pkl",
        help="Output path for embeddings pickle file",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for embedding extraction (default: cuda)",
    )
    parser.add_argument(
        "--model-dir",
        default="models/spkrec-ecapa",
        help="Directory to cache SpeechBrain model",
    )

    args = parser.parse_args()

    # Validate agents directory
    if not os.path.exists(args.agents_dir):
        print(f"ERROR: Agents directory not found: {args.agents_dir}")
        print(f"Create it and add agent voice samples:")
        print(f"  {args.agents_dir}/agent_NAME/sample1.wav")
        sys.exit(1)

    # Check for agent subdirectories with audio files
    agent_dirs = [
        d for d in os.listdir(args.agents_dir)
        if os.path.isdir(os.path.join(args.agents_dir, d))
    ]

    if not agent_dirs:
        print(f"ERROR: No agent directories found in {args.agents_dir}")
        print("Create subdirectories like: agent_abhishek/, agent_rahul/")
        sys.exit(1)

    print(f"Found {len(agent_dirs)} agent directories:")
    for d in sorted(agent_dirs):
        audio_count = len([
            f for f in os.listdir(os.path.join(args.agents_dir, d))
            if f.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
        ])
        print(f"  {d}: {audio_count} audio files")

    if not any(
        any(
            f.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
            for f in os.listdir(os.path.join(args.agents_dir, d))
        )
        for d in agent_dirs
    ):
        print("\nERROR: No audio files found in any agent directory!")
        print("Add .wav, .mp3, or .flac files to agent directories.")
        sys.exit(1)

    # Run enrollment
    from src.embedding import EmbeddingExtractor

    extractor = EmbeddingExtractor(
        device=args.device,
        model_dir=args.model_dir,
    )

    try:
        agent_embeddings = extractor.enroll_all_agents(
            agents_dir=args.agents_dir,
            output_path=args.output,
        )
    finally:
        extractor.unload_model()

    # Summary
    print(f"\n{'='*50}")
    print(f"ENROLLMENT COMPLETE")
    print(f"{'='*50}")
    print(f"Agents enrolled: {len(agent_embeddings)}")
    for name in sorted(agent_embeddings.keys()):
        print(f"  ✓ {name}")
    print(f"\nEmbeddings saved to: {args.output}")
    print(f"\nYou can now run:")
    print(f"  python main.py --input path/to/call.wav")


if __name__ == "__main__":
    main()
