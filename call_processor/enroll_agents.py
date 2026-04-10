"""
Agent voice enrollment script.

Builds a speaker embedding database from audio files under data/agent_samples.

Usage:
    python enroll_agents.py
    python enroll_agents.py --device cpu
"""

import argparse
import logging
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Enroll agent voices and create the speaker embedding database"
    )
    parser.add_argument(
        "--agents-dir",
        default="data/agent_samples",
        help="Directory containing agent_NAME subdirectories with audio samples",
    )
    parser.add_argument(
        "--output",
        default="embeddings/agent_embeddings.pkl",
        help="Output path for the embeddings pickle file",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for embedding extraction (default: cpu)",
    )
    parser.add_argument(
        "--model-dir",
        default="models/spkrec-ecapa",
        help="Directory to cache the SpeechBrain model",
    )
    args = parser.parse_args()

    if not os.path.exists(args.agents_dir):
        print(f"ERROR: Agents directory not found: {args.agents_dir}")
        print(f"Create it and add agent voice samples:")
        print(f"  {args.agents_dir}/agent_NAME/sample1.wav")
        sys.exit(1)

    agent_dirs = [
        name
        for name in os.listdir(args.agents_dir)
        if os.path.isdir(os.path.join(args.agents_dir, name))
    ]
    if not agent_dirs:
        print(f"ERROR: No agent directories found in {args.agents_dir}")
        print("Create subdirectories like: agent_adil, agent_angelina")
        sys.exit(1)

    print(f"Found {len(agent_dirs)} agent directories:")
    non_empty_agent_dirs = []
    for name in sorted(agent_dirs):
        audio_count = len(
            [
                filename
                for filename in os.listdir(os.path.join(args.agents_dir, name))
                if filename.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
            ]
        )
        print(f"  {name}: {audio_count} audio files")
        if audio_count > 0:
            non_empty_agent_dirs.append(name)

    if not non_empty_agent_dirs:
        print("\nERROR: No audio files found in any agent directory.")
        sys.exit(1)

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

    print(f"\n{'=' * 50}")
    print("ENROLLMENT COMPLETE")
    print(f"{'=' * 50}")
    print(f"Agents enrolled: {len(agent_embeddings)}")
    for name in sorted(agent_embeddings.keys()):
        print(f"  [OK] {name}")
    print(f"\nEmbeddings saved to: {args.output}")
    print("\nYou can now run:")
    print("  python main.py --input path/to/call.wav")


if __name__ == "__main__":
    main()
