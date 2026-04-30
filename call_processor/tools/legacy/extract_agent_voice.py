"""
Agent Voice Extractor

These desk recordings contain the agent talking to multiple customers over
30 minutes. We need to extract ONLY the agent's voice segments for enrollment.

Strategy:
  1. Use pyannote diarization on each recording to detect speakers
  2. The agent is typically the MOST FREQUENT speaker (they're at their desk)
  3. Extract the dominant speaker's segments as clean voice clips
  4. Save these clips for embedding enrollment

Usage:
    python extract_agent_voice.py
    python extract_agent_voice.py --agent agent_adil --max-clips 10
    python extract_agent_voice.py --all --min-duration 3 --max-duration 15
"""

import os
import sys
import gc
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import soundfile as sf
import numpy as np

# Fix Windows console encoding
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


def get_dominant_speaker(diarization_result) -> str:
    """Find the speaker with the most total speaking time."""
    speaker_durations = defaultdict(float)
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_durations[speaker] += turn.end - turn.start

    if not speaker_durations:
        return None

    dominant = max(speaker_durations, key=speaker_durations.get)
    total = sum(speaker_durations.values())
    logger.info(f"Speaker breakdown:")
    for spk, dur in sorted(speaker_durations.items(), key=lambda x: -x[1]):
        pct = (dur / total) * 100
        marker = " <-- AGENT (dominant)" if spk == dominant else ""
        logger.info(f"  {spk}: {dur:.1f}s ({pct:.0f}%){marker}")

    return dominant


def extract_clips_for_agent(
    audio_path: str,
    diarization_result,
    dominant_speaker: str,
    output_dir: str,
    clip_prefix: str,
    min_duration: float = 3.0,
    max_duration: float = 15.0,
    max_clips: int = 20,
) -> list:
    """Extract clean voice clips of the dominant speaker."""
    os.makedirs(output_dir, exist_ok=True)

    # Load audio using soundfile (avoids torchcodec DLL crash on Windows)
    waveform_np, sample_rate = sf.read(audio_path, always_2d=True)
    waveform = torch.from_numpy(waveform_np.T).float()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Collect ALL segments for the dominant speaker first
    raw_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        if speaker == dominant_speaker:
            raw_segments.append([turn.start, turn.end])

    # Merge segments that are closer than 0.5s to get longer chunks
    merged_segments = []
    if raw_segments:
        raw_segments.sort(key=lambda x: x[0])
        current = raw_segments[0]
        for nxt in raw_segments[1:]:
            if nxt[0] - current[1] <= 0.5:
                # Merge
                current[1] = max(current[1], nxt[1])
            else:
                merged_segments.append((current[0], current[1], current[1] - current[0]))
                current = nxt
        merged_segments.append((current[0], current[1], current[1] - current[0]))

    # Now filter by duration
    segments = []
    for start, end, duration in merged_segments:
        if duration >= 1.5 and duration <= 20.0:
            segments.append((start, end, duration))

    # Sort by duration (prefer longer, cleaner segments)
    segments.sort(key=lambda x: -x[2])
    segments = segments[:max_clips]

    saved_paths = []
    for i, (start, end, dur) in enumerate(segments):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        chunk = waveform[:, start_sample:end_sample]

        # Basic energy check — skip very quiet segments (likely silence/noise)
        rms = chunk.float().pow(2).mean().sqrt().item()
        if rms < 0.001:
            logger.debug(f"Skipping low-energy segment {start:.1f}-{end:.1f}s (RMS={rms:.5f})")
            continue

        filename = f"{clip_prefix}_clip{i:02d}_{start:.0f}s_{end:.0f}s.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, chunk.squeeze().numpy(), sample_rate)
        saved_paths.append(filepath)

    logger.info(f"Saved {len(saved_paths)} clips ({min_duration}-{max_duration}s range)")
    return saved_paths


def process_agent(
    agent_dir: str,
    output_dir: str,
    pipeline,
    min_duration: float = 3.0,
    max_duration: float = 15.0,
    max_clips_per_recording: int = 5,
    max_recordings: int = 3,
):
    """Process recordings for a single agent directory."""
    agent_dir = Path(agent_dir)
    agent_name = agent_dir.name
    clean_output = os.path.join(output_dir, agent_name)

    audio_files = sorted([
        f for f in agent_dir.iterdir()
        if f.suffix.lower() in ('.mp3', '.wav', '.flac', '.m4a', '.ogg')
    ])

    if not audio_files:
        logger.warning(f"No audio files in {agent_dir}")
        return

    # Use a subset of recordings (don't need all 10)
    audio_files = audio_files[:max_recordings]
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {agent_name} ({len(audio_files)} recordings)")
    logger.info(f"{'='*60}")

    total_clips = 0
    for j, audio_file in enumerate(audio_files):
        logger.info(f"\n  [{j+1}/{len(audio_files)}] {audio_file.name}")
        try:
            # Load audio with soundfile to avoid torchcodec issues
            wav_np, sr = sf.read(str(audio_file), always_2d=True)
            wav_tensor = torch.from_numpy(wav_np.T).float()
            diarization = pipeline({"waveform": wav_tensor, "sample_rate": sr})
            if hasattr(diarization, "speaker_diarization"):
                diarization = diarization.speaker_diarization
            dominant = get_dominant_speaker(diarization)
            if dominant is None:
                logger.warning(f"  No speakers detected, skipping")
                continue

            clip_prefix = f"{agent_name}_rec{j:02d}"
            clips = extract_clips_for_agent(
                str(audio_file),
                diarization,
                dominant,
                clean_output,
                clip_prefix,
                min_duration=min_duration,
                max_duration=max_duration,
                max_clips=max_clips_per_recording,
            )
            total_clips += len(clips)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    logger.info(f"\n  {agent_name}: Total {total_clips} clean voice clips saved to {clean_output}")
    return total_clips


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean agent voice clips from desk recordings"
    )
    parser.add_argument(
        "--agents-dir",
        default="data/agent_samples",
        help="Directory with agent_NAME/ subdirectories containing recordings",
    )
    parser.add_argument(
        "--output-dir",
        default="data/agent_clean_clips",
        help="Output directory for extracted clean clips",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Process only this agent directory (e.g., agent_adil)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all agents",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum clip duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum clip duration in seconds",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=5,
        help="Max clips to extract per recording",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=3,
        help="Max recordings to process per agent (saves time)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for diarization",
    )

    args = parser.parse_args()

    # Resolve HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HuggingFace token required for pyannote.audio diarization")
        print("Set via: $env:HF_TOKEN = 'hf_your_token'")
        sys.exit(1)

    # Load pyannote pipeline ONCE
    from pyannote.audio import Pipeline
    import huggingface_hub
    huggingface_hub.login(token=hf_token)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading pyannote diarization pipeline on {device}...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )
    pipeline.to(device)
    logger.info("Diarization pipeline loaded")

    agents_dir = Path(args.agents_dir)
    if not agents_dir.is_dir():
        print(f"ERROR: Agents directory not found: {agents_dir}")
        sys.exit(1)

    # Determine which agents to process
    if args.agent:
        agent_dirs = [agents_dir / args.agent]
        if not agent_dirs[0].is_dir():
            print(f"ERROR: Agent directory not found: {agent_dirs[0]}")
            sys.exit(1)
    else:
        agent_dirs = sorted([
            d for d in agents_dir.iterdir()
            if d.is_dir() and d.name.startswith("agent_")
        ])

    if not agent_dirs:
        print("No agent directories found")
        sys.exit(1)

    # Process each agent
    total_all = 0
    for agent_dir in agent_dirs:
        clips = process_agent(
            str(agent_dir),
            args.output_dir,
            pipeline,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            max_clips_per_recording=args.max_clips,
            max_recordings=args.max_recordings,
        )
        total_all += (clips or 0)

    # Cleanup
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total clips extracted: {total_all}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nNext step: Run enrollment on clean clips:")
    print(f"  python enroll_agents.py --agents-dir {args.output_dir}")


if __name__ == "__main__":
    main()
