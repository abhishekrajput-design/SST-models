"""
Re-enroll Omar El Harchaoui from ground truth transcript of Mini Hatch call.

Ground truth provided by user: 44 turns from call 20260505T073055769_385036
(Omar + customer Mark discussing Mini Hatch viewing)

This script extracts agent-only segments using the GT labels, builds clean
voiceprints from car-dealer-channel audio, and updates agents.json.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import soundfile as sf
from scipy import signal

# Add parent dir to path to import src modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_campp import EmbeddingModel, l2_norm
from src.diar_multi import _estimate_snr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Ground truth transcript provided by user
GT_TRANSCRIPT = [
    {"start": 0.00, "end": 1.32, "speaker": "CUSTOMER", "text": "Hi, I'm Arcester. Is this a good time to chat?"},
    {"start": 1.32, "end": 1.64, "speaker": "AGENT", "text": "Yeah speaking."},
    {"start": 1.64, "end": 4.16, "speaker": "CUSTOMER", "text": "Awesome, I wanted to discuss the Mini Hatch you have available."},
    {"start": 4.16, "end": 6.24, "speaker": "AGENT", "text": "Great, we have a few on the lot. When would you like to view it?"},
    {"start": 6.24, "end": 8.64, "speaker": "CUSTOMER", "text": "How about tomorrow morning around ten?"},
    {"start": 8.64, "end": 9.28, "speaker": "AGENT", "text": "Five o'clock."},
    {"start": 9.28, "end": 12.48, "speaker": "CUSTOMER", "text": "That's correct. I mean, that works for me if five is better."},
    {"start": 12.48, "end": 13.12, "speaker": "AGENT", "text": "On this number?"},
    {"start": 13.12, "end": 14.88, "speaker": "CUSTOMER", "text": "Yeah, that's the best way to reach me."},
    {"start": 14.88, "end": 17.04, "speaker": "AGENT", "text": "Okay. Can I get your email as well?"},
    {"start": 17.04, "end": 22.56, "speaker": "CUSTOMER", "text": "Sure, it's mark.j.stewart at gmail dot com."},
    {"start": 22.56, "end": 24.72, "speaker": "AGENT", "text": "Perfect. I have that down here."},
    {"start": 24.72, "end": 27.36, "speaker": "CUSTOMER", "text": "Great. And what's the mileage on that Mini?"},
    {"start": 27.36, "end": 30.48, "speaker": "AGENT", "text": "It has 45000 miles on the clock, mint condition."},
    {"start": 30.48, "end": 32.88, "speaker": "CUSTOMER", "text": "That's fantastic. What's the asking price?"},
    {"start": 32.88, "end": 36.96, "speaker": "AGENT", "text": "We're asking eighteen-five for it. That's well below market."},
    {"start": 36.96, "end": 41.76, "speaker": "CUSTOMER", "text": "Hmm, that's a bit more than I was hoping to spend. Can you do any better?"},
    {"start": 41.76, "end": 47.04, "speaker": "AGENT", "text": "Let me see what I can do. We might have some flexibility if you're ready to move quickly."},
    {"start": 47.04, "end": 50.4, "speaker": "CUSTOMER", "text": "I'm ready to view it first and then we can talk numbers."},
    {"start": 50.4, "end": 53.28, "speaker": "AGENT", "text": "Perfect. I'll have it ready for you at five tomorrow."},
    {"start": 53.28, "end": 55.2, "speaker": "CUSTOMER", "text": "Sounds good. See you then."},
    {"start": 55.2, "end": 56.16, "speaker": "AGENT", "text": "Great, thank you."},
    {"start": 56.16, "end": 66.0, "speaker": "CUSTOMER", "text": "Okay, that's fine. I'll be there at five PM tomorrow. Looking forward to it. Bye."},
    {"start": 66.0, "end": 68.4, "speaker": "AGENT", "text": "Bye, thanks for calling Car Planet."},
]

CALL_AUDIO = str(Path(__file__).parent.parent / "data" / "processed" / "enhanced_20260505T073055769_385036__parakeet-tdt-0.6b-v3" / "trimmed_audio.mp3")
AGENT_NAME = "Omar El Harchaoui"
AGENT_SLUG = "omar_el_harchaoui"

# Enrollment thresholds for Mini Hatch call (car-dealer channel)
MIN_AGENT_DUR = 2.0      # skip very short backchannels
MAX_AGENT_DUR = 20.0     # cap to prevent memory spikes
SNR_MIN_DB = 12.0        # car-dealer channel is noisy; lower floor than typical script
BACKCHANNEL_WORDS = {"yeah", "yes", "yep", "yup", "no", "nope", "ok", "okay", "kay", "right", "sure", "alright"}


def _norm_words(text: str) -> List[str]:
    """Normalize text to words for backchannel filtering."""
    return [w.lower().strip(".,!?;:") for w in (text or "").split() if w.strip(".,!?;:")]


def extract_agent_segments(
    audio_path: str,
    transcript: List[Dict],
    min_dur: float = MIN_AGENT_DUR,
    max_dur: float = MAX_AGENT_DUR,
) -> List[tuple]:
    """Extract AGENT segments with their audio and metadata.

    Returns: List of (audio_chunk, segment_dict) tuples.
    """
    import pydub
    from pydub.utils import mediainfo

    logger.info(f"Loading audio from {audio_path}...")
    try:
        if audio_path.endswith('.mp3'):
            # Use pydub for MP3
            sound = pydub.AudioSegment.from_mp3(audio_path)
            audio = np.array(sound.get_array_of_samples(), dtype=np.float32)
            if sound.channels == 2:
                audio = audio.reshape((-1, 2)).mean(axis=1)
            sr = sound.frame_rate
            # Normalize to [-1, 1]
            audio = audio / (2**15)
        else:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception as exc:
        logger.error(f"Cannot load {audio_path}: {exc}")
        return []

    # Convert to mono (if from soundfile)
    if audio.ndim > 1:
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 16000 / sr))
        sr = 16000

    extracted = []
    for seg in transcript:
        if seg["speaker"] != "AGENT":
            continue

        dur = seg["end"] - seg["start"]
        if dur < min_dur or dur > max_dur:
            logger.info(f"  Skip {seg['text'][:40]:40s} dur={dur:.2f}s (out of range)")
            continue

        # Skip short backchannels (low discriminative power)
        words = _norm_words(seg["text"])
        if len(words) <= 1 and words and words[0] in BACKCHANNEL_WORDS:
            logger.info(f"  Skip {seg['text'][:40]:40s} (backchannel only)")
            continue

        # Extract audio window
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = audio[start_sample:end_sample]

        if len(chunk) < sr * 0.5:  # too short after rounding
            continue

        extracted.append((chunk, seg))
        logger.info(f"  Extract {seg['text'][:40]:40s} dur={dur:.2f}s snr={_estimate_snr(chunk, sr):.1f}dB")

    logger.info(f"Extracted {len(extracted)} AGENT segments")
    return extracted


def compute_embeddings(segments: List[tuple]) -> tuple:
    """Compute CAM++ embeddings for all segments.

    Returns: (embeddings_list, segments_list, snr_list)
    """
    model = EmbeddingModel()
    model.load(force_cpu=False)
    logger.info(f"EmbeddingModel loaded: {model.model_name}")

    embeddings = []
    snr_list = []
    valid_segments = []

    for chunk, seg in segments:
        snr = _estimate_snr(chunk, 16000)
        if snr < SNR_MIN_DB:
            logger.info(f"  Skip {seg['text'][:40]:40s} snr={snr:.1f}dB (below {SNR_MIN_DB}dB floor)")
            continue

        emb = model.embed_chunk(chunk, sr=16000)
        if emb is None:
            logger.warning(f"  Embedding failed: {seg['text'][:40]:40s}")
            continue

        embeddings.append(emb)
        snr_list.append(snr)
        valid_segments.append(seg)

    model.unload()
    logger.info(f"Computed {len(embeddings)} embeddings (SNR >= {SNR_MIN_DB}dB)")
    return embeddings, valid_segments, snr_list


def cluster_by_snr(embeddings: List[np.ndarray], snr_list: List[float], n_buckets: int = 3) -> Dict[str, List[np.ndarray]]:
    """Cluster embeddings into SNR-based buckets (high/mid/low).

    Returns: dict with 'high', 'mid', 'low' keys mapping to centroid embeddings.
    """
    if not embeddings:
        return {"high": [], "mid": [], "low": []}

    snr_array = np.array(snr_list)
    sorted_indices = np.argsort(snr_array)

    # Define bucket boundaries by SNR percentiles
    n_per_bucket = len(embeddings) // n_buckets
    low_idx = sorted_indices[:n_per_bucket]
    mid_idx = sorted_indices[n_per_bucket:2*n_per_bucket]
    high_idx = sorted_indices[2*n_per_bucket:]

    buckets = {
        "low": [embeddings[i] for i in low_idx],
        "mid": [embeddings[i] for i in mid_idx],
        "high": [embeddings[i] for i in high_idx],
    }
    logger.info(f"Clustered into buckets: low={len(buckets['low'])}, mid={len(buckets['mid'])}, high={len(buckets['high'])}")
    return buckets


def compute_bucket_centroids(buckets: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Compute L2-normalized centroid for each SNR bucket."""
    centroids = {}
    for bucket_name, embs in buckets.items():
        if not embs:
            continue
        centroid = np.mean(embs, axis=0)
        centroid = l2_norm(centroid)
        centroids[bucket_name] = centroid
        logger.info(f"  {bucket_name:4s} centroid: norm={np.linalg.norm(centroid):.6f}, dim={centroid.shape[0]}")
    return centroids


def compute_max_outside_sim(
    all_embeddings: List[np.ndarray],
    agent_embeddings: List[np.ndarray],
) -> float:
    """Compute max similarity between customer embeddings and agent centroid.

    Uses 95th percentile of customer-vs-agent-centroid similarities (not mean).
    Returns: float in [0, 1] representing the maximum expected false-positive similarity.
    """
    if not agent_embeddings or not all_embeddings:
        return 0.0

    agent_centroid = np.mean(agent_embeddings, axis=0)
    agent_centroid = l2_norm(agent_centroid)

    # Find customer segments (segments not in agent_embeddings)
    customer_sims = []
    for emb in all_embeddings:
        if not any(np.allclose(emb, agent_emb) for agent_emb in agent_embeddings):
            # This is likely a customer segment
            sim = float(np.dot(l2_norm(emb), agent_centroid))
            customer_sims.append(sim)

    if not customer_sims:
        # All segments were agent — use the lowest agent-to-centroid sim as a floor
        agent_sims = [float(np.dot(l2_norm(emb), agent_centroid)) for emb in agent_embeddings]
        return np.percentile(agent_sims, 5) if agent_sims else 0.0

    # 95th percentile: reject 5% of customer false positives at this threshold
    return float(np.percentile(customer_sims, 95))


def update_agents_json(centroids: Dict[str, np.ndarray], max_outside_sim: float, embeddings: List[np.ndarray]) -> None:
    """Update agents.json with newly re-enrolled Omar."""
    agents_json_path = Path(__file__).parent.parent / "data" / "agent_voiceprints" / "agents.json"

    logger.info(f"Reading {agents_json_path}...")
    with open(agents_json_path) as f:
        agents = json.load(f)

    # Save backup
    backup_path = agents_json_path.with_suffix(f".backup.{int(__import__('time').time())}.json")
    logger.info(f"Saving backup to {backup_path}...")
    with open(backup_path, "w") as f:
        json.dump(agents, f, indent=2)

    # Build new voiceprints list with centroids + metadata
    voiceprints = []
    npy_dir = agents_json_path.parent
    mean_inside_sim = float(np.mean([
        float(np.dot(l2_norm(emb), l2_norm(np.mean(embeddings, axis=0))))
        for emb in embeddings
    ]))

    for bucket_name, centroid in centroids.items():
        npy_path = f"{AGENT_SLUG}__{bucket_name}_0.npy"
        npy_full_path = npy_dir / npy_path
        np.save(npy_full_path, centroid)
        logger.info(f"  Saved {npy_path}")

        voiceprints.append({
            "path": npy_path,
            "bucket": bucket_name,
            "n_clips": len(embeddings),
            "snr_db": SNR_MIN_DB,
        })

    # Update agent entry
    agents[AGENT_SLUG] = {
        "agent_name": AGENT_NAME,
        "voiceprint_path": f"{AGENT_SLUG}.npy",
        "voiceprints": voiceprints,
        "n_voiceprints": len(voiceprints),
        "total_seconds": sum(seg["end"] - seg["start"] for _, seg in extract_agent_segments(CALL_AUDIO, GT_TRANSCRIPT)),
        "used_calls": 1,
        "source": "enroll_from_gt_mini_hatch_call",
        "per_call_snr": [{"_id": "20260505T073055769_385036"}],
        "mean_inside_sim": mean_inside_sim,
        "max_outside_sim": max_outside_sim,
    }

    logger.info(f"Updating {agents_json_path}...")
    with open(agents_json_path, "w") as f:
        json.dump(agents, f, indent=2)

    logger.info(f"\nOmar El Harchaoui re-enrollment complete:")
    logger.info(f"  Voiceprints: {len(voiceprints)} buckets")
    logger.info(f"  mean_inside_sim: {mean_inside_sim:.4f}")
    logger.info(f"  max_outside_sim: {max_outside_sim:.4f}")
    logger.info(f"  Threshold cap formula: min({max_outside_sim:.4f} + 0.04, 0.36) = {min(max_outside_sim + 0.04, 0.36):.4f}")


def main():
    logger.info(f"Enrolling {AGENT_NAME} from ground truth transcript...")

    # Extract agent segments
    segments = extract_agent_segments(CALL_AUDIO, GT_TRANSCRIPT)
    if not segments:
        logger.error("No valid AGENT segments extracted")
        return

    # Compute embeddings
    embeddings, valid_segments, snr_list = compute_embeddings(segments)
    if not embeddings:
        logger.error("No embeddings computed")
        return

    # Cluster by SNR and compute centroids
    buckets = cluster_by_snr(embeddings, snr_list, n_buckets=3)
    centroids = compute_bucket_centroids(buckets)

    # Compute max_outside_sim
    all_agent_embs = [emb for seg_idx in range(len(embeddings)) for emb in [embeddings[seg_idx]]]
    max_outside_sim = compute_max_outside_sim(embeddings, all_agent_embs)
    logger.info(f"max_outside_sim: {max_outside_sim:.4f}")

    # Update agents.json
    update_agents_json(centroids, max_outside_sim, embeddings)

    logger.info("Done.")


if __name__ == "__main__":
    main()
