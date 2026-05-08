"""
Clean speaker-first diarization pipeline.

Architecture:
  Stage 1 — Pure diarization: detect distinct speakers (SPEAKER_00, SPEAKER_01, ...)
            using NeMo Sortformer (best for 2-4 speakers, phone audio) with
            pyannote fallback for >4 speakers.
  Stage 2 — Voiceprint matching: compute per-cluster centroid, match against
            enrolled voiceprints, assign agent_name to the highest-similarity cluster.
  Stage 3 — Role labeling: cluster matching agent → AGENT, all others → CUSTOMER (or
            CUSTOMER_1, CUSTOMER_2, ... if multiple non-agent clusters).

No hardcoded heuristics. No anti-flip passes. No filler/farewell special cases.
Just: diarize → cluster centroids → voiceprint match → label.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (no magic strings)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"
DEFAULT_HF_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"

# Voiceprint matching threshold — if best agent cluster's mean similarity is below
# this floor, we consider the agent "not present" and label all clusters CUSTOMER_*
AGENT_PRESENCE_FLOOR = 0.35

# Minimum number of segments a cluster must contain to be considered a speaker.
MIN_CLUSTER_SEGMENTS = 2

# Display labels (configurable, no hardcoded strings scattered in logic)
LABEL_AGENT = "AGENT"
LABEL_CUSTOMER = "CUSTOMER"
LABEL_UNKNOWN = "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DIARIZATION (speaker count + per-segment speaker IDs)
# ──────────────────────────────────────────────────────────────────────────────

class DiarizationBackend:
    """Abstract diarizer that returns [(start, end, speaker_id), ...]."""

    def diarize(self, audio_path: str, max_speakers: Optional[int] = None) -> List[Dict]:
        raise NotImplementedError


class SortformerDiarizer(DiarizationBackend):
    """NVIDIA NeMo Sortformer — best for 2-4 speaker phone calls (2024-2025 SOTA).

    Uses the full model by default. The streaming variant remains available
    for low-memory fallback.
    """

    # Streaming variant has chunked inference — much smaller GPU footprint.
    STREAMING_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"

    def __init__(self, model_name: str = DEFAULT_SORTFORMER_MODEL, use_streaming: bool = False):
        self.model_name = self.STREAMING_MODEL if use_streaming else model_name
        self._model = None

    def load(self):
        if self._model is not None:
            return
        import torch, gc
        # Aggressively free GPU memory before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        from nemo.collections.asr.models import SortformerEncLabelModel
        logger.info(f"Loading Sortformer: {self.model_name}")
        self._model = SortformerEncLabelModel.from_pretrained(self.model_name)
        self._model = self._model.to("cuda")
        self._model.eval()
        logger.info("Sortformer loaded on cuda")

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def diarize(self, audio_path: str, max_speakers: Optional[int] = None) -> List[Dict]:
        self.load()
        # Sortformer returns per-frame speaker probabilities; we threshold and merge.
        results = self._model.diarize(audio=[audio_path], batch_size=1)
        # results is List[List[str]] — each inner list contains "start end speaker" lines
        if not results or not results[0]:
            return []

        segments = []
        for line in results[0]:
            parts = str(line).strip().split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    spk = parts[2]
                    segments.append({"start": start, "end": end, "speaker": spk})
                except (ValueError, IndexError):
                    continue
        return _merge_consecutive_same_speaker(segments)


class PyannoteDiarizer(DiarizationBackend):
    """Pyannote community-1 — unlimited speakers, best for 5+ speaker scenarios."""

    def __init__(self, hf_token: str, model_name: str = DEFAULT_HF_PYANNOTE_MODEL):
        self.hf_token = hf_token
        self.model_name = model_name
        self._pipeline = None

    def load(self):
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline
        logger.info(f"Loading pyannote: {self.model_name}")
        self._pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=self.hf_token)

    def diarize(self, audio_path: str, max_speakers: Optional[int] = None) -> List[Dict]:
        self.load()
        kwargs = {}
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

        diarization = self._pipeline(audio_path, **kwargs)
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,  # pyannote already gives "SPEAKER_00", "SPEAKER_01"
            })
        return _merge_consecutive_same_speaker(segments)


def _merge_consecutive_same_speaker(segments: List[Dict], gap_tolerance: float = 0.5) -> List[Dict]:
    """Merge adjacent same-speaker segments (gap < tolerance)."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s["start"])
    merged = [dict(segments[0])]
    for s in segments[1:]:
        last = merged[-1]
        if s["speaker"] == last["speaker"] and (s["start"] - last["end"]) <= gap_tolerance:
            last["end"] = s["end"]
        else:
            merged.append(dict(s))
    return merged


def _renumber_speakers_to_canonical(segments: List[Dict]) -> List[Dict]:
    """
    Rename speakers in arrival order: first speaker → SPEAKER_00, next new → SPEAKER_01, ...
    Ensures consistent IDs regardless of backend.
    """
    mapping: Dict[str, str] = {}
    counter = 0
    out = []
    for s in sorted(segments, key=lambda x: x["start"]):
        orig = s["speaker"]
        if orig not in mapping:
            mapping[orig] = f"SPEAKER_{counter:02d}"
            counter += 1
        out.append({**s, "speaker": mapping[orig]})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — VOICEPRINT MATCHING (which cluster is the agent?)
# ──────────────────────────────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _embed_segment(audio: np.ndarray, sr: int, start: float, end: float, embedder) -> Optional[np.ndarray]:
    start_samp = int(start * sr)
    end_samp = int(end * sr)
    if end_samp > len(audio) or end_samp - start_samp < int(0.5 * sr):
        return None
    window = audio[start_samp:end_samp]
    try:
        emb = embedder.embed_chunk(window, sr=sr)
        if emb is None or np.isnan(emb).any():
            return None
        return _l2_norm(emb)
    except Exception:
        return None


def compute_cluster_centroids(
    speaker_segments: List[Dict],
    audio: np.ndarray,
    sr: int,
    embedder,
    min_segments: int = MIN_CLUSTER_SEGMENTS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """For each SPEAKER_NN, compute centroid embedding from its segments."""
    by_speaker: Dict[str, List[np.ndarray]] = {}
    for seg in speaker_segments:
        emb = _embed_segment(audio, sr, seg["start"], seg["end"], embedder)
        if emb is not None:
            by_speaker.setdefault(seg["speaker"], []).append(emb)

    centroids = {}
    counts = {}
    for spk, embs in by_speaker.items():
        if len(embs) < min_segments:
            continue
        centroid = np.mean(np.stack(embs), axis=0)
        centroids[spk] = _l2_norm(centroid)
        counts[spk] = len(embs)
    return centroids, counts


def match_agent_to_cluster(
    cluster_centroids: Dict[str, np.ndarray],
    enrolled_voiceprints: Dict[str, Tuple[str, np.ndarray]],
    presence_floor: float = AGENT_PRESENCE_FLOOR,
    target_agent_slug: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Dict]]:
    """
    Find which speaker cluster best matches an enrolled agent.

    Args:
        cluster_centroids: {SPEAKER_NN: centroid_vector}
        enrolled_voiceprints: {agent_slug: (display_name, voiceprint_stack [N, dim])}
        presence_floor: min similarity to assert agent is present

    Returns:
        (agent_speaker_id, agent_slug, best_similarity, full_match_table)
        agent_speaker_id is None if no cluster passes the floor.
    """
    match_table: Dict[str, Dict] = {}
    best_overall = (None, None, 0.0)
    skipped_dim = 0
    if target_agent_slug:
        enrolled_voiceprints = {
            slug: value
            for slug, value in enrolled_voiceprints.items()
            if slug == target_agent_slug
        }
        if not enrolled_voiceprints:
            logger.warning("Target agent slug %s not found in enrolled voiceprints", target_agent_slug)

    for spk, centroid in cluster_centroids.items():
        cluster_match = {}
        centroid_dim = centroid.shape[0]
        for slug, (name, vp_stack) in enrolled_voiceprints.items():
            # Skip voiceprints whose dim doesn't match cluster centroid (e.g. CAM++ 512 vs TitaNet 192)
            if vp_stack.shape[1] != centroid_dim:
                skipped_dim += 1
                continue
            # Max cosine across all voiceprints for this agent (handles multi-cluster enrollment)
            sims = vp_stack @ centroid  # (N,)
            best_for_agent = float(np.max(sims)) if len(sims) > 0 else 0.0
            cluster_match[slug] = {"name": name, "similarity": best_for_agent}

            if best_for_agent > best_overall[2]:
                best_overall = (spk, slug, best_for_agent)

        match_table[spk] = cluster_match

    if skipped_dim > 0:
        logger.info(f"Skipped {skipped_dim} dim-mismatched voiceprints during matching")

    agent_spk, agent_slug, best_sim = best_overall
    if best_sim < presence_floor:
        return None, None, best_sim, match_table
    return agent_spk, agent_slug, best_sim, match_table


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ROLE LABELING (no hardcoded "AGENT"/"CUSTOMER" strings — use constants)
# ──────────────────────────────────────────────────────────────────────────────

def assign_roles(
    speaker_segments: List[Dict],
    agent_speaker_id: Optional[str],
    agent_slug: Optional[str],
    agent_name: Optional[str],
) -> List[Dict]:
    """
    Apply role labels in-place based on which SPEAKER_NN is the agent.

    All segments belonging to agent_speaker_id get role=AGENT + agent_name.
    Other speakers become CUSTOMER (single) or CUSTOMER_1/2/... (multiple).
    """
    # Collect non-agent speaker IDs in arrival order
    non_agent_speakers = []
    for s in speaker_segments:
        if s["speaker"] != agent_speaker_id and s["speaker"] not in non_agent_speakers:
            non_agent_speakers.append(s["speaker"])

    # Map non-agents to display labels. The machine-readable role stays
    # CUSTOMER for every non-agent speaker; the original SPEAKER_NN id is kept
    # separately in "speaker" so downstream systems can still inspect speaker IDs.
    if len(non_agent_speakers) == 1:
        customer_label = {non_agent_speakers[0]: "Customer"}
    else:
        customer_label = {
            spk: f"Customer {i+1}"
            for i, spk in enumerate(non_agent_speakers)
        }

    out = []
    for s in speaker_segments:
        seg = dict(s)
        if s["speaker"] == agent_speaker_id and agent_speaker_id is not None:
            seg["identified_speaker"] = LABEL_AGENT
            if agent_name:
                seg["agent_name"] = agent_name
            if agent_slug:
                seg["agent_slug"] = agent_slug
            seg["display_speaker"] = agent_name or LABEL_AGENT
        else:
            seg["identified_speaker"] = LABEL_CUSTOMER
            seg["display_speaker"] = customer_label.get(s["speaker"], LABEL_UNKNOWN)
        out.append(seg)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def diarize_clean(
    audio_path: str,
    transcribed_segments: Optional[List[Dict]] = None,
    backend: str = "auto",
    max_speakers: Optional[int] = 4,
    sortformer_streaming: bool = False,
    embedder=None,
    voiceprints: Optional[Dict[str, Tuple[str, np.ndarray]]] = None,
    hf_token: Optional[str] = None,
    presence_floor: float = AGENT_PRESENCE_FLOOR,
    target_agent_slug: Optional[str] = None,
) -> Dict:
    """
    Run clean speaker-first diarization + agent identification.

    Args:
        audio_path: 16 kHz mono WAV/MP3
        transcribed_segments: optional [{start, end, text}] to overlay text on speaker segments
        backend: "sortformer" | "pyannote" | "auto" (auto picks sortformer if max_speakers<=4 else pyannote)
        max_speakers: hint for diarizer (None = auto-detect)
        embedder: speaker embedder (must have embed_chunk(audio, sr) method).
                  Defaults to CAM++ via embedding_campp.get_model().
        voiceprints: {slug: (name, stack)} — defaults to loading from agents.json
        hf_token: needed for pyannote backend
        presence_floor: min cosine sim to assert agent is on call

    Returns: {
        "segments": [...],            # text segments with identified_speaker labels
        "speaker_segments": [...],    # raw diarization output (no text)
        "speaker_count": int,
        "agent_speaker_id": "SPEAKER_NN" or None,
        "agent_name": str or None,
        "agent_slug": str or None,
        "agent_similarity": float,
        "cluster_match_table": {...},  # full sim matrix per (speaker, agent_slug)
        "backend": "sortformer" | "pyannote",
    }
    """
    # ── Stage 1: Diarize ──
    if backend == "auto":
        backend = "sortformer" if (max_speakers is not None and max_speakers <= 4) else "pyannote"

    if backend == "sortformer":
        diarizer = SortformerDiarizer(use_streaming=sortformer_streaming)
    elif backend == "pyannote":
        if not hf_token:
            raise ValueError("pyannote backend requires hf_token")
        diarizer = PyannoteDiarizer(hf_token=hf_token)
    else:
        raise ValueError(f"unknown backend: {backend}")

    speaker_segments = diarizer.diarize(audio_path, max_speakers=max_speakers)
    speaker_segments = _renumber_speakers_to_canonical(speaker_segments)

    speaker_count = len(set(s["speaker"] for s in speaker_segments))
    logger.info(f"Diarization detected {speaker_count} speakers across {len(speaker_segments)} segments")

    # ── Stage 2: Voiceprint matching ──
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception:
            pass

    if embedder is None:
        from src.embedding_campp import get_model
        embedder = get_model(force_cpu=True)

    if voiceprints is None:
        from src.diar_multi import _load_voiceprints
        voiceprints = _load_voiceprints()

    cluster_centroids, cluster_counts = compute_cluster_centroids(
        speaker_segments, audio, sr, embedder
    )

    agent_spk, agent_slug, agent_sim, match_table = match_agent_to_cluster(
        cluster_centroids,
        voiceprints,
        presence_floor=presence_floor,
        target_agent_slug=target_agent_slug,
    )
    agent_name = voiceprints[agent_slug][0] if agent_slug else None

    logger.info(
        f"Agent matching: speaker={agent_spk}, agent={agent_name} "
        f"({agent_slug}), similarity={agent_sim:.3f}"
    )

    # ── Stage 3: Apply role labels ──
    labeled_speaker_segments = assign_roles(
        speaker_segments, agent_spk, agent_slug, agent_name
    )
    per_speaker: Dict[str, Dict[str, float]] = {}
    for seg in labeled_speaker_segments:
        label = seg.get("display_speaker") or seg.get("speaker") or LABEL_UNKNOWN
        per_speaker.setdefault(label, {"turns": 0, "seconds": 0.0})
        per_speaker[label]["turns"] += 1
        per_speaker[label]["seconds"] += max(float(seg["end"]) - float(seg["start"]), 0.0)

    # ── Overlay onto transcribed segments if provided ──
    text_segments = []
    if transcribed_segments:
        text_segments = _overlay_speakers_on_text(transcribed_segments, labeled_speaker_segments)

    return {
        "segments": text_segments,
        "speaker_segments": labeled_speaker_segments,
        "speaker_count": speaker_count,
        "agent_speaker_id": agent_spk,
        "agent_name": agent_name,
        "agent_slug": agent_slug,
        "agent_similarity": float(agent_sim),
        "target_agent_slug": target_agent_slug,
        "cluster_match_table": match_table,
        "cluster_segment_counts": cluster_counts,
        "per_speaker": per_speaker,
        "speaker_mode": "speaker_first_voiceprint",
        "speaker_id_backend": backend,
        "sortformer_streaming": bool(sortformer_streaming) if backend == "sortformer" else False,
        "matched_backend_dim": getattr(embedder, "dim", None),
        "voiceprint_dims": {
            str(vp_stack.shape[1]): int(vp_stack.shape[0])
            for _, vp_stack in voiceprints.values()
            if getattr(vp_stack, "ndim", 0) == 2
        },
        "backend": backend,
    }


def detect_leading_speech_offset(audio_path: str) -> float:
    """Estimate leading silence offset for training JSON timestamp alignment."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    frame = max(int(0.05 * sr), 1)
    hop = max(int(0.01 * sr), 1)
    if len(audio) <= frame:
        return 0.0
    values = []
    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start:start + frame]
        values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    floor = float(np.percentile(arr, 20))
    peak = float(np.percentile(arr, 95))
    threshold = max(0.004, floor + (peak - floor) * 0.15)
    for i, value in enumerate(arr):
        if value >= threshold:
            return round(i * hop / sr, 3)
    return 0.0


def _overlay_speakers_on_text(text_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
    """For each transcribed segment, find the speaker that overlaps most with its time range."""
    out = []
    for ts in text_segments:
        ts_start, ts_end = float(ts["start"]), float(ts["end"])
        best_spk_seg = None
        best_overlap = 0.0

        for ss in speaker_segments:
            ss_start, ss_end = ss["start"], ss["end"]
            overlap = max(0.0, min(ts_end, ss_end) - max(ts_start, ss_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk_seg = ss

        seg = dict(ts)
        if best_spk_seg:
            seg["speaker"] = best_spk_seg["speaker"]
            seg["identified_speaker"] = best_spk_seg.get("identified_speaker", LABEL_UNKNOWN)
            seg["display_speaker"] = best_spk_seg.get("display_speaker", LABEL_UNKNOWN)
            if "agent_name" in best_spk_seg:
                seg["agent_name"] = best_spk_seg["agent_name"]
            if "agent_slug" in best_spk_seg:
                seg["agent_slug"] = best_spk_seg["agent_slug"]
        else:
            seg["speaker"] = "SPEAKER_99"
            seg["identified_speaker"] = LABEL_UNKNOWN
            seg["display_speaker"] = LABEL_UNKNOWN
        out.append(seg)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI for testing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("--backend", default="auto", choices=["auto", "sortformer", "pyannote"])
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--streaming-sortformer", action="store_true",
                        help="use streaming Sortformer instead of the full GPU model")
    parser.add_argument("--gt", help="Optional ground truth JSON for accuracy comparison")
    parser.add_argument("--target-agent-slug", help="limit voice matching to one expected agent slug")
    parser.add_argument("--presence-floor", type=float, default=AGENT_PRESENCE_FLOOR,
                        help="minimum cosine required to assert the target agent is present")
    parser.add_argument("--gt-auto-offset", action="store_true",
                        help="shift GT segment times by detected leading speech offset")
    parser.add_argument("--out", help="write full JSON result to this path")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    transcribed = None
    gt_offset = 0.0
    if args.gt:
        with open(args.gt) as f:
            gt_data = json.load(f)
        if args.gt_auto_offset:
            gt_offset = detect_leading_speech_offset(args.audio)
        transcribed = [
            {
                "start": float(s["start"]) + gt_offset,
                "end": float(s["end"]) + gt_offset,
                "text": s["text"],
            }
            for s in gt_data.get("segments", [])
        ]

    result = diarize_clean(
        audio_path=args.audio,
        transcribed_segments=transcribed,
        backend=args.backend,
        max_speakers=args.max_speakers,
        sortformer_streaming=args.streaming_sortformer,
        target_agent_slug=args.target_agent_slug,
        presence_floor=args.presence_floor,
        hf_token=os.environ.get("HF_TOKEN"),
    )

    print(f"\n=== DIARIZATION RESULT ===")
    print(f"Backend: {result['backend']}")
    print(f"Speaker count: {result['speaker_count']}")
    print(f"Agent: {result['agent_name']} (speaker_id={result['agent_speaker_id']}, sim={result['agent_similarity']:.3f})")
    if args.gt_auto_offset:
        print(f"GT timestamp offset: +{gt_offset:.3f}s")
    print(f"\nCluster segment counts: {result['cluster_segment_counts']}")
    print(f"\nMatch table (cluster vs enrolled agents, top-3 only):")
    for spk, matches in result["cluster_match_table"].items():
        sorted_matches = sorted(matches.items(), key=lambda x: -x[1]["similarity"])[:3]
        line = f"  {spk}: " + ", ".join(f"{m[1]['name']}={m[1]['similarity']:.3f}" for m in sorted_matches)
        print(line)

    if args.gt:
        # Compute accuracy
        correct = 0
        agent_correct = 0; agent_total = 0
        customer_correct = 0; customer_total = 0
        for ts, gt in zip(result["segments"], gt_data["segments"]):
            gt_role = LABEL_AGENT if gt["speaker"] == "agent" else LABEL_CUSTOMER
            pred_normalized = ts["identified_speaker"]
            if pred_normalized == gt_role:
                correct += 1
                if gt_role == LABEL_AGENT: agent_correct += 1
                else: customer_correct += 1
            if gt_role == LABEL_AGENT: agent_total += 1
            else: customer_total += 1

        n = len(result["segments"])
        print(f"\n=== ACCURACY vs GROUND TRUTH ===")
        print(f"Overall: {correct}/{n} = {correct/n*100:.1f}%")
        print(f"AGENT:   {agent_correct}/{agent_total} = {agent_correct/max(1,agent_total)*100:.1f}%")
        print(f"CUSTOMER:{customer_correct}/{customer_total} = {customer_correct/max(1,customer_total)*100:.1f}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {args.out}")
