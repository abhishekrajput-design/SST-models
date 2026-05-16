"""
Clean speaker-first diarization pipeline.

Architecture:
  Stage 1 — Pure diarization: detect distinct speakers (SPEAKER_00, SPEAKER_01, ...)
            using NeMo Sortformer (best for 2-4 speakers, phone audio) with
            pyannote fallback for >4 speakers.
  Stage 2 — Voiceprint matching: compute per-cluster centroids, match against
            enrolled voiceprints, and assign any strong enrolled-agent cluster.
  Stage 3 — Role labeling: matched enrolled agents → AGENT with their real name,
            all unmatched speakers → CUSTOMER.

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

# A single long diarization turn is enough evidence to build a voiceprint
# centroid. Short one-off noises still need multiple segments.
MIN_CLUSTER_SECONDS = 1.5

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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _speaker_embedding_ranges(start: float, end: float) -> List[Tuple[float, float]]:
    """Split long diarization turns before embedding so one long turn is not skipped."""
    dur = max(float(end) - float(start), 0.0)
    if dur <= 0:
        return []
    min_window_s = max(_env_float("SST_CLUSTER_EMBED_MIN_WINDOW_SECONDS", 1.0), 0.5)
    window_s = max(_env_float("SST_CLUSTER_EMBED_WINDOW_SECONDS", 6.0), min_window_s)
    stride_s = max(_env_float("SST_CLUSTER_EMBED_STRIDE_SECONDS", 4.0), 0.5)
    if dur <= window_s * 1.35:
        return [(start, end)] if dur >= min_window_s else []

    ranges: List[Tuple[float, float]] = []
    cursor = start
    while cursor < end:
        right = min(cursor + window_s, end)
        if right - cursor >= min_window_s:
            ranges.append((cursor, right))
        if right >= end:
            break
        cursor += stride_s
    if not ranges and dur >= min_window_s:
        ranges.append((start, end))
    return ranges


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
    speaker_seconds: Dict[str, float] = {}
    for seg in speaker_segments:
        spk = str(seg["speaker"])
        start = float(seg["start"])
        end = float(seg["end"])
        speaker_seconds[spk] = speaker_seconds.get(spk, 0.0) + max(end - start, 0.0)
        for left, right in _speaker_embedding_ranges(start, end):
            emb = _embed_segment(audio, sr, left, right, embedder)
            if emb is not None:
                by_speaker.setdefault(spk, []).append(emb)

    centroids = {}
    counts = {}
    for spk, embs in by_speaker.items():
        if len(embs) < min_segments and speaker_seconds.get(spk, 0.0) < MIN_CLUSTER_SECONDS:
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
    best_target = (None, None, 0.0)
    skipped_dim = 0
    if target_agent_slug and target_agent_slug not in enrolled_voiceprints:
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
            if target_agent_slug and slug == target_agent_slug and best_for_agent > best_target[2]:
                best_target = (spk, slug, best_for_agent)

        match_table[spk] = cluster_match

    if skipped_dim > 0:
        logger.info(f"Skipped {skipped_dim} dim-mismatched voiceprints during matching")

    agent_spk, agent_slug, best_sim = best_target if target_agent_slug else best_overall
    if best_sim < presence_floor:
        return None, None, best_sim, match_table
    return agent_spk, agent_slug, best_sim, match_table


def _load_agent_match_hints(
    presence_floor: float,
    agents_index_path: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Read per-agent similarity/margin gates saved during clean enrollment."""
    index_path = agents_index_path or str(
        Path(__file__).resolve().parent.parent / "data" / "agent_voiceprints" / "agents.json"
    )
    thresholds: Dict[str, float] = {}
    margins: Dict[str, float] = {}
    default_margin = _env_float("SST_MULTI_AGENT_CLUSTER_MIN_MARGIN", 0.08)
    outside_margin = _env_float("SST_MULTI_AGENT_OUTSIDE_MARGIN", 0.06)

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            agents = json.load(f)
    except Exception:
        return thresholds, margins

    for slug, info in (agents or {}).items():
        if not isinstance(info, dict):
            continue
        min_sims: List[float] = []
        min_margins: List[float] = []
        for entry in info.get("voiceprints") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("segment_role_min_similarity") is not None:
                try:
                    min_sims.append(float(entry["segment_role_min_similarity"]))
                except (TypeError, ValueError):
                    pass
            if entry.get("segment_role_min_margin") is not None:
                try:
                    min_margins.append(float(entry["segment_role_min_margin"]))
                except (TypeError, ValueError):
                    pass

        if min_sims:
            thresholds[slug] = max(float(presence_floor), min(min_sims))
        elif info.get("max_outside_sim") is not None:
            try:
                thresholds[slug] = max(float(presence_floor), float(info["max_outside_sim"]) + outside_margin)
            except (TypeError, ValueError):
                thresholds[slug] = float(presence_floor)
        else:
            thresholds[slug] = float(presence_floor)

        margins[slug] = max(default_margin, min(min_margins) if min_margins else default_margin)
    return thresholds, margins


def select_multi_agent_cluster_matches(
    match_table: Dict[str, Dict],
    speaker_durations: Dict[str, float],
    enrolled_voiceprints: Dict[str, Tuple[str, np.ndarray]],
    presence_floor: float = AGENT_PRESENCE_FLOOR,
    agents_index_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """Return {speaker_id: agent match} for every cluster with strong voiceprint evidence."""
    if not _env_bool("SST_MULTI_AGENT_CLUSTER_MATCH", True):
        return {}

    min_seconds = _env_float("SST_MULTI_AGENT_CLUSTER_MIN_SECONDS", 2.0)
    thresholds, margins = _load_agent_match_hints(presence_floor, agents_index_path)
    selected: Dict[str, Dict] = {}

    for spk, matches in (match_table or {}).items():
        if float(speaker_durations.get(spk, 0.0)) < min_seconds:
            continue
        scored = []
        for slug, row in (matches or {}).items():
            if slug not in enrolled_voiceprints or not isinstance(row, dict):
                continue
            try:
                scored.append((slug, float(row.get("similarity") or 0.0), row.get("name") or enrolled_voiceprints[slug][0]))
            except (TypeError, ValueError):
                continue
        if not scored:
            continue
        scored.sort(key=lambda item: item[1], reverse=True)
        top_slug, top_sim, top_name = scored[0]
        second_sim = scored[1][1] if len(scored) > 1 else 0.0
        required_sim = max(float(presence_floor), float(thresholds.get(top_slug, presence_floor)))
        required_margin = float(margins.get(top_slug, _env_float("SST_MULTI_AGENT_CLUSTER_MIN_MARGIN", 0.08)))
        margin = top_sim - second_sim
        if top_sim >= required_sim and margin >= required_margin:
            selected[spk] = {
                "agent_slug": top_slug,
                "agent_name": top_name,
                "similarity": top_sim,
                "margin": margin,
                "second_similarity": second_sim,
                "required_similarity": required_sim,
                "required_margin": required_margin,
                "speaker_seconds": float(speaker_durations.get(spk, 0.0)),
            }
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ROLE LABELING (no hardcoded "AGENT"/"CUSTOMER" strings — use constants)
# ──────────────────────────────────────────────────────────────────────────────

def assign_roles(
    speaker_segments: List[Dict],
    agent_speaker_id: Optional[str],
    agent_slug: Optional[str],
    agent_name: Optional[str],
    agent_cluster_matches: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """
    Apply role labels in-place based on matched SPEAKER_NN agent clusters.

    Matched agent clusters get role=AGENT + agent_name.
    Other speakers become CUSTOMER (single) or CUSTOMER_1/2/... (multiple).
    """
    agent_cluster_matches = agent_cluster_matches or {}

    # Collect non-agent speaker IDs in arrival order
    non_agent_speakers = []
    for s in speaker_segments:
        is_agent = (
            s["speaker"] in agent_cluster_matches
            or (not agent_cluster_matches and s["speaker"] == agent_speaker_id and agent_speaker_id is not None)
        )
        if not is_agent and s["speaker"] not in non_agent_speakers:
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
        cluster_match = agent_cluster_matches.get(s["speaker"])
        is_agent = cluster_match is not None or (
            not agent_cluster_matches and s["speaker"] == agent_speaker_id and agent_speaker_id is not None
        )
        if is_agent:
            match_slug = (cluster_match or {}).get("agent_slug") or agent_slug
            match_name = (cluster_match or {}).get("agent_name") or agent_name
            seg["identified_speaker"] = LABEL_AGENT
            if match_name:
                seg["agent_name"] = match_name
            if match_slug:
                seg["agent_slug"] = match_slug
            if cluster_match:
                seg["agent_match_similarity"] = round(float(cluster_match.get("similarity") or 0.0), 4)
                seg["agent_match_margin"] = round(float(cluster_match.get("margin") or 0.0), 4)
            seg["display_speaker"] = match_name or LABEL_AGENT
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
    speaker_durations: Dict[str, float] = {}
    for seg in speaker_segments:
        speaker_durations[seg["speaker"]] = speaker_durations.get(seg["speaker"], 0.0) + max(
            float(seg["end"]) - float(seg["start"]),
            0.0,
        )
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
    audio_duration_s = len(audio) / float(sr or 16000)
    min_agent_seconds = float(os.getenv("SST_AGENT_CLUSTER_MIN_SECONDS", "6.0") or "6.0")
    min_agent_ratio = float(os.getenv("SST_AGENT_CLUSTER_MIN_AUDIO_RATIO", "0.0") or "0.0")
    required_agent_seconds = max(min_agent_seconds, audio_duration_s * min_agent_ratio)
    agent_decision = {
        "selected_speaker": agent_spk,
        "selected_slug": agent_slug,
        "similarity": float(agent_sim),
        "required_seconds": round(float(required_agent_seconds), 3),
        "selected_seconds": round(float(speaker_durations.get(agent_spk or "", 0.0)), 3),
        "rejected": False,
        "reason": "",
    }
    if agent_spk is not None and speaker_durations.get(agent_spk, 0.0) < required_agent_seconds:
        agent_decision["rejected"] = True
        agent_decision["reason"] = "agent_cluster_too_short_for_desk_recording"
        logger.info(
            "Rejecting agent cluster %s: %.2fs < required %.2fs",
            agent_spk,
            speaker_durations.get(agent_spk, 0.0),
            required_agent_seconds,
        )
        agent_spk = None
        agent_slug = None
    agent_name = voiceprints[agent_slug][0] if agent_slug else None

    logger.info(
        f"Agent matching: speaker={agent_spk}, agent={agent_name} "
        f"({agent_slug}), similarity={agent_sim:.3f}"
    )

    # ── Stage 3: Apply role labels ──
    multi_agent_matches: Dict[str, Dict] = {}
    if target_agent_slug is None:
        multi_agent_matches = select_multi_agent_cluster_matches(
            match_table,
            speaker_durations,
            voiceprints,
            presence_floor=presence_floor,
        )
        if multi_agent_matches:
            logger.info(
                "Multi-agent cluster matches: %s",
                {
                    spk: {
                        "agent": row.get("agent_slug"),
                        "similarity": round(float(row.get("similarity") or 0.0), 3),
                        "margin": round(float(row.get("margin") or 0.0), 3),
                    }
                    for spk, row in multi_agent_matches.items()
                },
            )

    labeled_speaker_segments = assign_roles(
        speaker_segments,
        agent_spk,
        agent_slug,
        agent_name,
        agent_cluster_matches=multi_agent_matches or None,
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
        "cluster_durations": {k: round(float(v), 3) for k, v in speaker_durations.items()},
        "agent_cluster_decision": agent_decision,
        "multi_agent_cluster_matches": multi_agent_matches,
        "identified_agents": [
            {
                "speaker": spk,
                "agent_slug": row.get("agent_slug"),
                "agent_name": row.get("agent_name"),
                "similarity": round(float(row.get("similarity") or 0.0), 4),
                "margin": round(float(row.get("margin") or 0.0), 4),
                "seconds": round(float(row.get("speaker_seconds") or 0.0), 3),
            }
            for spk, row in sorted(multi_agent_matches.items())
        ],
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
    """Assign text segments to diarized speakers with padded, strict role overlap.

    Desk recordings often have word-edge drift and far-field/background speech.
    Padding helps early/late words attach to the right nearby speaker, while the
    agent-specific overlap checks stop a weak background agent cluster from
    claiming a full customer utterance.
    """
    pad_s = float(os.getenv("SST_ROLE_OVERLAY_PAD_S", "0.35") or "0.35")
    agent_min_ratio = float(os.getenv("SST_AGENT_OVERLAY_MIN_RATIO", "0.30") or "0.30")
    agent_min_seconds = float(os.getenv("SST_AGENT_OVERLAY_MIN_SECONDS", "0.35") or "0.35")
    agent_margin_ratio = float(os.getenv("SST_AGENT_OVERLAY_MARGIN_RATIO", "1.25") or "1.25")
    agent_margin_seconds = float(os.getenv("SST_AGENT_OVERLAY_MARGIN_SECONDS", "0.15") or "0.15")

    out = []
    for ts in text_segments:
        ts_start, ts_end = float(ts["start"]), float(ts["end"])
        ts_dur = max(ts_end - ts_start, 0.001)
        speaker_scores: Dict[str, float] = {}
        role_scores: Dict[str, float] = {}
        speaker_best_seg: Dict[str, Dict] = {}
        speaker_best_overlap: Dict[str, float] = {}

        for ss in speaker_segments:
            ss_start = max(0.0, float(ss["start"]) - pad_s)
            ss_end = float(ss["end"]) + pad_s
            overlap = max(0.0, min(ts_end, ss_end) - max(ts_start, ss_start))
            if overlap <= 0:
                continue
            spk = str(ss.get("speaker") or LABEL_UNKNOWN)
            role = str(ss.get("identified_speaker") or LABEL_UNKNOWN)
            speaker_scores[spk] = speaker_scores.get(spk, 0.0) + overlap
            role_scores[role] = role_scores.get(role, 0.0) + overlap
            if overlap > speaker_best_overlap.get(spk, 0.0):
                speaker_best_overlap[spk] = overlap
                speaker_best_seg[spk] = ss

        seg = dict(ts)
        if speaker_scores:
            best_spk, best_overlap = max(speaker_scores.items(), key=lambda item: item[1])
            best_spk_seg = speaker_best_seg[best_spk]
            chosen_role = best_spk_seg.get("identified_speaker", LABEL_UNKNOWN)

            if chosen_role == LABEL_AGENT:
                agent_overlap = role_scores.get(LABEL_AGENT, 0.0)
                customer_overlap = role_scores.get(LABEL_CUSTOMER, 0.0)
                enough_agent_overlap = (
                    agent_overlap >= min(agent_min_seconds, max(ts_dur * 0.60, 0.05))
                    and (agent_overlap / ts_dur) >= agent_min_ratio
                )
                agent_dominates = agent_overlap >= (customer_overlap * agent_margin_ratio + agent_margin_seconds)
                if not (enough_agent_overlap and agent_dominates):
                    customer_speakers = [
                        (spk, score)
                        for spk, score in speaker_scores.items()
                        if (speaker_best_seg.get(spk, {}).get("identified_speaker") == LABEL_CUSTOMER)
                    ]
                    if customer_speakers:
                        best_spk, best_overlap = max(customer_speakers, key=lambda item: item[1])
                        best_spk_seg = speaker_best_seg[best_spk]
                        chosen_role = LABEL_CUSTOMER
                        seg["agent_overlap_rejected"] = True
                    else:
                        best_spk_seg = None
                        chosen_role = LABEL_UNKNOWN
                        seg["agent_overlap_rejected"] = True

            if best_spk_seg:
                seg["speaker"] = best_spk_seg["speaker"]
                seg["identified_speaker"] = chosen_role
                seg["display_speaker"] = best_spk_seg.get("display_speaker", LABEL_UNKNOWN)
                if chosen_role == LABEL_AGENT:
                    if "agent_name" in best_spk_seg:
                        seg["agent_name"] = best_spk_seg["agent_name"]
                    if "agent_slug" in best_spk_seg:
                        seg["agent_slug"] = best_spk_seg["agent_slug"]
                else:
                    seg.pop("agent_name", None)
                    seg.pop("agent_slug", None)
                seg["role_overlap"] = {
                    "agent": round(float(role_scores.get(LABEL_AGENT, 0.0)), 3),
                    "customer": round(float(role_scores.get(LABEL_CUSTOMER, 0.0)), 3),
                    "padding_s": round(float(pad_s), 3),
                }
                seg["role_overlap_ratio"] = round(float(best_overlap / ts_dur), 3)
                seg["role_assignment_reason"] = (
                    "padded_strict_overlap"
                    if not seg.get("agent_overlap_rejected")
                    else "background_agent_overlap_rejected"
                )
            else:
                seg["speaker"] = "SPEAKER_99"
                seg["identified_speaker"] = LABEL_UNKNOWN
                seg["display_speaker"] = LABEL_UNKNOWN
                seg["role_overlap"] = {
                    "agent": round(float(role_scores.get(LABEL_AGENT, 0.0)), 3),
                    "customer": round(float(role_scores.get(LABEL_CUSTOMER, 0.0)), 3),
                    "padding_s": round(float(pad_s), 3),
                }
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
