"""
Voiceprint-first multi-speaker diarization.

For each transcript segment, this module compares the segment voice embedding
against enrolled agent voiceprints and labels the chosen agent as AGENT. Any
non-agent speech is grouped into customer speakers for UI display.

Both supported enrollment backends are handled:
  - ECAPA voiceprints: 192 dimensions
  - CAM++ voiceprints: 512 dimensions
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from src.voiceprints import resolve_voiceprint_path

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MIN_SEG_S_FOR_EMB = 0.3
PER_SEG_THRESHOLD = 0.30           # default / floor
PER_AGENT_MARGIN  = 0.05            # added on top of max_outside_sim per-agent
PER_AGENT_THRESH_CAP = 0.42         # tightened: 0.55 was too strict, dropped many true-agent segs
AGENT_MIN_MATCHED = 3
MAX_CUSTOMER_CLUSTERS = 3
MIN_CONF_DUR = 1.0
MERGE_TO_AGENT_SIM = 0.28
SHORT_REPLY_MAX_DUR = 0.95
SHORT_REPLY_MAX_SIM = 0.30
FAREWELL_AGENT_MIN_SIM = 0.18
CLUSTER_FIRST_MIN_DUR = 60.0           # was 180 — short phone calls benefit too
CLUSTER_FIRST_MIN_SEGMENTS = 15        # was 30
CLUSTER_FIRST_AGENT_RATIO = 0.55       # was 0.68 — allow more even agent/customer splits

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_AGENTS_INDEX = os.path.join(_DATA_DIR, "agent_voiceprints", "agents.json")


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _norm_text(text: str) -> str:
    return " ".join(_norm_words(text))


def _looks_like_question(text: str) -> bool:
    words = _norm_words(text)
    if "?" in (text or ""):
        return True
    if not words:
        return False
    return words[0] in {
        "are",
        "can",
        "could",
        "do",
        "does",
        "did",
        "is",
        "what",
        "when",
        "where",
        "who",
        "why",
        "will",
        "would",
    }


def _short_customer_reply(text: str) -> bool:
    words = _norm_words(text)
    if not words or len(words) > 3:
        return False
    norm = " ".join(words)
    if norm in {
        "yeah",
        "yes",
        "yep",
        "ok",
        "okay",
        "sure",
        "alright",
        "all right",
        "speaking yes",
    }:
        return True
    # Short one-word answers to an agent question are often customer place/name
    # replies that can be pulled into the agent cluster by voiceprint similarity.
    return len(words) == 1 and len(words[0]) <= 16


def _agent_phrase(text: str) -> bool:
    norm = _norm_text(text)
    return norm in {
        "okay okay",
        "okay perfect",
        "perfect",
        "cheers",
        "thank you",
        "what are your plans",
    }


def _farewell(text: str) -> bool:
    norm = _norm_text(text).replace("bye bye", "bye-bye")
    return norm in {"bye", "bye-bye", "goodbye"}


def _load_voiceprints(path: Optional[str] = None) -> Dict[str, Tuple[str, np.ndarray]]:
    """Returns ``{slug: (display_name, stack)}`` where ``stack`` has shape
    ``(N, dim)``. Prefers the multi-VP ``voiceprints`` list; falls back to the
    legacy single ``voiceprint_path``. Centroids with mismatched dim within
    one agent are dropped (the dominant dim wins).
    """
    p = path or _AGENTS_INDEX
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            agents = json.load(f)
    except Exception as e:
        logger.warning("agents.json read failed: %s", e)
        return {}

    out: Dict[str, Tuple[str, np.ndarray]] = {}
    for slug, info in agents.items():
        if not isinstance(info, dict):
            continue

        raw_paths: List[str] = []
        vps_field = info.get("voiceprints")
        if isinstance(vps_field, list):
            for entry in vps_field:
                pp = entry.get("path") if isinstance(entry, dict) else entry
                if pp:
                    raw_paths.append(pp)
        if not raw_paths:
            legacy = info.get("voiceprint_path") or info.get("voiceprint")
            if legacy:
                raw_paths.append(legacy)
        if not raw_paths:
            continue

        loaded: List[np.ndarray] = []
        for raw in raw_paths:
            vp_path = resolve_voiceprint_path(raw, p)
            if not os.path.isfile(vp_path):
                continue
            try:
                vp = np.load(vp_path).astype(np.float32).squeeze()
            except Exception as e:
                logger.warning("Skipping %s: failed to load voiceprint (%s)", slug, e)
                continue
            if vp.ndim != 1:
                logger.warning("Skipping %s entry: invalid shape %s", slug, vp.shape)
                continue
            n = np.linalg.norm(vp)
            if n > 0:
                vp = vp / n
            loaded.append(vp)
        if not loaded:
            continue

        dims = {v.shape[0] for v in loaded}
        if len(dims) > 1:
            counts = {d: sum(1 for v in loaded if v.shape[0] == d) for d in dims}
            best_dim = max(counts, key=counts.get)
            loaded = [v for v in loaded if v.shape[0] == best_dim]

        stack = np.stack(loaded).astype(np.float32)
        name = info.get("agent_name") or info.get("name") or slug
        out[slug] = (name, stack)
    return out


def _per_agent_thresholds(
    global_thresh: float,
    index_path: Optional[str] = None,
) -> Dict[str, float]:
    """Per-agent threshold = max(global, max_outside_sim + PER_AGENT_MARGIN), capped.

    Lets high-purity enrolments (low max_outside_sim) keep the loose global
    threshold while forcing fuzzy enrolments (e.g. Amandeep max_outside=0.669,
    inside=0.65) to use a stricter cutoff so customer audio doesn't slip in.
    """
    p = index_path or _AGENTS_INDEX
    out: Dict[str, float] = {}
    if not os.path.isfile(p):
        return out
    try:
        with open(p, encoding="utf-8") as f:
            agents = json.load(f)
    except Exception:
        return out
    for slug, info in agents.items():
        if not isinstance(info, dict):
            continue
        max_out = info.get("max_outside_sim")
        try:
            t = max(global_thresh, float(max_out) + PER_AGENT_MARGIN) if max_out is not None else global_thresh
        except (TypeError, ValueError):
            t = global_thresh
        out[slug] = float(min(t, PER_AGENT_THRESH_CAP))
    return out


def _speech_ratio(chunk: np.ndarray, sr: int) -> float:
    """Fraction of 25ms frames whose RMS exceeds a noise floor.

    Cheap voice-activity surrogate: speech-dominant chunks score >0.5,
    silence/noise <0.3. Used to reject embedding inputs that won't yield
    a stable speaker vector.
    """
    if chunk.size < int(sr * 0.05):
        return 0.0
    win = int(sr * 0.025)
    n = chunk.size // win
    if n < 4:
        return 0.0
    frames = chunk[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    # Adaptive floor — 1.5x the median frame RMS of this chunk.
    floor = float(np.median(rms)) * 1.5
    floor = max(floor, 0.005)
    return float((rms > floor).mean())


def _seg_embeddings(
    segments: List[dict],
    audio: np.ndarray,
    sr: int,
    model,
    device: str,
) -> Tuple[List[Optional[np.ndarray]], np.ndarray]:
    """Extract ECAPA embedding for each segment.

    For short segments (<1.5s) we widen the audio window symmetrically up to
    1.5s total. We also reject chunks dominated by silence/noise (speech ratio
    <0.30) — those produce unstable embeddings that mis-match voiceprints.
    """
    from src.speaker_role import _embed

    # Conservative widening: only pad segments <0.8s, and only by ±0.2s.
    # Wider padding pulled in surrounding speaker audio and polluted short
    # customer back-channels with neighbouring agent voice → false AGENT match.
    SHORT_THRESH = int(sr * 0.8)
    PAD_EACH = int(sr * 0.2)
    embs: List[Optional[np.ndarray]] = []
    for seg in segments:
        s = int(float(seg["start"]) * sr)
        e = min(int(float(seg["end"]) * sr), len(audio))
        if e - s < int(sr * MIN_SEG_S_FOR_EMB):
            embs.append(None)
            continue
        if e - s < SHORT_THRESH:
            s2 = max(0, s - PAD_EACH)
            e2 = min(len(audio), e + PAD_EACH)
            chunk = audio[s2:e2]
        else:
            chunk = audio[s:e]
        if _speech_ratio(chunk, sr) < 0.25:
            embs.append(None)
            continue
        embs.append(_embed(model, chunk, device))
    mask = np.array([e is not None for e in embs], dtype=bool)
    return embs, mask


def _cluster_customers(X: np.ndarray, max_k: int = MAX_CUSTOMER_CLUSTERS) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return np.zeros(len(X), dtype=int)

    if len(X) < 6:
        return np.zeros(len(X), dtype=int)

    best_labels = np.zeros(len(X), dtype=int)
    best_score = -1.0
    for k in range(2, min(max_k, max(1, len(X) // 4)) + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, metric="cosine")
        except Exception:
            continue
        logger.info("customer cluster k=%d silhouette=%.3f", k, score)
        if score > best_score + 0.05:
            best_score = score
            best_labels = labels
    return best_labels


def _cluster_speaker_roles(
    segments: List[dict],
    embs: List[Optional[np.ndarray]],
    sims: np.ndarray,
    j_agent: int,
    agent_slug: str,
    agent_name: str,
) -> Tuple[bool, Dict[str, object]]:
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return False, {"reason": "sklearn unavailable"}

    valid_idxs = [i for i, emb in enumerate(embs) if emb is not None]
    if len(valid_idxs) < CLUSTER_FIRST_MIN_SEGMENTS:
        return False, {"reason": "not enough embeddable segments"}

    X = np.stack([embs[i] for i in valid_idxs]).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    labels = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(X)
    valid_sims = np.array(
        [
            float(sims[i, j_agent]) if j_agent >= 0 else 0.0
            for i in valid_idxs
        ],
        dtype=np.float32,
    )

    cluster_stats: Dict[int, Dict[str, float]] = {}
    for pos, seg_i in enumerate(valid_idxs):
        cid = int(labels[pos])
        dur = max(
            float(segments[seg_i]["end"]) - float(segments[seg_i]["start"]),
            0.0,
        )
        stat = cluster_stats.setdefault(
            cid,
            {
                "count": 0,
                "seconds": 0.0,
                "sim_sum": 0.0,
                "first_start": float(segments[seg_i]["start"]),
            },
        )
        stat["count"] += 1
        stat["seconds"] += dur
        stat["sim_sum"] += float(valid_sims[pos])
        stat["first_start"] = min(stat["first_start"], float(segments[seg_i]["start"]))

    for stat in cluster_stats.values():
        stat["mean_sim"] = stat["sim_sum"] / max(stat["count"], 1)

    agent_cluster = max(cluster_stats, key=lambda c: cluster_stats[c]["mean_sim"])
    customer_clusters = sorted(
        (c for c in cluster_stats if c != agent_cluster),
        key=lambda c: cluster_stats[c]["first_start"],
    )
    customer_name = {cid: f"Customer {i + 1}" for i, cid in enumerate(customer_clusters)}
    customer_speaker = {cid: f"SPEAKER_{i + 1:02d}" for i, cid in enumerate(customer_clusters)}

    idx_to_label = {seg_i: int(labels[pos]) for pos, seg_i in enumerate(valid_idxs)}
    valid_mids = [
        (float(segments[i]["start"]) + float(segments[i]["end"])) / 2.0
        for i in valid_idxs
    ]

    # Cluster centroids — used to reconcile per-segment label with cosine sim.
    # A segment that lands in the agent cluster but has sim < CLUSTER_AGENT_FLOOR
    # to the agent voiceprint AND is closer to the customer centroid in
    # embedding space gets re-assigned to customer.
    CLUSTER_AGENT_FLOOR = 0.20
    centroids: Dict[int, np.ndarray] = {}
    for cid in cluster_stats:
        cluster_X = X[labels == cid]
        c = cluster_X.mean(axis=0)
        n = np.linalg.norm(c)
        centroids[cid] = c / n if n > 0 else c
    customer_cids = list(customer_clusters)

    agent_count = 0
    agent_sims: List[float] = []
    reassigned = 0
    for i, seg in enumerate(segments):
        if i in idx_to_label:
            cid = idx_to_label[i]
        else:
            seg_mid = (float(seg["start"]) + float(seg["end"])) / 2.0
            nearest_pos = int(np.argmin([abs(seg_mid - mid) for mid in valid_mids]))
            cid = idx_to_label[valid_idxs[nearest_pos]]

        sim = float(sims[i, j_agent]) if j_agent >= 0 and i < len(sims) else 0.0
        seg["_best_sim"] = sim
        seg["_best_match"] = agent_slug

        # Reconcile: if cluster says agent but sim is too low and there's a
        # better customer cluster match, demote.
        if cid == agent_cluster and sim < CLUSTER_AGENT_FLOOR and embs[i] is not None and customer_cids:
            emb = embs[i].astype(np.float32)
            n_emb = np.linalg.norm(emb)
            if n_emb > 0:
                emb = emb / n_emb
                agent_cent_sim = float(emb @ centroids[agent_cluster])
                cust_cent_sims = {c: float(emb @ centroids[c]) for c in customer_cids}
                best_cust = max(cust_cent_sims, key=cust_cent_sims.get)
                if cust_cent_sims[best_cust] > agent_cent_sim + 0.05:
                    cid = best_cust
                    reassigned += 1

        # Reverse reconcile: cluster says CUSTOMER but sim is high.
        #   sim >= 0.50 → unconditional promote to AGENT (very high voiceprint match).
        #   0.42 <= sim < 0.50 → promote only if embedding closer to agent than to customer cluster.
        promoted_to_agent = False
        if cid != agent_cluster and sim >= 0.50:
            cid = agent_cluster
            promoted_to_agent = True
        elif cid != agent_cluster and sim >= 0.42 and embs[i] is not None:
            emb = embs[i].astype(np.float32)
            n_emb = np.linalg.norm(emb)
            if n_emb > 0 and cid in centroids:
                emb_n = emb / n_emb
                cust_cent_sim = float(emb_n @ centroids[cid])
                if sim > cust_cent_sim:
                    cid = agent_cluster
                    promoted_to_agent = True

        if cid == agent_cluster:
            seg["speaker"] = "SPEAKER_00"
            seg["identified_speaker"] = "AGENT"
            seg["agent_name"] = agent_name
            seg["display_speaker"] = agent_name
            agent_count += 1
            agent_sims.append(sim)
        else:
            seg["speaker"] = customer_speaker.get(cid, "SPEAKER_01")
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = customer_name.get(cid, "Customer 1")
            seg.pop("agent_name", None)
    if reassigned:
        logger.info("cluster_first: reassigned %d weak agent-cluster segs to customer", reassigned)

    return True, {
        "agent_cluster": int(agent_cluster),
        "cluster_stats": {
            str(cid): {
                "count": int(stat["count"]),
                "seconds": round(float(stat["seconds"]), 3),
                "mean_sim": round(float(stat["mean_sim"]), 4),
            }
            for cid, stat in cluster_stats.items()
        },
        "agent_count": agent_count,
        "agent_sims": agent_sims,
    }


def _unknown_result(segments: List[dict], reason: str) -> Dict[str, object]:
    logger.warning("diarize_multi: %s; marking speaker ID as unknown", reason)
    for seg in segments:
        seg["speaker"] = seg.get("speaker") or "SPEAKER_99"
        seg["identified_speaker"] = "CUSTOMER"
        seg.pop("agent_name", None)
        seg["display_speaker"] = "Unknown"
        seg["_best_sim"] = 0.0
        seg["_best_match"] = ""

    total_seconds = sum(
        max(float(s.get("end", 0)) - float(s.get("start", 0)), 0.0)
        for s in segments
    )
    return {
        "segments": segments,
        "agent_slug": "",
        "agent_name": "Unknown Agent",
        "agent_similarity": 0.0,
        "n_speakers": 1 if segments else 0,
        "match_counts": {},
        "other_agent_count": {},
        "per_speaker": {
            "Unknown": {"turns": len(segments), "seconds": total_seconds}
        },
        "matched_backend_dim": None,
        "voiceprint_dims": {},
        "warning": reason,
    }


def _group_voiceprints_by_dim(
    voiceprints: Dict[str, Tuple[str, np.ndarray]],
) -> Dict[int, Dict[str, Tuple[str, np.ndarray]]]:
    grouped: Dict[int, Dict[str, Tuple[str, np.ndarray]]] = {}
    for slug, value in voiceprints.items():
        name, stack = value
        if stack.ndim == 1:
            stack = stack[None, :]
        dim = int(stack.shape[1])
        if dim not in (192, 512):
            logger.warning("Skipping %s: unsupported voiceprint dim=%s", slug, dim)
            continue
        grouped.setdefault(dim, {})[slug] = (name, stack)
    return grouped


def _segment_embeddings_for_dim(
    dim: int,
    segments: List[dict],
    audio: np.ndarray,
    sr: int,
    force_cpu: bool,
) -> Tuple[List[Optional[np.ndarray]], np.ndarray]:
    embs: List[Optional[np.ndarray]] = []
    if dim == 512:
        from src.embedding_campp import EmbeddingModel

        model = EmbeddingModel()
        try:
            model.load(force_cpu=force_cpu)
            if model.dim != dim:
                logger.warning(
                    "CAM++ matching unavailable: backend=%s dim=%s expected=%s",
                    model.model_name,
                    model.dim,
                    dim,
                )
                embs = [None] * len(segments)
            else:
                SHORT_THRESH = int(sr * 0.8)
                PAD_EACH = int(sr * 0.2)
                for seg in segments:
                    s = int(float(seg["start"]) * sr)
                    e = min(int(float(seg["end"]) * sr), len(audio))
                    if e - s < int(sr * MIN_SEG_S_FOR_EMB):
                        embs.append(None)
                        continue
                    if e - s < SHORT_THRESH:
                        s2 = max(0, s - PAD_EACH)
                        e2 = min(len(audio), e + PAD_EACH)
                        chunk = audio[s2:e2]
                    else:
                        chunk = audio[s:e]
                    if _speech_ratio(chunk, sr) < 0.25:
                        embs.append(None)
                        continue
                    embs.append(model.embed_chunk(chunk, sr))
        finally:
            model.unload()
    elif dim == 192:
        from src.speaker_role import _free, _load_ecapa

        model, device = _load_ecapa(force_cpu=force_cpu)
        try:
            embs, _ = _seg_embeddings(segments, audio, sr, model, device)
        finally:
            _free(model)
    else:
        embs = [None] * len(segments)

    valid = np.array([e is not None for e in embs], dtype=bool)
    return embs, valid


def _load_audio(norm_wav: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(norm_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import torch
        import torchaudio.functional as F_ta

        audio = F_ta.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
        sr = TARGET_SR
    return audio, sr


def diarize_multi(
    segments: List[dict],
    norm_wav: str,
    threshold: float = PER_SEG_THRESHOLD,
    agents_index_path: Optional[str] = None,
    force_cpu: bool = True,
) -> Dict[str, object]:
    if not segments:
        return _unknown_result(segments, "no transcript segments")

    try:
        audio, sr = _load_audio(norm_wav)
    except Exception as e:
        return _unknown_result(segments, f"cannot read normalized audio: {e}")

    voiceprints = _load_voiceprints(agents_index_path)
    voiceprints_by_dim = _group_voiceprints_by_dim(voiceprints)
    voiceprint_dims = {dim: len(vps) for dim, vps in voiceprints_by_dim.items()}
    if not voiceprints_by_dim:
        return _unknown_result(segments, "no supported enrolled voiceprints")

    per_agent_thresh = _per_agent_thresholds(threshold, agents_index_path)

    embs_by_dim: Dict[int, List[Optional[np.ndarray]]] = {}
    valid_by_dim: Dict[int, np.ndarray] = {}
    sims_by_dim: Dict[int, np.ndarray] = {}
    slugs_by_dim: Dict[int, List[str]] = {}

    for dim in sorted(voiceprints_by_dim, reverse=True):
        try:
            embs_dim, valid_dim = _segment_embeddings_for_dim(
                dim, segments, audio, sr, force_cpu
            )
        except Exception as e:
            logger.warning("Could not build %s-dim segment embeddings: %s", dim, e)
            embs_dim = [None] * len(segments)
            valid_dim = np.zeros(len(segments), dtype=bool)

        embs_by_dim[dim] = embs_dim
        valid_by_dim[dim] = valid_dim
        slugs = list(voiceprints_by_dim[dim].keys())
        slugs_by_dim[dim] = slugs

        sims = np.zeros((len(segments), len(slugs)), dtype=np.float32)
        if slugs:
            # Each agent has an (N, dim) stack of centroids; per-segment
            # similarity is the max cosine across that agent's centroids.
            stacks = [voiceprints_by_dim[dim][s][1] for s in slugs]
            for i, emb in enumerate(embs_dim):
                if emb is None:
                    continue
                emb = np.asarray(emb, dtype=np.float32).squeeze()
                if emb.ndim != 1 or emb.shape[0] != dim:
                    continue
                en = emb / max(np.linalg.norm(emb), 1e-8)
                for j, stack in enumerate(stacks):
                    sims[i, j] = float(np.max(stack @ en))
        sims_by_dim[dim] = sims

    if not any(valid.any() for valid in valid_by_dim.values()):
        return _unknown_result(segments, "no valid segment embeddings")

    agent_scores: Dict[str, float] = {}
    agent_backend: Dict[str, Tuple[int, int]] = {}
    for dim, slugs in slugs_by_dim.items():
        valid_rows = np.where(valid_by_dim[dim])[0]
        if not len(valid_rows):
            continue
        for j, slug in enumerate(slugs):
            col = sims_by_dim[dim][valid_rows, j]
            k_top = max(5, int(len(col) * 0.30))
            agent_scores[slug] = float(np.mean(np.sort(col)[-k_top:]))
            agent_backend[slug] = (dim, j)

    if not agent_scores:
        return _unknown_result(segments, "no agent scores could be computed")

    ranked = sorted(agent_scores, key=agent_scores.get, reverse=True)
    agent_slug = ranked[0]
    agent_dim, j_agent = agent_backend[agent_slug]
    agent_name = voiceprints_by_dim[agent_dim][agent_slug][0]
    logger.info(
        "agent-rank (top30%% mean): %s",
        {s: round(agent_scores[s], 3) for s in ranked[:5]},
    )

    valid = valid_by_dim[agent_dim]
    sims = sims_by_dim[agent_dim]
    embs = embs_by_dim[agent_dim]

    # Per-agent threshold derived from agents.json (max_outside_sim + margin),
    # falling back to the global threshold for agents without enrollment stats.
    agent_threshold = float(per_agent_thresh.get(agent_slug, threshold))
    if agent_threshold != threshold:
        logger.info(
            "per-agent threshold for %s = %.3f (global=%.3f)",
            agent_slug, agent_threshold, threshold,
        )
    seg_best_agent: List[Optional[str]] = []
    match_counts: Dict[str, int] = {}
    match_sims: Dict[str, List[float]] = {}
    for i in range(len(segments)):
        if not valid[i]:
            seg_best_agent.append(None)
            continue
        sim = float(sims[i, j_agent])
        if sim >= agent_threshold:
            seg_best_agent.append(agent_slug)
            match_counts[agent_slug] = match_counts.get(agent_slug, 0) + 1
            match_sims.setdefault(agent_slug, []).append(sim)
        else:
            seg_best_agent.append(None)

    if match_counts.get(agent_slug, 0) < AGENT_MIN_MATCHED:
        logger.info(
            "Only %d segments beat threshold for %s; falling back to Unknown",
            match_counts.get(agent_slug, 0),
            agent_slug,
        )
        agent_slug = ""
        agent_name = "Unknown Agent"
        seg_best_agent = [None] * len(segments)
        match_counts = {}
        match_sims = {}

    agent_avg_sim = (
        float(np.mean(match_sims[agent_slug]))
        if agent_slug in match_sims
        else 0.0
    )

    other_agent_count: Dict[str, int] = {}
    for i, seg in enumerate(segments):
        matched_slug = seg_best_agent[i]
        if matched_slug is None:
            seg["speaker"] = None
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = None
            seg.pop("agent_name", None)
            seg["_best_sim"] = float(sims[i, j_agent]) if j_agent >= 0 and valid[i] else 0.0
            seg["_best_match"] = agent_slug
            continue

        sim = float(sims[i, j_agent])
        seg["speaker"] = "SPEAKER_00"
        seg["identified_speaker"] = "AGENT"
        seg["agent_name"] = agent_name
        seg["display_speaker"] = agent_name
        seg["_best_sim"] = sim
        seg["_best_match"] = matched_slug

    unmatched_idxs = [i for i, seg in enumerate(segments) if seg["speaker"] is None]
    unmatched_embs = [embs[i] for i in unmatched_idxs if embs[i] is not None]
    if unmatched_embs:
        Xu = np.stack(unmatched_embs).astype(np.float32)
        norms = np.linalg.norm(Xu, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Xu = Xu / norms
        labels_u = _cluster_customers(Xu, MAX_CUSTOMER_CLUSTERS)

        valid_unmatched = [i for i in unmatched_idxs if embs[i] is not None]
        cluster_first: Dict[int, float] = {}
        for k, seg_i in enumerate(valid_unmatched):
            cid = int(labels_u[k])
            if cid not in cluster_first:
                cluster_first[cid] = float(segments[seg_i]["start"])
        order = sorted(cluster_first, key=lambda c: cluster_first[c])

        cluster_name = {cid: f"Customer {i + 1}" for i, cid in enumerate(order)}
        next_spk = 1 + len(other_agent_count)
        cid_to_spk = {cid: f"SPEAKER_{next_spk + i:02d}" for i, cid in enumerate(order)}

        # Compute customer-cluster centroids in normalised embedding space.
        # Used below to gate the "soft reclaim" — a segment is only pulled back
        # to AGENT if its cosine to the agent voiceprint EXCEEDS its similarity
        # to every customer-cluster centroid by a margin. This prevents the
        # previous over-reclaim where any sim>=0.28 segment became AGENT (which
        # mis-labelled customer turns at borderline phone-quality cosine).
        cust_centroids: Dict[int, np.ndarray] = {}
        for cid_u in set(int(c) for c in labels_u):
            mask_c = (labels_u == cid_u)
            if not mask_c.any():
                continue
            c = Xu[mask_c].mean(axis=0)
            n = np.linalg.norm(c)
            if n > 0:
                cust_centroids[cid_u] = c / n

        k_iter = 0
        soft_reclaimed = 0
        for seg_i in unmatched_idxs:
            seg = segments[seg_i]
            if embs[seg_i] is None:
                seg["display_speaker"] = None
                seg["speaker"] = None
                continue
            cid = int(labels_u[k_iter])
            k_iter += 1
            own_sim = float(sims[seg_i, j_agent]) if agent_slug and j_agent >= 0 else 0.0
            # Soft reclaim conditions:
            #  1. Cosine to agent voiceprint ≥ MERGE_TO_AGENT_SIM (0.28) — basic floor
            #  2. Cosine to agent voiceprint ≥ cosine to OWN customer cluster centroid
            #     (i.e. closer to the agent than to its assigned customer peers)
            #  3. OR cosine ≥ (agent_threshold − 0.07) — close to the hard threshold
            reclaim = False
            if agent_slug and own_sim >= MERGE_TO_AGENT_SIM and embs[seg_i] is not None:
                emb = embs[seg_i].astype(np.float32)
                n_emb = np.linalg.norm(emb)
                if n_emb > 0 and cid in cust_centroids:
                    emb_n = emb / n_emb
                    cust_sim = float(emb_n @ cust_centroids[cid])
                    if own_sim >= cust_sim:
                        reclaim = True
                # Borderline-but-close-to-threshold also reclaims (rescues short
                # agent acks like "Hi Edgar." / "What are your plans?" with sim
                # in the 0.30–0.42 band that the centroid check rejected).
                if not reclaim and own_sim >= max(agent_threshold - 0.07, 0.28):
                    reclaim = True
            if reclaim:
                seg["speaker"] = "SPEAKER_00"
                seg["identified_speaker"] = "AGENT"
                seg["agent_name"] = agent_name
                seg["display_speaker"] = agent_name
                seg["_best_sim"] = own_sim
                seg["_best_match"] = agent_slug
                soft_reclaimed += 1
            else:
                seg["speaker"] = cid_to_spk[cid]
                seg["display_speaker"] = cluster_name[cid]
                seg["identified_speaker"] = "CUSTOMER"
                seg.pop("agent_name", None)
        logger.info(
            "soft-reclaimed %d segments (gated by centroid distance)",
            soft_reclaimed,
        )

    speaker_mode = "per_segment_similarity"
    cluster_report: Dict[str, object] = {}
    total_segment_dur = sum(
        max(float(seg["end"]) - float(seg["start"]), 0.0) for seg in segments
    )
    initial_agent_dur = sum(
        max(float(seg["end"]) - float(seg["start"]), 0.0)
        for seg in segments
        if seg.get("identified_speaker") == "AGENT"
    )
    initial_agent_ratio = initial_agent_dur / max(total_segment_dur, 1e-6)
    valid_count = int(np.sum(valid))
    if (
        agent_slug
        and total_segment_dur >= CLUSTER_FIRST_MIN_DUR
        and valid_count >= CLUSTER_FIRST_MIN_SEGMENTS
        and initial_agent_ratio >= CLUSTER_FIRST_AGENT_RATIO
    ):
        cluster_ok, cluster_report = _cluster_speaker_roles(
            segments,
            embs,
            sims,
            j_agent,
            agent_slug,
            agent_name,
        )
        if cluster_ok:
            speaker_mode = "cluster_first_voiceprint"
            agent_cluster_sims = cluster_report.get("agent_sims") or []
            match_counts = {agent_slug: int(cluster_report.get("agent_count") or 0)}
            match_sims = {agent_slug: list(agent_cluster_sims)}
            agent_avg_sim = (
                float(np.mean(agent_cluster_sims)) if agent_cluster_sims else 0.0
            )
            cluster_report.pop("agent_sims", None)
            logger.info(
                "cluster-first role assignment enabled: dur=%.1fs valid=%d "
                "initial_agent_ratio=%.2f clusters=%s",
                total_segment_dur,
                valid_count,
                initial_agent_ratio,
                cluster_report.get("cluster_stats"),
            )
        else:
            logger.info("cluster-first role assignment skipped: %s", cluster_report)

    def _apply(seg: dict, ref: dict) -> None:
        seg["speaker"] = ref["speaker"]
        seg["identified_speaker"] = ref["identified_speaker"]
        seg["display_speaker"] = ref["display_speaker"]
        if ref["identified_speaker"] == "AGENT":
            seg["agent_name"] = ref.get("agent_name", agent_name)
        else:
            seg.pop("agent_name", None)

    n = len(segments)
    for i, seg in enumerate(segments):
        if seg.get("display_speaker"):
            continue
        left = next(
            (
                segments[j]
                for j in range(i - 1, -1, -1)
                if segments[j].get("display_speaker")
                and (float(segments[j]["end"]) - float(segments[j]["start"])) >= MIN_CONF_DUR
            ),
            None,
        )
        right = next(
            (
                segments[j]
                for j in range(i + 1, n)
                if segments[j].get("display_speaker")
                and (float(segments[j]["end"]) - float(segments[j]["start"])) >= MIN_CONF_DUR
            ),
            None,
        )
        if left and right:
            if left["display_speaker"] == right["display_speaker"]:
                _apply(seg, left)
            else:
                dl = float(seg["start"]) - float(left["end"])
                dr = float(right["start"]) - float(seg["end"])
                _apply(seg, left if dl <= dr else right)
        elif left:
            _apply(seg, left)
        elif right:
            _apply(seg, right)
        else:
            seg["speaker"] = "SPEAKER_99"
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = "Unknown"
            seg.pop("agent_name", None)

    # ── Anti-flip pass 1: tight (≤0.6s sandwiched between same speaker) ──
    for i in range(1, n - 1):
        dur = float(segments[i]["end"]) - float(segments[i]["start"])
        if dur > 0.6:
            continue
        prev = segments[i - 1].get("display_speaker")
        nxt = segments[i + 1].get("display_speaker")
        cur = segments[i].get("display_speaker")
        if prev and nxt and prev == nxt and cur != prev:
            _apply(segments[i], segments[i - 1])

    # ── Anti-flip pass 2: low-confidence sandwich (<2.5s, sim<0.30) ──
    # Catches the "Yeah." / "Okay." back-channels that get smoothed to AGENT
    # even though their cosine to the agent voiceprint is essentially zero.
    # If neighbours agree, trust them over the noisy embedding.
    for i in range(1, n - 1):
        dur = float(segments[i]["end"]) - float(segments[i]["start"])
        sim = float(segments[i].get("_best_sim") or 0.0)
        if dur > 2.5 or sim >= 0.30:
            continue
        prev = segments[i - 1].get("display_speaker")
        nxt = segments[i + 1].get("display_speaker")
        cur = segments[i].get("display_speaker")
        if prev and nxt and prev == nxt and cur != prev:
            # Only flip if at least one neighbour has a strong (≥0.40) signal
            prev_sim = float(segments[i - 1].get("_best_sim") or 0.0)
            nxt_sim = float(segments[i + 1].get("_best_sim") or 0.0)
            if max(prev_sim, nxt_sim) >= 0.40:
                _apply(segments[i], segments[i - 1])

    # ── Anti-flip pass 2.5: low-sim AGENT segments starting with back-channel ──
    # Customer acknowledgements like "Yeah, please.", "Okay, that's right.",
    # "Yeah, I'm good, thank you." are mis-labelled AGENT when the embedding
    # cosine is borderline. Rule: if the FIRST word is a back-channel AND total
    # words ≤ 5 AND duration < 2.5s AND cosine < 0.30, demote to CUSTOMER.
    BACKCHANNEL_STARTS = {
        "yeah", "yes", "yep", "yup", "ok", "okay", "right", "sure",
        "alright", "uhuh", "mhm", "mm", "mhmm", "kay",
    }
    backchannel_demoted = 0
    for i, seg in enumerate(segments):
        if seg.get("identified_speaker") != "AGENT":
            continue
        sim = float(seg.get("_best_sim") or 0.0)
        if sim >= 0.30:
            continue
        dur = float(seg["end"]) - float(seg["start"])
        if dur > 4.0:
            continue
        words = _norm_words(seg.get("text") or "")
        if not (1 <= len(words) <= 15):
            continue
        if words[0] not in BACKCHANNEL_STARTS:
            continue
        seg["identified_speaker"] = "CUSTOMER"
        seg["display_speaker"] = "Customer 1"
        seg["speaker"] = "SPEAKER_01"
        seg.pop("agent_name", None)
        backchannel_demoted += 1
    if backchannel_demoted:
        logger.info("demoted %d low-sim back-channel-led segs AGENT → CUSTOMER", backchannel_demoted)

    # ── Anti-flip pass 3: zero-similarity AGENT segments are smoothing artefacts ──
    # If a segment is labelled AGENT but its cosine to the agent voiceprint is
    # below 0.10, it was almost certainly assigned by neighbour-vote on noisy
    # input. If the *previous* meaningful segment is CUSTOMER, demote it.
    for i, seg in enumerate(segments):
        if seg.get("identified_speaker") != "AGENT":
            continue
        sim = float(seg.get("_best_sim") or 0.0)
        if sim >= 0.10:
            continue
        # Check if surrounded by customer segments
        prev_cust = any(
            segments[j].get("identified_speaker") == "CUSTOMER"
            for j in range(max(0, i - 2), i)
        )
        next_cust = any(
            segments[j].get("identified_speaker") == "CUSTOMER"
            for j in range(i + 1, min(n, i + 3))
        )
        if prev_cust and next_cust:
            ref = _nearest_customer_ref(i) if "_nearest_customer_ref" in dir() else None
            # Use inline lookup instead of _nearest_customer_ref (defined below)
            for j in (i - 1, i + 1):
                if 0 <= j < n and segments[j].get("identified_speaker") == "CUSTOMER":
                    _apply(seg, segments[j])
                    break

    def _nearest_customer_ref(idx: int) -> Optional[dict]:
        center = (float(segments[idx]["start"]) + float(segments[idx]["end"])) / 2.0
        refs = [
            s
            for s in segments
            if s.get("identified_speaker") == "CUSTOMER" and s.get("display_speaker")
        ]
        if not refs:
            return None
        return min(
            refs,
            key=lambda s: abs(
                ((float(s["start"]) + float(s["end"])) / 2.0) - center
            ),
        )

    def _apply_agent(seg: dict) -> None:
        seg["speaker"] = "SPEAKER_00"
        seg["identified_speaker"] = "AGENT"
        seg["agent_name"] = agent_name
        seg["display_speaker"] = agent_name
        seg["_best_match"] = agent_slug

    role_corrections = {"agent_to_customer": 0, "customer_to_agent": 0}
    for i, seg in enumerate(segments):
        dur = float(seg["end"]) - float(seg["start"])
        sim = float(seg.get("_best_sim") or 0.0)
        prev_seg = segments[i - 1] if i > 0 else None
        next_seg = segments[i + 1] if i + 1 < n else None
        prev_is_agent = bool(prev_seg and prev_seg.get("identified_speaker") == "AGENT")
        prev_is_customer = bool(
            prev_seg and prev_seg.get("identified_speaker") == "CUSTOMER"
        )
        next_is_customer = bool(
            next_seg and next_seg.get("identified_speaker") == "CUSTOMER"
        )

        if (
            seg.get("identified_speaker") == "AGENT"
            and dur <= SHORT_REPLY_MAX_DUR
            and sim <= SHORT_REPLY_MAX_SIM
            and not _agent_phrase(str(seg.get("text") or ""))
            and _short_customer_reply(str(seg.get("text") or ""))
            and prev_is_agent
            and (
                _looks_like_question(str(prev_seg.get("text") or ""))
                or next_is_customer
            )
        ):
            ref = _nearest_customer_ref(i)
            if ref:
                _apply(seg, ref)
                role_corrections["agent_to_customer"] += 1
                continue

        at_call_end = i >= n - 2
        if (
            seg.get("identified_speaker") == "CUSTOMER"
            and dur <= SHORT_REPLY_MAX_DUR
            and sim >= FAREWELL_AGENT_MIN_SIM
            and prev_is_customer
            and at_call_end
            and _farewell(str(seg.get("text") or ""))
        ):
            _apply_agent(seg)
            role_corrections["customer_to_agent"] += 1

    if any(role_corrections.values()):
        logger.info("role corrections applied: %s", role_corrections)

    per_speaker: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        lbl = seg.get("display_speaker", "?")
        per_speaker.setdefault(lbl, {"turns": 0, "seconds": 0.0})
        per_speaker[lbl]["turns"] += 1
        per_speaker[lbl]["seconds"] += float(seg["end"]) - float(seg["start"])

    return {
        "segments": segments,
        "agent_slug": agent_slug,
        "agent_name": agent_name,
        "agent_similarity": round(agent_avg_sim, 3),
        "n_speakers": len(per_speaker),
        "match_counts": match_counts,
        "other_agent_count": other_agent_count,
        "per_speaker": per_speaker,
        "matched_backend_dim": agent_dim if agent_slug else None,
        "voiceprint_dims": voiceprint_dims,
        "role_corrections": role_corrections,
        "speaker_mode": speaker_mode,
        "agent_threshold_used": round(float(agent_threshold), 3) if agent_slug else round(float(threshold), 3),
        "cluster_report": cluster_report,
    }
