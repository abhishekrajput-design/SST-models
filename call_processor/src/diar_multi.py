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
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from src.voiceprints import resolve_voiceprint_path

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MIN_SEG_S_FOR_EMB = 0.3
PER_SEG_THRESHOLD = 0.30
AGENT_MIN_MATCHED = 3
MAX_CUSTOMER_CLUSTERS = 3
MIN_CONF_DUR = 1.0
MERGE_TO_AGENT_SIM = 0.28

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_AGENTS_INDEX = os.path.join(_DATA_DIR, "agent_voiceprints", "agents.json")


def _load_voiceprints(path: Optional[str] = None) -> Dict[str, Tuple[str, np.ndarray]]:
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
        vp_path = info.get("voiceprint_path") or info.get("voiceprint")
        if not vp_path:
            continue
        vp_path = resolve_voiceprint_path(vp_path, p)
        if not os.path.isfile(vp_path):
            continue
        try:
            vp = np.load(vp_path).astype(np.float32).squeeze()
        except Exception as e:
            logger.warning("Skipping %s: failed to load voiceprint (%s)", slug, e)
            continue
        if vp.ndim != 1:
            logger.warning("Skipping %s: invalid voiceprint shape %s", slug, vp.shape)
            continue
        n = np.linalg.norm(vp)
        if n > 0:
            vp = vp / n
        name = info.get("agent_name") or info.get("name") or slug
        out[slug] = (name, vp)
    return out


def _seg_embeddings(
    segments: List[dict],
    audio: np.ndarray,
    sr: int,
    model,
    device: str,
) -> Tuple[List[Optional[np.ndarray]], np.ndarray]:
    from src.speaker_role import _embed

    embs: List[Optional[np.ndarray]] = []
    for seg in segments:
        s = int(float(seg["start"]) * sr)
        e = min(int(float(seg["end"]) * sr), len(audio))
        chunk = audio[s:e]
        if len(chunk) < int(sr * MIN_SEG_S_FOR_EMB):
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
        name, vp = value
        dim = int(vp.shape[0])
        if dim not in (192, 512):
            logger.warning("Skipping %s: unsupported voiceprint dim=%s", slug, dim)
            continue
        grouped.setdefault(dim, {})[slug] = (name, vp)
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
                for seg in segments:
                    s = int(float(seg["start"]) * sr)
                    e = min(int(float(seg["end"]) * sr), len(audio))
                    chunk = audio[s:e]
                    if len(chunk) < int(sr * MIN_SEG_S_FOR_EMB):
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
            V = np.stack([voiceprints_by_dim[dim][s][1] for s in slugs])
            for i, emb in enumerate(embs_dim):
                if emb is None:
                    continue
                emb = np.asarray(emb, dtype=np.float32).squeeze()
                if emb.ndim != 1 or emb.shape[0] != dim:
                    continue
                en = emb / max(np.linalg.norm(emb), 1e-8)
                sims[i] = V @ en
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

    seg_best_agent: List[Optional[str]] = []
    match_counts: Dict[str, int] = {}
    match_sims: Dict[str, List[float]] = {}
    for i in range(len(segments)):
        if not valid[i]:
            seg_best_agent.append(None)
            continue
        sim = float(sims[i, j_agent])
        if sim >= threshold:
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
            if agent_slug and own_sim >= MERGE_TO_AGENT_SIM:
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
            "soft-reclaimed %d segments via per-seg sim>=%.2f",
            soft_reclaimed,
            MERGE_TO_AGENT_SIM,
        )

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

    for i in range(1, n - 1):
        dur = float(segments[i]["end"]) - float(segments[i]["start"])
        if dur > 0.6:
            continue
        prev = segments[i - 1].get("display_speaker")
        nxt = segments[i + 1].get("display_speaker")
        cur = segments[i].get("display_speaker")
        if prev and nxt and prev == nxt and cur != prev:
            _apply(segments[i], segments[i - 1])

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
    }
