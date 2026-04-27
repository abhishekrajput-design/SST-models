"""
Voiceprint-first multi-speaker diarization.

Pipeline for a single call:
  1. Per-segment ECAPA embedding.
  2. For each segment, cosine vs EVERY enrolled voiceprint.
     If max sim >= PER_SEG_THRESHOLD -> label segment with that agent name.
  3. Count matches per enrolled agent; the agent with the most matched segments
     (and minimum match count) is THE AGENT of this call.  That agent's
     segments render on the LEFT (AGENT).
  4. Segments matched to OTHER enrolled agents render on the RIGHT with their
     agent name ("Other Agent: Sarah Aziz" etc.).
  5. Unmatched segments (sim < threshold for all voiceprints) are clustered
     with KMeans (k auto 1..3) into "Customer 1", "Customer 2", ....
  6. Short / low-confidence segments inherit the label of the nearest
     confident neighbour.

Segment fields set:
  speaker             SPEAKER_00 (agent) | SPEAKER_01..SPEAKER_99 (other)
  identified_speaker  "AGENT" | "CUSTOMER"  (CUSTOMER for non-matched agent too)
  agent_name          enrolled agent name on AGENT segments
  display_speaker     name to show in the bubble
  _best_sim           max cosine across all voiceprints
  _best_match         slug of the enrolled agent best-matched
  _cluster_label      final display name
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MIN_SEG_S_FOR_EMB    = 0.3
PER_SEG_THRESHOLD    = 0.35     # per-segment cosine to call it a match
AGENT_MIN_MATCHED    = 5        # an agent needs at least this many matched segs to claim the call
MAX_CUSTOMER_CLUSTERS = 3
MIN_CONF_DUR         = 1.0

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
        if not os.path.isabs(vp_path):
            vp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), vp_path)
        if not os.path.isfile(vp_path):
            continue
        try:
            vp = np.load(vp_path).astype(np.float32).squeeze()
        except Exception:
            continue
        if vp.ndim != 1:
            continue
        n = np.linalg.norm(vp)
        if n > 0:
            vp = vp / n
        name = info.get("agent_name") or slug
        out[slug] = (name, vp)
    return out


def _seg_embeddings(segments: List[dict], audio: np.ndarray, sr: int,
                     model, device: str):
    from src.speaker_role import _embed
    embs = []
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
    """Split unmatched embeddings into 1..max_k clusters using KMeans + silhouette."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return np.zeros(len(X), dtype=int)

    if len(X) < 6:
        return np.zeros(len(X), dtype=int)

    best_labels = np.zeros(len(X), dtype=int)
    best_k = 1
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
            best_k = k
    return best_labels


def diarize_multi(
    segments: List[dict],
    norm_wav: str,
    threshold: float = PER_SEG_THRESHOLD,
    agents_index_path: Optional[str] = None,
    force_cpu: bool = True,
) -> Dict[str, object]:
    voiceprints = _load_voiceprints(agents_index_path)

    # ── Load audio ────────────────────────────────────────────────────────────
    audio, sr = sf.read(norm_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import torch, torchaudio.functional as F_ta
        audio = F_ta.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
        sr = TARGET_SR

    # ── Per-segment embeddings ───────────────────────────────────────────────
    # Use the embedding model whose dimension matches the enrolled voiceprints.
    # ECAPA=192-dim (default), CAM++=512-dim (if wespeaker is installed and
    # agents were re-enrolled after upgrading speaker_role.py).
    vp_dims = {vp.shape[0] for _, vp in voiceprints.values()} if voiceprints else {192}
    target_dim = 512 if 512 in vp_dims else 192
    if voiceprints and len(vp_dims) > 1:
        voiceprints = {s: (n, vp) for s, (n, vp) in voiceprints.items()
                       if vp.shape[0] == target_dim}

    if target_dim == 512:
        from src.embedding_campp import EmbeddingModel as _EmbModel
        _emb = _EmbModel()
        _emb.load(force_cpu=force_cpu)
        embs = []
        for seg in segments:
            s = int(float(seg["start"]) * sr)
            e = min(int(float(seg["end"]) * sr), len(audio))
            chunk = audio[s:e]
            if len(chunk) < int(sr * MIN_SEG_S_FOR_EMB):
                embs.append(None)
                continue
            embs.append(_emb.embed_chunk(chunk, sr))
        valid = np.array([e is not None for e in embs], dtype=bool)
        _emb.unload()
    else:
        from src.speaker_role import _load_ecapa, _free
        model, device = _load_ecapa(force_cpu=force_cpu)
        try:
            embs, valid = _seg_embeddings(segments, audio, sr, model, device)
        finally:
            _free(model)

    # If no voiceprints or no valid embeddings — fall back to single speaker
    if not voiceprints or not valid.any():
        for s in segments:
            s["speaker"] = "SPEAKER_00"
            s["identified_speaker"] = "AGENT"
            s["agent_name"] = "Unknown Agent"
            s["display_speaker"] = "Unknown Agent"
            s["_best_sim"] = 0.0
            s["_best_match"] = ""
        return {"segments": segments, "agent_slug": "", "agent_name": "Unknown Agent",
                "agent_similarity": 0.0, "n_speakers": 1,
                "match_counts": {}, "per_cluster_match": {}}

    # Similarity matrix (segments × agents)
    slugs = list(voiceprints.keys())
    V = np.stack([voiceprints[s][1] for s in slugs])             # (A, 192)
    X_list = [e for e in embs]
    sims = np.zeros((len(segments), len(slugs)), dtype=np.float32)
    for i, e in enumerate(embs):
        if e is None:
            continue
        en = e / max(np.linalg.norm(e), 1e-8)
        sims[i] = V @ en

    # ── Pick THE agent for this call ──────────────────────────────────────────
    # Rank enrolled agents by top-30% mean cosine across all valid segments.
    # This focuses on segments where the agent is actually speaking while being
    # robust to noisy non-agent segments.
    agent_scores: Dict[str, float] = {}
    valid_rows = np.where(valid)[0]
    if len(valid_rows) > 0:
        for j, slug_k in enumerate(slugs):
            col = sims[valid_rows, j]
            k_top = max(5, int(len(col) * 0.30))
            agent_scores[slug_k] = float(np.mean(np.sort(col)[-k_top:]))
    else:
        agent_scores = {s: 0.0 for s in slugs}

    ranked = sorted(agent_scores, key=agent_scores.get, reverse=True)
    agent_slug = ranked[0] if ranked else ""
    agent_name = voiceprints[agent_slug][0] if agent_slug else "Unknown Agent"
    logger.info("agent-rank (top30%% mean): %s",
                {s: round(agent_scores[s], 3) for s in ranked[:5]})

    # Per-segment classification using a RELATIVE score: how much higher is
    # the cosine to the chosen agent vs the mean cosine to all OTHER enrolled
    # agents? This controls for segments whose ECAPA embedding is generically
    # similar to any voiceprint (noisy short segments) while boosting segments
    # that specifically match the chosen agent.
    j_agent = slugs.index(agent_slug) if agent_slug else -1
    seg_best_agent: List[Optional[int]] = []
    match_counts: Dict[str, int] = {}
    match_sims: Dict[str, List[float]] = {}

    if j_agent >= 0:
        # Absolute threshold: cosine to the chosen agent >= threshold (0.35).
        for i in range(len(segments)):
            if not valid[i]:
                seg_best_agent.append(None); continue
            if sims[i, j_agent] >= threshold:
                seg_best_agent.append(j_agent)
                match_counts[agent_slug] = match_counts.get(agent_slug, 0) + 1
                match_sims.setdefault(agent_slug, []).append(float(sims[i, j_agent]))
            else:
                seg_best_agent.append(None)
    else:
        seg_best_agent = [None] * len(segments)

    if agent_slug and match_counts.get(agent_slug, 0) < AGENT_MIN_MATCHED:
        logger.info("Only %d segments beat baseline for %s — falling back to Unknown",
                    match_counts.get(agent_slug, 0), agent_slug)
        agent_slug = ""
        agent_name = "Unknown Agent"
        seg_best_agent = [None] * len(segments)
        match_counts = {}
        match_sims = {}

    agent_avg_sim = (float(np.mean(match_sims[agent_slug]))
                     if agent_slug in match_sims else 0.0)

    # ── Label every segment ───────────────────────────────────────────────────
    # 1. Segments matched to AGENT slug → AGENT
    # 2. Segments matched to another enrolled agent → CUSTOMER with that agent's name
    # 3. Unmatched segments → temporarily "CUSTOMER ?" (will cluster later)
    other_agent_count: Dict[str, int] = {}
    for i, seg in enumerate(segments):
        j = seg_best_agent[i]
        if j is None:
            seg["speaker"] = None          # to be assigned after customer clustering
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = None
            seg["_best_sim"] = 0.0
            seg["_best_match"] = ""
            continue
        matched_slug = slugs[j]
        matched_name = voiceprints[matched_slug][0]
        sim = float(sims[i, j])
        if matched_slug and matched_slug == agent_slug:
            seg["speaker"] = "SPEAKER_00"
            seg["identified_speaker"] = "AGENT"
            seg["agent_name"] = matched_name
            seg["display_speaker"] = matched_name
        else:
            # Other enrolled agent — display on right labelled with their name
            other_agent_count[matched_slug] = other_agent_count.get(matched_slug, 0) + 1
            seg["speaker"] = f"SPEAKER_{(list(other_agent_count.keys()).index(matched_slug) + 1):02d}"
            seg["identified_speaker"] = "CUSTOMER"     # anything not the matched agent = right side
            seg["display_speaker"] = matched_name
            seg.pop("agent_name", None)
        seg["_best_sim"] = sim
        seg["_best_match"] = matched_slug

    # ── Cluster unmatched segments into Customer 1/2/3 ────────────────────────
    unmatched_idxs = [i for i, seg in enumerate(segments) if seg["speaker"] is None]
    unmatched_embs = [embs[i] for i in unmatched_idxs if embs[i] is not None]
    if unmatched_embs:
        Xu = np.stack(unmatched_embs).astype(np.float32)
        norms = np.linalg.norm(Xu, axis=1, keepdims=True); norms[norms == 0] = 1
        Xu = Xu / norms
        labels_u = _cluster_customers(Xu, MAX_CUSTOMER_CLUSTERS)

        # Map cluster id -> first appearance order
        valid_unmatched = [i for i in unmatched_idxs if embs[i] is not None]
        cluster_first: Dict[int, float] = {}
        for k, seg_i in enumerate(valid_unmatched):
            cid = int(labels_u[k])
            if cid not in cluster_first:
                cluster_first[cid] = float(segments[seg_i]["start"])
        order = sorted(cluster_first, key=lambda c: cluster_first[c])
        cluster_name = {cid: f"Customer {i + 1}" for i, cid in enumerate(order)}

        # Find the next free SPEAKER_NN id
        next_spk = 1 + len(other_agent_count)
        cid_to_spk = {}
        for cid in order:
            cid_to_spk[cid] = f"SPEAKER_{next_spk:02d}"
            next_spk += 1

        k_iter = 0
        for seg_i in unmatched_idxs:
            seg = segments[seg_i]
            if embs[seg_i] is None:
                # Will smooth from neighbours
                seg["display_speaker"] = None
                seg["speaker"] = None
                continue
            cid = int(labels_u[k_iter]); k_iter += 1
            seg["speaker"] = cid_to_spk[cid]
            seg["display_speaker"] = cluster_name[cid]
            seg["identified_speaker"] = "CUSTOMER"

    # ── Fill in any still-unknown short segments by neighbour vote ────────────
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
        # Nearest confident neighbour
        left = next((segments[j] for j in range(i - 1, -1, -1)
                      if segments[j].get("display_speaker") and
                      (float(segments[j]["end"]) - float(segments[j]["start"])) >= MIN_CONF_DUR),
                     None)
        right = next((segments[j] for j in range(i + 1, n)
                       if segments[j].get("display_speaker") and
                       (float(segments[j]["end"]) - float(segments[j]["start"])) >= MIN_CONF_DUR),
                      None)
        if left and right:
            # Prefer agreeing neighbours
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

    # Cleanup: short flips surrounded by same neighbour label (≤0.6 s)
    for i in range(1, n - 1):
        dur = float(segments[i]["end"]) - float(segments[i]["start"])
        if dur > 0.6:
            continue
        prev = segments[i - 1].get("display_speaker")
        nxt  = segments[i + 1].get("display_speaker")
        cur  = segments[i].get("display_speaker")
        if prev and nxt and prev == nxt and cur != prev:
            _apply(segments[i], segments[i - 1])

    # ── Summary stats ─────────────────────────────────────────────────────────
    per_speaker: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        lbl = seg.get("display_speaker", "?")
        per_speaker.setdefault(lbl, {"turns": 0, "seconds": 0.0})
        per_speaker[lbl]["turns"] += 1
        per_speaker[lbl]["seconds"] += float(seg["end"]) - float(seg["start"])

    return {
        "segments":          segments,
        "agent_slug":        agent_slug,
        "agent_name":        agent_name,
        "agent_similarity":  round(agent_avg_sim, 3),
        "n_speakers":        len(per_speaker),
        "match_counts":      match_counts,
        "other_agent_count": other_agent_count,
        "per_speaker":       per_speaker,
    }
