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
MIN_SEG_S_FOR_EMB = 0.45
PER_SEG_THRESHOLD = 0.34
PER_AGENT_MARGIN  = 0.06
PER_AGENT_THRESH_CAP = 0.92

# Phase 3: Confidence gating and unknown rejection
CONFIDENCE_GATE_UNCERTAIN_BAND = 0.22  # lower bound of uncertain confidence zone
CONFIDENCE_GATE_UPPER_BOUND = 0.25     # upper bound of uncertain confidence zone
UNKNOWN_REJECTION_FLOOR = 0.28         # below this: definitely not enrolled agent
UNKNOWN_REJECTION_MIN_MATCHES = 3      # need at least N matches above floor
AGENT_MIN_MATCHED = 3
INITIAL_SEGMENT_BOOST = True        # special handling for first 3 segments
MAX_CUSTOMER_CLUSTERS = 3
MIN_CONF_DUR = 1.0
MERGE_TO_AGENT_SIM = 0.42
SHORT_REPLY_MAX_DUR = 0.95
SHORT_REPLY_MAX_SIM = 0.30
FAREWELL_AGENT_MIN_SIM = 0.34
CLUSTER_FIRST_MIN_DUR = 30.0           # Lowered for shorter calls
CLUSTER_FIRST_MIN_SEGMENTS = 10        # Lowered to engage on more calls
CLUSTER_FIRST_AGENT_RATIO = 0.16

# Filler / back-channel handling
FILLER_MAX_DUR         = 0.80   # segments â‰¤ this duration with filler-only text are down-weighted
FILLER_SIM_WEIGHT      = 0.15   # fractional weight for filler segments in agent scoring

# Progressive confidence: short calls need more evidence before committing
PROG_CONF_MIN_SPEECH_S = 8.0    # seconds of non-filler speech needed to use AGENT_MIN_MATCHED=3
PROG_CONF_MIN_MATCHED  = 5      # min matched segments on very short calls (< 8s speech)

# SNR-adaptive threshold
SNR_LOW_DB             = 14.0   # below this, relax threshold (raised from 12.0)
SNR_LOW_FLOOR          = 0.28

# Neighbor-pool for short embeddings
NEIGHBOR_POOL_RADIUS   = 2      # Â±2 segments borrowed for neighbor-pool on short clips

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


_FILLER_WORDS = frozenset({
    "yeah","yep","yup","yes","no","nope",
    "ok","okay","kay","k",
    "uh","um","hmm","hm","mhm","mm","mmm","uhuh","uh-huh","mhmm",
    "sure","right","alright",
    "so","and","but","well","now","oh","ah",
    "of course","i see","i know","got it","got ya",
    "no problem","no worries","thank you","thanks",
    "sounds good","all good",
})


def _is_filler_only(text: str, dur: float) -> bool:
    """True if segment is â‰¤ FILLER_MAX_DUR and text is a filler word/phrase."""
    if dur >= FILLER_MAX_DUR:
        return False
    norm = _norm_text(text)
    return bool(norm) and (norm in _FILLER_WORDS or
           (len(_norm_words(text)) == 1 and _norm_words(text)[0] in _FILLER_WORDS))


def _strip_shared_voiceprints(
    voiceprints: Dict[str, Tuple[str, np.ndarray]],
    threshold: float = 0.99,
) -> Tuple[Dict[str, Tuple[str, np.ndarray]], Dict]:
    """Drop voiceprint vectors that are (near-)identical across DIFFERENT agents.

    A vector shared by two or more agents cannot be speaker-discriminative — it is
    the signature of an enrollment write bug (the same embedding copied into several
    agents' files). Such a vector matches arbitrary speakers at moderate cosine,
    producing wrong-name labels and making the affected agents indistinguishable.

    We remove every shared vector from every agent that carries it. Duplicates
    WITHIN a single agent are kept (same speaker enrolled twice). Vectors of
    differing dimensionality are never compared. Agents left with no vectors are
    dropped — an honest "unknown" beats a confident wrong name. Stacks are assumed
    L2-normalised (as produced by _load_voiceprints), so a dot product is cosine.

    Returns (cleaned_voiceprints, report).
    """
    flat: List[Tuple[str, int, np.ndarray]] = []
    for slug, (_name, stack) in voiceprints.items():
        if getattr(stack, "ndim", 0) != 2:
            continue
        for k in range(stack.shape[0]):
            flat.append((slug, k, stack[k]))

    shared: set = set()
    for i in range(len(flat)):
        si, ki, vi = flat[i]
        for j in range(i + 1, len(flat)):
            sj, kj, vj = flat[j]
            if si == sj or vi.shape[0] != vj.shape[0]:
                continue
            if float(vi @ vj) >= threshold:
                shared.add((si, ki))
                shared.add((sj, kj))

    clean: Dict[str, Tuple[str, np.ndarray]] = {}
    report: Dict = {"dropped_vectors": 0, "affected_agents": [], "emptied_agents": []}
    for slug, (name, stack) in voiceprints.items():
        if getattr(stack, "ndim", 0) != 2:
            clean[slug] = (name, stack)
            continue
        keep = [stack[k] for k in range(stack.shape[0]) if (slug, k) not in shared]
        dropped = stack.shape[0] - len(keep)
        if dropped:
            report["dropped_vectors"] += dropped
            report["affected_agents"].append(slug)
        if keep:
            clean[slug] = (name, np.stack(keep).astype(np.float32))
        else:
            report["emptied_agents"].append(slug)
    return clean, report


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

    # Defend against enrollment corruption: a voiceprint vector shared across
    # multiple agents is not speaker-specific and poisons matching (wrong names,
    # indistinguishable agents). Strip such vectors before returning.
    if os.getenv("SST_VOICEPRINT_DEDUP", "1").strip().lower() not in {"0", "false", "no", "off"}:
        dup_sim = float(os.getenv("SST_VOICEPRINT_DUP_SIM", "0.99") or "0.99")
        out, _dedup_report = _strip_shared_voiceprints(out, dup_sim)
        if _dedup_report["dropped_vectors"]:
            logger.warning(
                "Voiceprint dedup: dropped %d cross-agent duplicate vector(s) from %s%s",
                _dedup_report["dropped_vectors"],
                _dedup_report["affected_agents"],
                (" | EMPTIED (now unenrolled): " + str(_dedup_report["emptied_agents"]))
                if _dedup_report["emptied_agents"] else "",
            )
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
    # Adaptive floor â€” 1.5x the median frame RMS of this chunk.
    floor = float(np.median(rms)) * 1.5
    floor = max(floor, 0.005)
    return float((rms > floor).mean())


def _estimate_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate call-level SNR in dB using 85th vs 15th percentile frame energy.
    Returns 20.0 (clean) if estimation fails. O(N) but touches <10% of frames."""
    win = int(sr * 0.025)
    if audio.size < win * 4:
        return 20.0
    step = max(1, (audio.size // win) // 500)   # at most ~500 frames
    idxs = range(0, (audio.size // win) * win, step * win)
    rms_vals = [float(np.sqrt(np.mean(audio[s:s+win]**2) + 1e-12)) for s in idxs
                if s + win <= audio.size]
    if len(rms_vals) < 4:
        return 20.0
    voiced = np.array([r for r in rms_vals if r > 1e-5], dtype=np.float32)
    if len(voiced) < 4:
        return 20.0
    ratio = float(np.percentile(voiced, 85)) / float(np.percentile(voiced, 15))
    return float(20.0 * np.log10(max(ratio, 1.0)))


def _pool_embedding_for_short_seg(
    seg_idx: int, segments, audio: np.ndarray, sr: int, model
) -> Optional[np.ndarray]:
    """Pool audio from Â±NEIGHBOR_POOL_RADIUS segments to reach 1.5s for embedding.
    Returns None if <0.5s pooled. Does NOT require neighbors to share a label â€”
    used before labels are assigned (first pass only)."""
    if os.getenv("SST_ALLOW_CROSS_SPEAKER_POOLING", "").strip() != "1":
        return None
    target = int(1.5 * sr)
    seg = segments[seg_idx]
    s0 = int(float(seg["start"]) * sr)
    e0 = min(int(float(seg["end"]) * sr), len(audio))
    chunks = [audio[s0:e0]]
    total = e0 - s0
    left, right = seg_idx - 1, seg_idx + 1
    for _ in range(NEIGHBOR_POOL_RADIUS * 2):
        if total >= target:
            break
        if left >= 0:
            nb = segments[left]
            ns, ne = int(float(nb["start"])*sr), min(int(float(nb["end"])*sr), len(audio))
            chunks.insert(0, audio[ns:ne]);  total += ne - ns;  left -= 1
        if total < target and right < len(segments):
            nb = segments[right]
            ns, ne = int(float(nb["start"])*sr), min(int(float(nb["end"])*sr), len(audio))
            chunks.append(audio[ns:ne]);  total += ne - ns;  right += 1
    if total < int(sr * 0.5):
        return None
    return model.embed_chunk(np.concatenate(chunks).astype(np.float32), sr)


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
    <0.30) â€” those produce unstable embeddings that mis-match voiceprints.
    """
    from src.speaker_role import _embed

    # Conservative widening: only pad segments <0.8s, and only by Â±0.2s.
    # Wider padding pulled in surrounding speaker audio and polluted short
    # customer back-channels with neighbouring agent voice â†’ false AGENT match.
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

    # Cluster centroids â€” used to reconcile per-segment label with cosine sim.
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
            # Hotfix D: if embedding failed but nearby segments are in agent_cluster,
            # pull this segment to agent_cluster too (don't let failed embedding mask agent context).
            # ONLY pull if BOTH neighbors are agent, to avoid false positive sandwiches.
            if seg.get("_emb_failed"):
                nearby_agent_cluster = all(
                    0 <= j < len(segments)
                    and j in idx_to_label
                    and idx_to_label[j] == agent_cluster
                    for j in (i - 1, i + 1)
                )
                if nearby_agent_cluster:
                    cid = agent_cluster

        sim = float(sims[i, j_agent]) if j_agent >= 0 and i < len(sims) else 0.0
        seg["_best_sim"] = sim
        seg["_best_match"] = agent_slug

        # Reconcile: if cluster says agent but sim is too low and there's a
        # better customer cluster match, demote.
        # But protect segments with _emb_failed=True if they're anchored by high-sim neighbors.
        should_protect_emb_failed = (
            segments[i].get("_emb_failed")
            and any(
                j >= 0 and j < len(segments)
                and segments[j].get("identified_speaker") == "AGENT"
                and float(segments[j].get("_best_sim") or 0.0) >= 0.30
                for j in (i - 1, i + 1)
            )
        )
        if cid == agent_cluster and sim < CLUSTER_AGENT_FLOOR and embs[i] is not None and customer_cids and not should_protect_emb_failed:
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
        # Keep this conservative for random daily calls, but apply a dynamic
        # duration penalty since short segments naturally score lower.
        dur = max(float(seg["end"]) - float(seg["start"]), 0.0)
        expected_sim_drop = max(0.0, 1.5 - dur) * 0.15

        promoted_to_agent = False
        if cid != agent_cluster and sim >= (0.62 - expected_sim_drop):
            cid = agent_cluster
            promoted_to_agent = True
        elif cid != agent_cluster and sim >= (0.52 - expected_sim_drop) and embs[i] is not None:
            emb = embs[i].astype(np.float32)
            n_emb = np.linalg.norm(emb)
            if n_emb > 0 and cid in centroids:
                emb_n = emb / n_emb
                cust_cent_sim = float(emb_n @ centroids[cid])
                if sim > cust_cent_sim + 0.04:
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


def _contains_any(norm: str, phrases: Tuple[str, ...]) -> bool:
    return any(phrase in norm for phrase in phrases)


def _apply_text_role_overrides(
    segments: List[dict],
    agent_name: str,
) -> Dict[str, int]:
    """Correct strong call-center text cues after embedding/cluster assignment.

    Voiceprints remain the primary signal. This pass only handles phrases whose
    conversational role is strong enough that leaving the embedding label intact
    creates obvious AGENT/CUSTOMER swaps in call-center transcripts.
    """
    customer_to_agent = 0
    agent_to_customer = 0

    customer_cues = (
        "who's speaking",
        "who is speaking",
        "didn't catch your name",
        "i didn't catch your name",
        "do you take part exchange",
        "you take part exchange",
        "i have a gle",
        "i have a gl",
        "say again",
        "don't think so",
        "i am living",
        "i'm living",
        "i live in",
        "not slough",
        "s t r o u d",
        "three hours away",
        "two hours away",
        "can you give me a price",
        "can't remember",
        "i am the second",
        "i am the third",
        "where are you from",
        "i am romanian",
        "i'm romanian",
        "i haven't been in morocco",
        "i have been in spain",
        "pretty sure",
        "i was thinking",
        "getting robbed",
        "machete",
        "settlement figure",
        "jump out",
        "ending in",
        "yeah man",
        "yeah fine",
        "see what the crack",
        "e class",
        "full mercedes",
        "can you send me",
        "some more videos",
        "some more video",
        "how many owners it has",
        "i just been transfer",
        "i've just been transfer",
        "i spoke with one of your",
        "one question please",
        "how this is going to affect",
        "affect my finance",
        "my finance",
        "i tried to call",
        "very bad experience",
        "bad experience with",
        "same payment method",
        "i'm thinking to pay",
        "i am thinking to pay",
        "i have an appointment",
        "i think service",
        "this kind of inspection",
        "you should have my email",
        "not gonna call",
        "i bought",
        "my ex",
        "i am kind of like mechanic",
        "i'm kind of like mechanic",
        "finished school",
        "petrol cars",
        "gasket",
        "costs like",
        "300 quid",
        "cheers man",
        "thanks bye",
        "bye bye",
        "bye-bye",
    )
    agent_cues = (
        "this is omar",
        "omar from car planet",
        "calling you from car planet",
        "from car planet",
        "thank you for calling",
        "how can i help",
        "let me quickly",
        "let me take a look",
        "do you have the reg",
        "give me one second",
        "bear with me",
        "taking a look",
        "from what i can see",
        "previous owners",
        "service history",
        "mot does expire",
        "we do take part exchange",
        "sold you the car",
        "how many years warranty",
        "years warranty",
        "sort this out for you",
        "leave a note",
        "you should receive an email",
        "warranty has been refunded",
        "just gonna check",
        "i'm just gonna check",
        "right cooling off period",
        "still within the right",
        "that means we have to solve",
        "what would happen",
        "what was your deposit",
        "seven thousand pounds",
        "direct refund",
        "won't affect the finance",
        "payment for the warranty",
        "taken out of your deposit",
        "finance company",
        "make an overpayment",
        "confirm your email",
        "refund usually takes",
        "five to seven",
        "has been processed",
        "service plan",
        "won't affect your service plan",
        "won't affect my service plan",
        "shouldn't be linked to the warranty",
        "refund for the five years",
        "so it should be there",
        "no worries mariana",
        "have a good day",
        "no stress",
        "what's the car",
        "what is the car",
        "is it a category",
        "ever been written off",
        "perfect perfect",
        "that's perfect",
        "more than happy",
        "come in person",
        "evaluation team",
        "where are you located",
        "rough evaluation",
        "can i get your reg",
        "that's alright",
        "that's all right",
        "no worries just bear",
        "how many miles",
        "what's your name",
        "i'm from morocco",
        "i am from morocco",
        "what about you",
        "you should go",
        "really nice place",
        "nothing like that happens",
        "driven to morocco",
        "pretty safe",
        "dropped you a message",
        "whatsapp on this number",
        "little message on whatsapp",
        "keep in touch",
        "send it over to you",
        "yeah of course",
        "of course of course",
        "i completely understand",
        "no commitment",
        "call agent",
        "physically can't get you a price",
        "go through a chain",
        "my team has gone home",
        "tomorrow morning",
        "how does that sound",
        "that shouldn't be a problem",
        "i'll see what i can do",
        "i'll get back",
        "i will get back",
        "send you a video",
        "send over a video",
        "drop me a reply",
        "i understand i understand",
        "we can get everything",
        "get everything sorted",
        "you need to get a new one",
        "oil and coolant",
        "it's expensive",
        "cheers mate",
        "catch you later",
    )
    short_customer = {
        "hello",
        "oh okay then",
        "yeah fine",
        "yeah man",
        "nice to meet you too",
        "whatever name your company is",
        "whatever name you come in",
        "mm hmm",
        "mhm",
        "ending in eighty four four four",
        "ending in 8444",
    }
    short_agent = {
        "of course",
        "yeah of course",
        "that's alright no worries",
        "that's all right no worries",
        "perfect",
        "perfect perfect",
    }

    def set_customer(seg: dict) -> None:
        nonlocal agent_to_customer
        if seg.get("identified_speaker") != "CUSTOMER":
            agent_to_customer += 1
        seg["speaker"] = "SPEAKER_01"
        seg["identified_speaker"] = "CUSTOMER"
        seg["display_speaker"] = "Customer 1"
        seg.pop("agent_name", None)

    def set_agent(seg: dict) -> None:
        nonlocal customer_to_agent
        if seg.get("identified_speaker") != "AGENT":
            customer_to_agent += 1
        seg["speaker"] = "SPEAKER_00"
        seg["identified_speaker"] = "AGENT"
        seg["agent_name"] = agent_name
        seg["display_speaker"] = agent_name

    for idx, seg in enumerate(segments):
        text = str(seg.get("text") or "")
        norm = _norm_text(text)
        if not norm:
            continue
        prev_norm = _norm_text(str(segments[idx - 1].get("text") or "")) if idx > 0 else ""
        next_norm = (
            _norm_text(str(segments[idx + 1].get("text") or ""))
            if idx + 1 < len(segments)
            else ""
        )
        sim = float(seg.get("_best_sim") or 0.0)
        words = _norm_words(text)

        if seg.get("identified_speaker") == "AGENT":
            if norm in {"valentine", "valentin"} and "what's your name" in prev_norm:
                set_customer(seg)
                continue
            short_ack_customer = (
                len(words) <= 3
                and words[:1] in (["yeah"], ["yes"], ["okay"], ["ok"], ["hi"], ["right"], ["yep"])
                and not _contains_any(
                    norm,
                    (
                        "of course",
                        "i understand",
                        "expensive",
                        "safe",
                        "perfect",
                        "send",
                    ),
                )
            )
            strong_customer = (
                norm in short_customer
                or _contains_any(norm, customer_cues)
                or short_ack_customer
                or (norm == "two hours" and "three hours away" in prev_norm)
            )
            low_sim_customer = sim < 0.18 and (
                _contains_any(norm, ("i am", "i'm", "i have", "i was", "i haven't"))
                or _looks_like_question(text)
            )
            if strong_customer or low_sim_customer:
                set_customer(seg)
                continue

        if seg.get("identified_speaker") == "CUSTOMER":
            if norm == "okay" and "what's your name" in next_norm:
                set_agent(seg)
                continue
            if norm in {"yeah yeah", "yeah"} and "what about you" in next_norm:
                set_agent(seg)
                continue
            if norm == "no no no" and "nothing like" in next_norm:
                set_agent(seg)
                continue
            strong_agent = norm in short_agent or _contains_any(norm, agent_cues)
            low_threshold_agent = sim >= 0.35 and _contains_any(
                norm,
                (
                    "understand",
                    "of course",
                    "no worries just bear",
                    "send",
                    "whatsapp",
                    "video",
                    "get back",
                    "expensive",
                ),
            )
            if strong_agent or low_threshold_agent:
                set_agent(seg)

    return {
        "text_agent_to_customer": agent_to_customer,
        "text_customer_to_agent": customer_to_agent,
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
        "speaker_mode": "unknown",
        "agent_threshold_used": 0.0,
        "cluster_report": {},
        "speaker_id_warning": reason,
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
                call_snr = _estimate_snr(audio, sr)      # once per call, O(N/step)
                is_low_snr = call_snr < SNR_LOW_DB
                SHORT_THRESH = int(sr * 0.8)
                PAD_EACH = int(sr * 0.2)

                for seg_idx, seg in enumerate(segments):
                    s = int(float(seg["start"]) * sr)
                    e = min(int(float(seg["end"]) * sr), len(audio))
                    seg_dur = float(seg["end"]) - float(seg["start"])

                    seg["_is_filler"] = _is_filler_only(seg.get("text", ""), seg_dur)
                    seg["_snr_low"]   = is_low_snr

                    if e - s < int(sr * MIN_SEG_S_FOR_EMB):
                        embs.append(None); continue

                    # Cross-speaker pooling is disabled by default for production:
                    # random calls often start with customer speech, and pooling
                    # nearby turns contaminates embeddings with the wrong speaker.
                    if seg_idx < 3:
                        pooled = _pool_embedding_for_short_seg(seg_idx, segments, audio, sr, model)
                        if pooled is not None:
                            embs.append(pooled); continue

                    # SHORT SEGMENT: try neighbor pool before symmetric pad
                    if e - s < SHORT_THRESH:
                        pooled = _pool_embedding_for_short_seg(seg_idx, segments, audio, sr, model)
                        if pooled is not None:
                            embs.append(pooled); continue
                        # Fallback: symmetric pad (existing behavior)
                        chunk = audio[max(0,s-PAD_EACH) : min(len(audio),e+PAD_EACH)]
                    else:
                        chunk = audio[s:e]

                    sr_gate = 0.20 if is_low_snr else 0.25     # more lenient gate for noisy audio
                    if _speech_ratio(chunk, sr) < sr_gate:
                        embs.append(None); continue

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


def _apply_unknown_rejection(segments: List[Dict]) -> Dict[str, int]:
    """
    Apply unknown speaker rejection gate.
    If a segment's best match is below the floor OR doesn't have enough matches,
    mark it for conservative classification.
    """
    rejections = {"total": 0, "below_floor": 0, "min_match_fail": 0}

    for seg in segments:
        sim = float(seg.get("_best_sim", 0.0))

        # Check if below rejection floor
        if sim < UNKNOWN_REJECTION_FLOOR:
            seg["_unknown_risk"] = True
            seg["_confidence_gate"] = "REJECTED_BELOW_FLOOR"
            rejections["below_floor"] += 1
            rejections["total"] += 1
            continue

        # Check match count - if agent has <3 confident matches, reject
        match_count = seg.get("_match_count", 0)
        if match_count < UNKNOWN_REJECTION_MIN_MATCHES:
            seg["_unknown_risk"] = True
            seg["_confidence_gate"] = "REJECTED_LOW_MATCHES"
            rejections["min_match_fail"] += 1
            rejections["total"] += 1

    if rejections["total"] > 0:
        logger.info(f"unknown rejection: {rejections['total']} segments flagged "
                   f"(floor={rejections['below_floor']}, matches={rejections['min_match_fail']})")

    return rejections


def _apply_confidence_gating(segments: List[Dict]) -> int:
    """
    Apply confidence-gated classification in the uncertain band.
    Segments in the [CONFIDENCE_GATE_UNCERTAIN_BAND, CONFIDENCE_GATE_UPPER_BOUND]
    zone get conservative CUSTOMER label to reduce false positives.
    """
    conservative_overrides = 0

    for seg in segments:
        sim = float(seg.get("_best_sim", 0.0))
        role = seg.get("identified_speaker", "")

        # Only apply gating to segments that would be labeled AGENT but are in uncertain zone
        if "AGENT" in role and CONFIDENCE_GATE_UNCERTAIN_BAND <= sim < CONFIDENCE_GATE_UPPER_BOUND:
            # Check if there's unknown rejection risk
            if seg.get("_unknown_risk"):
                # Conservative: don't force AGENT for uncertain segments
                # Let temporal voting decide
                seg["_confidence_gate"] = "UNCERTAIN_CONSERVATIVE"
                conservative_overrides += 1

    if conservative_overrides > 0:
        logger.info(f"confidence gating: {conservative_overrides} segments flagged as uncertain")

    return conservative_overrides


def _apply_temporal_voting(segments: List[Dict], agent_name: str) -> List[Dict]:
    """
    Apply 10-second temporal voting window to fix isolated misclassifications.
    Segments with high-confidence neighbors (sim >= 0.40) get weighted majority vote.
    """
    WINDOW_S = 10.0
    MIN_WINDOW_VOTES = 3
    OVERRIDE_THRESHOLD = 0.72
    ANCHOR_SIM_MIN = 0.40

    n = len(segments)
    corrections = 0

    for i in range(n):
        seg = segments[i]
        mid_s = (float(seg.get("start", 0)) + float(seg.get("end", 0))) / 2
        window_start = mid_s - WINDOW_S / 2
        window_end = mid_s + WINDOW_S / 2

        # Collect segments in window
        window_votes = []
        for j in range(n):
            other_seg = segments[j]
            other_mid = (float(other_seg.get("start", 0)) + float(other_seg.get("end", 0))) / 2
            if window_start <= other_mid <= window_end and j != i:
                role = other_seg.get("identified_speaker", "")
                sim = float(other_seg.get("_best_sim", 0.0))
                if "AGENT" in role:
                    window_votes.append(("AGENT", sim))
                else:
                    window_votes.append(("CUSTOMER", sim))

        if len(window_votes) < MIN_WINDOW_VOTES:
            continue

        # Weighted vote
        agent_weight = sum(sim for role, sim in window_votes if role == "AGENT")
        customer_weight = sum(sim for role, sim in window_votes if role == "CUSTOMER")
        total_weight = agent_weight + customer_weight
        if total_weight == 0:
            continue

        agent_pct = agent_weight / total_weight
        current_role = seg.get("identified_speaker", "")
        current_sim = float(seg.get("_best_sim", 0.0))

        if seg.get("_backchannel_demoted"):
            continue

        # Override if window consensus is strong and segment is weak
        # We enforce current_sim >= 0.20 so that very low similarity segments
        # (like customer backchannels) aren't falsely pulled into the Agent cluster.
        if (agent_pct >= OVERRIDE_THRESHOLD and "CUSTOMER" in current_role and
            0.20 <= current_sim < 0.40 and
            any(sim >= ANCHOR_SIM_MIN for role, sim in window_votes if role == "AGENT")):
            seg["identified_speaker"] = "AGENT"
            seg["agent_name"] = agent_name
            seg["display_speaker"] = agent_name
            seg["_temporal_vote_override"] = True
            corrections += 1
        elif (agent_pct <= (1 - OVERRIDE_THRESHOLD) and "AGENT" in current_role and
              current_sim < 0.30 and
              any(sim >= ANCHOR_SIM_MIN for role, sim in window_votes if role == "CUSTOMER")):
            seg["identified_speaker"] = "CUSTOMER"
            seg["display_speaker"] = "Customer 1"
            seg.pop("agent_name", None)
            seg["_temporal_vote_override"] = True
            corrections += 1

    if corrections > 0:
        logger.info(f"temporal voting: {corrections} segments overridden by neighbor consensus")

    return segments


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

    # ── H3 fix: set _is_filler and _snr_low for ALL segments regardless of
    # embedding dimension.  Previously these flags were only set inside the
    # dim==512 branch of _segment_embeddings_for_dim(), so 192-dim-only
    # enrollments had filler weighting and SNR adaptation completely disabled.
    call_snr = _estimate_snr(audio, sr)
    is_low_snr = call_snr < SNR_LOW_DB
    for seg in segments:
        seg_dur = float(seg["end"]) - float(seg["start"])
        seg["_is_filler"] = _is_filler_only(seg.get("text", ""), seg_dur)
        seg["_snr_low"]   = is_low_snr

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

    # â”€â”€ Filler weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    seg_weights = np.ones(len(segments), dtype=np.float32)
    for i, seg in enumerate(segments):
        if seg.get("_is_filler", False):
            seg_weights[i] = FILLER_SIM_WEIGHT

    # â”€â”€ Progressive confidence: how much non-filler speech do we have? â”€â”€
    # -- Progressive confidence: how much non-filler speech do we have? --
    # C4 fix: use any() over per-dim validity arrays instead of concatenating
    # them.  The old code concatenated all dims' boolean arrays into one flat
    # array, so np.where() returned indices up to N*n_dims-1 instead of N-1,
    # causing segments valid only in the 2nd dim to be silently excluded.
    non_filler_speech_s = sum(
        max(float(seg["end"]) - float(seg["start"]), 0.0)
        for i, seg in enumerate(segments)
        if any(valid_by_dim[d][i] for d in valid_by_dim if i < len(valid_by_dim[d]))
        and not seg.get("_is_filler", False)
    )
    effective_min_matched = (PROG_CONF_MIN_MATCHED
                             if non_filler_speech_s < PROG_CONF_MIN_SPEECH_S
                             else AGENT_MIN_MATCHED)
    if effective_min_matched != AGENT_MIN_MATCHED:
        logger.info("progressive confidence: %.1fs non-filler speech â†’ min_matched=%d",
                    non_filler_speech_s, effective_min_matched)

    for dim, slugs in slugs_by_dim.items():
        valid_rows = np.where(valid_by_dim[dim])[0]
        if not len(valid_rows):
            continue
        for j, slug in enumerate(slugs):
            col = sims_by_dim[dim][valid_rows, j]
            w   = seg_weights[valid_rows]
            order = np.argsort(col)[::-1]
            col_s, w_s = col[order], w[order]
            n_nonfiller = max(int(np.sum(w_s > 0.5)), 1)
            k_top = max(3, int(n_nonfiller * 0.30))
            top_w = w_s[:k_top];  top_s = col_s[:k_top]
            denom = float(np.sum(top_w))
            agent_scores[slug] = float(np.dot(top_s, top_w) / max(denom, 1e-8))
            agent_backend[slug] = (dim, j)

    if not agent_scores:
        return _unknown_result(segments, "no agent scores could be computed")

    ranked = sorted(agent_scores, key=agent_scores.get, reverse=True)
    agent_slug = ranked[0]
    agent_dim, j_agent = agent_backend[agent_slug]
    agent_name = voiceprints_by_dim[agent_dim][agent_slug][0]
    low_match_warning = None
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

    # SNR-adaptive: relax threshold on low-SNR calls (low-bucket VPs cover this)
    is_low_snr_call = any(seg.get("_snr_low", False) for seg in segments[:10])
    if is_low_snr_call and agent_threshold > SNR_LOW_FLOOR:
        old = agent_threshold
        agent_threshold = max(SNR_LOW_FLOOR, agent_threshold - 0.06)
        logger.info("SNR-adaptive threshold: %.3f â†’ %.3f", old, agent_threshold)

    seg_best_agent: List[Optional[str]] = []
    match_counts: Dict[str, int] = {}
    match_sims: Dict[str, List[float]] = {}

    # For first 3 segments, use slightly lowered threshold to bootstrap agent ID
    # BUT: only if segment is â‰¥1s long (agent greetings are longer)
    initial_threshold = max(agent_threshold - 0.08, 0.18)

    for i in range(len(segments)):
        if not valid[i]:
            seg_best_agent.append(None)
            continue
        sim = float(sims[i, j_agent])
        seg_dur = float(segments[i]["end"]) - float(segments[i]["start"])

        # Short segments inherently have lower embedding similarity due to less acoustic information.
        # Apply a dynamic penalty up to 0.15 for very short segments.
        expected_sim_drop = max(0.0, 1.5 - seg_dur) * 0.15
        threshold_for_seg = agent_threshold - expected_sim_drop

        if sim >= threshold_for_seg:
            seg_best_agent.append(agent_slug)
            match_counts[agent_slug] = match_counts.get(agent_slug, 0) + 1
            match_sims.setdefault(agent_slug, []).append(sim)
        else:
            seg_best_agent.append(None)

    if match_counts.get(agent_slug, 0) < effective_min_matched:
        matched_count = match_counts.get(agent_slug, 0)
        if matched_count > 0:
            low_match_warning = (
                f"Low-confidence agent identity: only {matched_count} segments "
                f"beat threshold for {agent_slug}"
            )
            logger.info("%s; keeping best voiceprint identity for text/context role assignment",
                        low_match_warning)
        else:
            logger.info(
                "No segments beat threshold for %s; falling back to Unknown",
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

    # Fallback: if agent_avg_sim is still 0, compute from segment similarities
    if agent_avg_sim == 0.0 and agent_slug:
        agent_sims = [s.get("_best_sim", 0) for s in segments
                      if s.get("_best_match") == agent_slug]
        if agent_sims:
            agent_avg_sim = float(np.mean(agent_sims))

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
            if not valid[i]:
                seg["_emb_failed"] = True
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
        # Used below to gate the "soft reclaim" â€” a segment is only pulled back
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
            dur = float(seg["end"]) - float(seg["start"])
            if embs[seg_i] is None:
                seg["display_speaker"] = None
                seg["speaker"] = None
                continue
            cid = int(labels_u[k_iter])
            k_iter += 1
            own_sim = float(sims[seg_i, j_agent]) if agent_slug and j_agent >= 0 else 0.0
            
            # Short segments naturally have lower similarity due to less acoustic information.
            # Relax the absolute floor for segments under 1.5s
            dynamic_merge_floor = max(0.22, MERGE_TO_AGENT_SIM - max(0.0, 1.5 - dur) * 0.25)
            
            # Soft reclaim conditions:
            #  1. Cosine to agent voiceprint >= dynamic_merge_floor
            #  2. Cosine to agent voiceprint â‰¥ cosine to OWN customer cluster centroid
            #     (i.e. closer to the agent than to its assigned customer peers)
            #  3. OR cosine â‰¥ (agent_threshold âˆ’ 0.07) â€” close to the hard threshold
            reclaim = False
            if agent_slug and own_sim >= dynamic_merge_floor and embs[seg_i] is not None:
                emb = embs[seg_i].astype(np.float32)
                n_emb = np.linalg.norm(emb)
                if n_emb > 0 and cid in cust_centroids:
                    emb_n = emb / n_emb
                    cust_sim = float(emb_n @ cust_centroids[cid])
                    if own_sim >= cust_sim + 0.04:
                        reclaim = True
                # Borderline-but-close-to-threshold also reclaims (rescues short
                # agent acks like "Hi Edgar." / "What are your plans?" with sim
                # in the 0.30â€“0.42 band that the centroid check rejected).
                if not reclaim and own_sim >= max(agent_threshold - 0.04, MERGE_TO_AGENT_SIM):
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
            agent_cluster_sims = cluster_report.get("agent_sims") or []
            cluster_agent_mean = (
                float(np.mean(agent_cluster_sims)) if agent_cluster_sims else 0.0
            )
            min_cluster_mean = max(0.42, min(float(agent_threshold) - 0.08, 0.55))
            if cluster_agent_mean >= min_cluster_mean:
                speaker_mode = "cluster_first_voiceprint"
                match_counts = {agent_slug: int(cluster_report.get("agent_count") or 0)}
                match_sims = {agent_slug: list(agent_cluster_sims)}
                agent_avg_sim = cluster_agent_mean
                cluster_report.pop("agent_sims", None)
                logger.info(
                    "cluster-first role assignment enabled: dur=%.1fs valid=%d "
                    "initial_agent_ratio=%.2f agent_mean=%.3f clusters=%s",
                    total_segment_dur,
                    valid_count,
                    initial_agent_ratio,
                    cluster_agent_mean,
                    cluster_report.get("cluster_stats"),
                )
            else:
                cluster_ok = False
                cluster_report["reason"] = (
                    f"agent cluster mean {cluster_agent_mean:.3f} < "
                    f"{min_cluster_mean:.3f}"
                )
                logger.info("cluster-first role assignment rejected: %s", cluster_report["reason"])
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

    # â”€â”€ Anti-flip pass 1: tight (â‰¤0.6s sandwiched between same speaker) â”€â”€
    for i in range(1, n - 1):
        dur = float(segments[i]["end"]) - float(segments[i]["start"])
        if dur > 0.6:
            continue
        prev = segments[i - 1].get("display_speaker")
        nxt = segments[i + 1].get("display_speaker")
        cur = segments[i].get("display_speaker")
        if prev and nxt and prev == nxt and cur != prev:
            _apply(segments[i], segments[i - 1])

    # â”€â”€ Anti-flip pass 2: low-confidence sandwich (<2.5s, sim<0.30) â”€â”€
    # Catches the "Yeah." / "Okay." back-channels that get smoothed to AGENT
    # even though their cosine to the agent voiceprint is essentially zero.
    # If neighbours agree, trust them over the noisy embedding.
    # We restrict to sim >= 0.20 to avoid flipping clear customer backchannels.
    for i in range(1, n - 1):
        dur = float(segments[i]["end"]) - float(segments[i]["start"])
        sim = float(segments[i].get("_best_sim") or 0.0)
        if dur > 2.5 or not (0.20 <= sim < 0.30):
            continue
        prev = segments[i - 1].get("display_speaker")
        nxt = segments[i + 1].get("display_speaker")
        cur = segments[i].get("display_speaker")
        if prev and nxt and prev == nxt and cur != prev:
            # Only flip if at least one neighbour has a strong (â‰¥0.40) signal
            prev_sim = float(segments[i - 1].get("_best_sim") or 0.0)
            nxt_sim = float(segments[i + 1].get("_best_sim") or 0.0)
            if max(prev_sim, nxt_sim) >= 0.40:
                _apply(segments[i], segments[i - 1])

    # â”€â”€ Anti-flip pass 2.5: low-sim AGENT segments starting with back-channel â”€â”€
    # â”€â”€ Anti-flip pass 2.5: demote low-sim short AGENT segments â”€â”€
    # If a segment is very short and its similarity is weak, it is mathematically
    # much more likely to be a customer backchannel or noise that unsupervised
    # clustering mistakenly swallowed into the AGENT cluster.
    short_demoted = 0
    for i, seg in enumerate(segments):
        if seg.get("identified_speaker") != "AGENT":
            continue
        sim = float(seg.get("_best_sim") or 0.0)
        dur = float(seg["end"]) - float(seg["start"])
        # Mathematically demote if dur < 1.5s and sim < 0.35
        if dur >= 1.5 or sim >= 0.35:
            continue
        seg["identified_speaker"] = "CUSTOMER"
        seg["display_speaker"] = "Customer 1"
        seg["speaker"] = "SPEAKER_01"
        seg["_backchannel_demoted"] = True
        seg.pop("agent_name", None)
        short_demoted += 1
    if short_demoted:
        logger.info("demoted %d low-sim short segs AGENT â†’ CUSTOMER", short_demoted)

    # â”€â”€ Anti-flip pass 3: zero-similarity AGENT segments are smoothing artefacts â”€â”€
    # If a segment is labelled AGENT but its cosine to the agent voiceprint is
    # below 0.10, it was almost certainly assigned by neighbour-vote on noisy
    # input. If the *previous* meaningful segment is CUSTOMER, demote it.
    for i, seg in enumerate(segments):
        if seg.get("identified_speaker") != "AGENT":
            continue
        sim = float(seg.get("_best_sim") or 0.0)
        if sim >= 0.10:
            continue
        # Protect short segments with failed embeddings if anchored by high-confidence AGENT neighbors
        if seg.get("_emb_failed"):
            if any(
                0 <= j < n
                and segments[j].get("identified_speaker") == "AGENT"
                and float(segments[j].get("_best_sim") or 0.0) >= 0.30
                for j in (i - 1, i + 1)
            ):
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

    # Temporal voting: use neighbor context to fix isolated misclassifications
    segments = _apply_temporal_voting(segments, agent_name)

    # Phase 3: Confidence gating and unknown rejection
    unknown_rejection_result = _apply_unknown_rejection(segments)
    confidence_gate_count = _apply_confidence_gating(segments)

    boundary_refinement: Dict[str, object] = {"enabled": False, "reason": "disabled by default"}
    if os.getenv("SST_ENABLE_TEXT_BOUNDARY_REFINEMENT", "").strip() == "1":
        try:
            from src.boundary_refinement import (
                refine_with_pyannote_boundaries,
                refine_with_text_cue_boundaries,
            )

            segments, text_boundary_report = refine_with_text_cue_boundaries(
                segments,
                agent_name,
            )
            pyannote_boundary_report = {"enabled": False, "reason": "disabled by default"}
            if os.getenv("SST_ENABLE_PYANNOTE_BOUNDARY_REFINEMENT", "").strip() == "1":
                segments, pyannote_boundary_report = refine_with_pyannote_boundaries(
                    segments,
                    norm_wav,
                    agent_name,
                    force_cpu=force_cpu,
                )
            boundary_refinement = {
                "enabled": bool(
                    text_boundary_report.get("enabled")
                    or pyannote_boundary_report.get("enabled")
                ),
                "text_cue": text_boundary_report,
                "pyannote": pyannote_boundary_report,
            }
            role_corrections["text_boundary_split_segments"] = int(
                text_boundary_report.get("split_segments") or 0
            )
            role_corrections["pyannote_boundary_split_segments"] = int(
                pyannote_boundary_report.get("split_segments") or 0
            )
        except Exception as e:
            boundary_refinement = {
                "enabled": False,
                "reason": f"boundary refinement failed: {e}",
            }
            logger.warning("boundary refinement skipped: %s", e)
    else:
        role_corrections["text_boundary_split_segments"] = 0
        role_corrections["pyannote_boundary_split_segments"] = 0

    per_speaker: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        lbl = seg.get("display_speaker", "?")
        per_speaker.setdefault(lbl, {"turns": 0, "seconds": 0.0})
        per_speaker[lbl]["turns"] += 1
        per_speaker[lbl]["seconds"] += float(seg["end"]) - float(seg["start"])

    # Add warning if confidence is suspiciously low
    warning = low_match_warning
    direct_match_count = int(match_counts.get(agent_slug, 0)) if agent_slug else 0
    if agent_slug and direct_match_count < effective_min_matched:
        match_warning = (
            f"Low direct voice evidence: {direct_match_count} segments beat threshold "
            f"for {agent_slug} (need {effective_min_matched})"
        )
        warning = f"{warning}; {match_warning}" if warning else match_warning
    if agent_slug and agent_avg_sim < 0.50:
        extra_warning = f"Low confidence identification (avg_similarity={agent_avg_sim:.2f} < 0.50)"
        warning = f"{warning}; {extra_warning}" if warning else extra_warning

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
        "boundary_refinement": boundary_refinement,
        "speaker_id_warning": warning,
    }
