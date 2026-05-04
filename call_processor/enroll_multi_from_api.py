"""
enroll_multi_from_api.py — Train **multiple** voiceprints per agent (low/mid/high
acoustic-quality buckets) so the matcher has a closer reference for noisy desk
recordings than a single averaged centroid can provide.

Pipeline (per agent):
  1. Pull recordings via API + payload (already cached in
     data/audiofy/_dataset/audio/ from scrape_dataset_api.py).
  2. For each call:
       - Slice agent-labeled phrases from API speaker_json (ground truth).
       - Estimate the call's SNR from the agent-only audio:
         snr = 10*log10(p90_rms / p10_rms) over 50 ms frames.
       - Bucket the call into HIGH (>= 15 dB), MID (8-15), LOW (< 8).
       - Sliding ECAPA windows -> tagged embeddings.
  3. Per (agent, bucket) with >= MIN_BUCKET_EMBS windows:
       - L2-normalise, run iterative_tighten() to drop outliers (kills any
         residual customer leakage).
       - K-means with k = min(2, n_kept // 30); save each centroid as one
         voiceprint .npy.
  4. Also save the legacy mean centroid (over the HIGH bucket only, falling
     back to all kept embeddings if HIGH is empty) so single-VP code keeps
     working.
  5. Update data/agent_voiceprints/agents.json with a `voiceprints` list per
     agent; legacy `voiceprint_path` still points at the mean.

Usage:
  python enroll_multi_from_api.py
  python enroll_multi_from_api.py --min-calls 5 --max-calls-per-agent 5
  python enroll_multi_from_api.py --agents "Omar" "Haris"
  python enroll_multi_from_api.py --keep-existing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent
os.chdir(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Reuse the existing enrollment helpers verbatim
from enroll_all_from_api import (  # type: ignore
    INDEX_PATH, AUDIO_DIR, VP_DIR, AGENTS_JSON,
    TARGET_SR, WINDOW_S, STRIDE_S,
    MIN_AGENT_TOTAL_S, MIN_TIGHT_N,
    slug, download_batch, load_mp3_mono_16k,
    extract_agent_samples, extract_embeddings,
    iterative_tighten, _centroid,
)

# ---- bucketing thresholds ---------------------------------------------------
# Tuned for SNR estimated from the *full untrimmed* call audio (silence gaps
# reveal the real noise floor). Clean phone calls land 22-32 dB; noisy desk
# recordings land 8-16 dB.
HIGH_SNR_DB     = 20.0
MID_SNR_DB      = 12.0
MIN_BUCKET_EMBS = 20      # minimum windows before a bucket gets its own VP
MAX_K_PER_BUCKET = 2      # at most 2 cluster centroids per bucket
KMEANS_RESEED_TRIES = 4

# Frame size for SNR estimate (50 ms)
SNR_FRAME_S = 0.050


def estimate_snr_db(samples: np.ndarray, sr: int = TARGET_SR) -> float:
    """Approximate clarity-SNR in dB.

    The earlier ``p90 / p10`` over the whole recording broke down on desk
    audio: laptop mics record near-zero floor during silence (no transmission
    hiss the way a phone call has), so noisy desk recordings paradoxically
    scored higher than clean phone audio.

    Instead, measure the dynamic range *within speech-only frames*:
        - speech frames = top 50 % by RMS (filters out silences and pauses).
        - within those: p95 / p20 RMS ratio.
    Clean speech has a wide vowel/consonant gap (~25-35 dB); noisy speech
    has the quiet end lifted by background hum, compressing the ratio to
    8-15 dB. Maps cleanly onto our high/mid/low buckets.
    """
    if samples is None or len(samples) < int(sr * SNR_FRAME_S * 4):
        return 0.0
    frame = int(sr * SNR_FRAME_S)
    n_frames = len(samples) // frame
    if n_frames < 8:
        return 0.0
    framed = samples[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1) + 1e-12)

    # Keep only the louder half — those are speech-bearing frames.
    speech_thresh = float(np.percentile(rms, 50))
    speech_rms = rms[rms >= speech_thresh]
    if speech_rms.size < 4:
        return 0.0
    high = float(np.percentile(speech_rms, 95))
    low  = float(np.percentile(speech_rms, 20))
    if low <= 1e-6:
        low = 1e-6
    return float(20.0 * np.log10(high / low))


def bucket_for(snr_db: float) -> str:
    if snr_db >= HIGH_SNR_DB:
        return "high"
    if snr_db >= MID_SNR_DB:
        return "mid"
    return "low"


def kmeans_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """Cosine k-means on L2-normalised rows. Returns (k, dim) centroids.

    Uses k-means++ init and a few re-seeds; keeps the run with the lowest
    within-cluster cosine distance sum. Falls back to a single centroid if
    sklearn is unavailable.
    """
    if k <= 1 or len(X) <= k:
        return _centroid(X)[None, :]
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return _centroid(X)[None, :]

    best_inertia = np.inf
    best_centers = None
    for seed in range(KMEANS_RESEED_TRIES):
        km = KMeans(n_clusters=k, n_init=4, random_state=seed)
        labels = km.fit_predict(X)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            # Re-derive centroids as L2-normalised cluster means (cosine space)
            centers = []
            for cid in range(k):
                mask = labels == cid
                if not mask.any():
                    continue
                centers.append(_centroid(X[mask]))
            if centers:
                best_centers = np.stack(centers)
    if best_centers is None:
        return _centroid(X)[None, :]
    return best_centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-calls", type=int, default=5)
    ap.add_argument("--max-calls-per-agent", type=int, default=5)
    ap.add_argument("--agents", nargs="*", default=None)
    ap.add_argument("--keep-existing", action="store_true",
                    help="Skip agents that already have a `voiceprints` list")
    args = ap.parse_args()

    VP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_PATH.exists():
        sys.exit(f"[ERROR] {INDEX_PATH} not found — run "
                 f"tools/legacy/scrape_dataset_api.py first")

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    by_agent: Dict[str, list] = {}
    for rec in index:
        by_agent.setdefault(rec["agent_name"], []).append(rec)

    selected: Dict[str, list] = {}
    for name, recs in by_agent.items():
        if args.agents and not any(a.lower() in name.lower() for a in args.agents):
            continue
        if len(recs) < args.min_calls:
            continue
        selected[name] = recs[: args.max_calls_per_agent]

    print(f"[multi-vp] {len(selected)} agents selected "
          f"(min-calls={args.min_calls})", flush=True)
    for name, recs in sorted(selected.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(recs):3d}  {name}", flush=True)
    if not selected:
        sys.exit("[multi-vp] no agents matched")

    if AGENTS_JSON.exists():
        with open(AGENTS_JSON, encoding="utf-8") as f:
            agents = json.load(f)
    else:
        agents = {}

    # Pre-download all needed MP3s in parallel
    tasks = []
    for name, recs in selected.items():
        if args.keep_existing:
            existing = agents.get(slug(name), {})
            if isinstance(existing.get("voiceprints"), list) and existing["voiceprints"]:
                continue
        for rec in recs:
            tasks.append((rec["horizon_s3"], AUDIO_DIR / f"{rec['_id']}.mp3"))
    print(f"\n[multi-vp] downloading {len(tasks)} MP3s in parallel ...",
          flush=True)
    download_batch(tasks, workers=8)

    # Load embedding model (CAM++ preferred, ECAPA-TDNN fallback)
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
        print(f"[multi-vp] {model.model_name} ready (dim={model.dim})", flush=True)
    except Exception as e:
        print(f"[multi-vp] GPU load failed ({e}), using CPU", flush=True)
        model.load(force_cpu=True)
        print(f"[multi-vp] {model.model_name} ready on CPU (dim={model.dim})",
              flush=True)

    try:
        for name, recs in sorted(selected.items(), key=lambda kv: -len(kv[1])):
            agent_slug = slug(name)
            existing = agents.get(agent_slug, {})
            if args.keep_existing and isinstance(existing.get("voiceprints"),
                                                  list) and existing["voiceprints"]:
                print(f"\n[multi-vp] {name}: already has {len(existing['voiceprints'])} "
                      f"voiceprints, skipping", flush=True)
                continue

            print(f"\n[multi-vp] === {name}  ({len(recs)} calls) ===", flush=True)

            # bucket -> list of np.ndarray embeddings
            embs_by_bucket: Dict[str, List[np.ndarray]] = {
                "high": [], "mid": [], "low": []}
            total_agent_s = 0.0
            used_calls = 0
            per_call_snr: List[Tuple[str, float, str, int]] = []

            for rec in recs:
                rid = rec["_id"]
                mp3 = AUDIO_DIR / f"{rid}.mp3"
                if not (mp3.exists() and mp3.stat().st_size > 1000):
                    print(f"  [skip {rid[:8]}] no MP3", flush=True)
                    continue

                agent_segs = [s for s in rec["speaker_json"]
                              if isinstance(s, dict)
                              and s.get("speaker") and s["speaker"] != "Customer"]
                if not agent_segs:
                    print(f"  [skip {rid[:8]}] no agent phrases", flush=True)
                    continue

                t0 = time.time()
                try:
                    audio, sr = load_mp3_mono_16k(mp3)
                except Exception as e:
                    print(f"  [skip {rid[:8]}] decode failed: {e}", flush=True)
                    continue

                # Estimate SNR on the FULL untrimmed call (the long silences
                # between phrases are what makes the 10th-percentile RMS a
                # real noise-floor reading instead of phoneme dynamic range).
                snr_db = estimate_snr_db(audio, sr)
                bkt = bucket_for(snr_db)

                samples, dur = extract_agent_samples(audio, sr, agent_segs)
                if dur < 3.0:
                    print(f"  [skip {rid[:8]}] only {dur:.1f}s agent audio",
                          flush=True)
                    continue

                embs = extract_embeddings(samples, model)
                if len(embs) < 3:
                    print(f"  [skip {rid[:8]}] only {len(embs)} embs",
                          flush=True)
                    continue

                embs_by_bucket[bkt].extend(embs)
                total_agent_s += dur
                used_calls += 1
                per_call_snr.append((rid, snr_db, bkt, len(embs)))
                print(f"  [ok {rid[:8]}] snr={snr_db:5.1f}dB ({bkt:>4}) "
                      f"{dur:.0f}s -> {len(embs)} embs  "
                      f"({time.time()-t0:.1f}s)", flush=True)

            n_total = sum(len(v) for v in embs_by_bucket.values())
            if n_total < 30 or total_agent_s < MIN_AGENT_TOTAL_S:
                print(f"  [multi-vp] {name}: only {n_total} embs / "
                      f"{total_agent_s:.0f}s — INSUFFICIENT, skipping",
                      flush=True)
                continue

            # Per-bucket processing
            saved_vps: List[Dict] = []
            kept_for_legacy: List[np.ndarray] = []

            for bkt in ("high", "mid", "low"):
                bucket_embs = embs_by_bucket[bkt]
                if len(bucket_embs) < MIN_BUCKET_EMBS:
                    if bucket_embs:
                        print(f"  [bucket {bkt}] only {len(bucket_embs)} embs "
                              f"(< {MIN_BUCKET_EMBS}) — folded into nearest bucket",
                              flush=True)
                    continue

                X = np.stack(bucket_embs).astype(np.float32)
                norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1
                X = X / norms

                # Tighten this bucket on its own (dropping any residual customer)
                keep, tight_centroid, inside, outside = iterative_tighten(X)
                n_kept = int(keep.sum())
                if n_kept < MIN_TIGHT_N:
                    keep = np.ones(len(X), dtype=bool)
                    n_kept = len(X)
                X_kept = X[keep]

                # Cluster within bucket
                k = max(1, min(MAX_K_PER_BUCKET, n_kept // 30))
                centroids = kmeans_centroids(X_kept, k)

                bucket_snr_med = float(np.median(
                    [s for (_, s, b, _) in per_call_snr if b == bkt]) or 0.0)

                for idx, c in enumerate(centroids):
                    vp_path = VP_DIR / f"{agent_slug}__{bkt}_{idx}.npy"
                    np.save(vp_path, c.astype(np.float32))
                    saved_vps.append({
                        "path":    str(vp_path),
                        "bucket":  bkt,
                        "n_clips": int(n_kept // len(centroids)),
                        "snr_db":  round(bucket_snr_med, 1),
                    })

                # The HIGH bucket feeds the legacy single-VP mean
                if bkt == "high":
                    kept_for_legacy.append(X_kept)

                print(f"  [bucket {bkt}] {n_kept} embs (snr~{bucket_snr_med:.1f}dB) "
                      f"-> {len(centroids)} centroid(s), inside={inside:.3f}, "
                      f"outside_max={outside:.3f}", flush=True)

            if not saved_vps:
                print(f"  [multi-vp] {name}: no bucket reached the minimum, "
                      f"skipping", flush=True)
                continue

            # Legacy mean — prefer HIGH-only, fall back to all kept embeddings
            if kept_for_legacy:
                legacy_mean = _centroid(np.concatenate(kept_for_legacy))
            else:
                all_kept = []
                for bkt_embs in embs_by_bucket.values():
                    if not bkt_embs:
                        continue
                    X = np.stack(bkt_embs).astype(np.float32)
                    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1
                    all_kept.append(X / norms)
                legacy_mean = _centroid(np.concatenate(all_kept))

            legacy_path = VP_DIR / f"{agent_slug}.npy"
            np.save(legacy_path, legacy_mean.astype(np.float32))

            agents[agent_slug] = {
                "agent_name":      name,
                "voiceprint_path": str(legacy_path),
                "voiceprints":     saved_vps,
                "n_voiceprints":   len(saved_vps),
                "total_seconds":   round(total_agent_s, 1),
                "used_calls":      used_calls,
                "source":          "multi_vp_v1",
                "per_call_snr":    [
                    {"_id": rid, "snr_db": round(s, 1), "bucket": b, "embs": n}
                    for (rid, s, b, n) in per_call_snr
                ],
            }
            print(f"  [multi-vp] {name}: saved {len(saved_vps)} centroids "
                  f"({total_agent_s:.0f}s, {used_calls} calls)", flush=True)

            # Checkpoint after each agent so a crash doesn't lose work
            with open(AGENTS_JSON, "w", encoding="utf-8") as f:
                json.dump(agents, f, ensure_ascii=False, indent=2)
    finally:
        model.unload()

    print(f"\n[multi-vp] DONE. agents.json now has {len(agents)} entries.")
    n_multi = sum(1 for v in agents.values()
                  if isinstance(v, dict) and isinstance(v.get("voiceprints"), list))
    print(f"[multi-vp] {n_multi} agents have multi-VP entries.", flush=True)


if __name__ == "__main__":
    main()
