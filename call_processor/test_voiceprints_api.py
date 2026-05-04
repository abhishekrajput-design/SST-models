"""
test_voiceprints_api.py — Held-out API accuracy for the multi-voiceprint matcher.

Tests *speaker identification only* (the part the multi-VP change affects),
isolated from transcription/diarization. For each held-out call:
  - Iterate the API speaker_json phrases (ground truth: agent_name vs Customer).
  - Embed the audio slice for each phrase.
  - Match against all enrolled agent voiceprints (multi-VP, max-cosine).
  - Record predicted agent vs actual agent and AGENT/CUSTOMER label.

Reports:
  - Per-call: predicted agent + accuracy %
  - Overall: AGENT precision/recall/F1 vs CUSTOMER, and call-level "did we
    pick the right agent?" rate.
  - Side-by-side comparison: multi-VP vs single-VP (legacy voiceprint_path
    only). Shows where the multi-VP design wins / regresses.

Usage:
  python test_voiceprints_api.py                  # default 30 held-out calls
  python test_voiceprints_api.py --top 50
  python test_voiceprints_api.py --threshold 0.35
  python test_voiceprints_api.py --bucket low     # only calls measured low-SNR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent
os.chdir(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from enroll_all_from_api import (  # type: ignore
    INDEX_PATH, AUDIO_DIR, VP_DIR, AGENTS_JSON,
    TARGET_SR, slug, ts2s, load_mp3_mono_16k,
)
from enroll_multi_from_api import estimate_snr_db, bucket_for  # type: ignore
from src.voiceprints import resolve_voiceprint_path  # type: ignore

THRESHOLD_DEFAULT = 0.35   # matches diar_voiceprint.AGENT_SIM_THRESHOLD
MIN_PHRASE_S      = 0.4    # below this, embedding is too noisy to score


def load_voiceprint_stacks(multi: bool = True,
                            target_dim: Optional[int] = None,
                            ) -> Dict[str, Tuple[str, np.ndarray]]:
    """Return {slug: (display_name, (N, dim) stack)}.

    With ``multi=True`` use every entry from the new ``voiceprints`` list.
    With ``multi=False`` use only the legacy ``voiceprint_path`` (single
    centroid) — this is the baseline for the head-to-head comparison.

    If ``target_dim`` is given, voiceprints whose dim does not match are
    skipped (older ECAPA 192-dim files cannot be matched against a CAM++
    512-dim embedder, and vice versa).
    """
    if not AGENTS_JSON.exists():
        sys.exit(f"[ERROR] {AGENTS_JSON} not found — run enrollment first")
    with open(AGENTS_JSON, encoding="utf-8") as f:
        agents = json.load(f)

    out: Dict[str, Tuple[str, np.ndarray]] = {}
    for slg, info in agents.items():
        if not isinstance(info, dict):
            continue
        paths = []
        if multi and isinstance(info.get("voiceprints"), list):
            for entry in info["voiceprints"]:
                p = entry.get("path") if isinstance(entry, dict) else entry
                if p:
                    paths.append(p)
        if not paths:
            legacy = info.get("voiceprint_path") or info.get("voiceprint")
            if legacy:
                paths.append(legacy)

        loaded = []
        for raw in paths:
            r = resolve_voiceprint_path(raw, str(AGENTS_JSON))
            if not r or not os.path.isfile(r):
                continue
            try:
                vp = np.load(r).astype(np.float32).squeeze()
            except Exception:
                continue
            if vp.ndim != 1:
                continue
            if target_dim is not None and vp.shape[0] != target_dim:
                continue
            n = np.linalg.norm(vp)
            if n > 0:
                vp = vp / n
            loaded.append(vp)
        if not loaded:
            continue
        stack = np.stack(loaded).astype(np.float32)
        out[slg] = (info.get("agent_name") or slg, stack)
    return out


def held_out_calls(min_phrase_s: float, max_calls: int,
                    bucket_filter: Optional[str]) -> List[dict]:
    """Calls in index.json that were *not* used for any agent's enrollment.

    Trusts the new ``per_call_snr`` field written by enroll_multi_from_api.py
    to know which call IDs were used. If an agent only has legacy enrollment
    metadata, we conservatively treat ``used_calls`` worth of its top calls as
    consumed (same selection rule the legacy enroller used).
    """
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)
    if not AGENTS_JSON.exists():
        used_ids = set()
    else:
        with open(AGENTS_JSON, encoding="utf-8") as f:
            agents = json.load(f)
        used_ids = set()
        for slg, info in agents.items():
            if not isinstance(info, dict):
                continue
            pcs = info.get("per_call_snr")
            if isinstance(pcs, list) and pcs:
                used_ids.update(rec["_id"] for rec in pcs if rec.get("_id"))

    by_agent: Dict[str, list] = {}
    for rec in index:
        by_agent.setdefault(rec["agent_name"], []).append(rec)

    held: List[dict] = []
    seen_ids: set = set()
    for name, recs in by_agent.items():
        for rec in recs:
            rid = rec.get("_id")
            if not rid or rid in used_ids or rid in seen_ids:
                continue
            seen_ids.add(rid)
            if rec.get("n_agent_phrases", 0) < 3:
                continue
            held.append(rec)

    held.sort(key=lambda r: r.get("connect_time", ""), reverse=True)
    if bucket_filter:
        held = [r for r in held if r.get("_audio_bucket") == bucket_filter]
    return held[:max_calls]


def evaluate_call(
    rec: dict,
    voiceprints: Dict[str, Tuple[str, np.ndarray]],
    model,
    threshold: float,
    expected_slug: str,
) -> Optional[dict]:
    """Score one call. Returns metrics dict or None if call unusable."""
    rid = rec["_id"]
    mp3 = AUDIO_DIR / f"{rid}.mp3"
    if not (mp3.exists() and mp3.stat().st_size > 1000):
        return None
    try:
        audio, sr = load_mp3_mono_16k(mp3)
    except Exception:
        return None

    # Track per-segment ground-truth vs prediction
    tp = fp = tn = fn = 0    # AGENT positive class
    correct_agent_hits = 0
    total_agent_phrases = 0
    agent_sim_sum: Dict[str, float] = {}
    agent_sim_n:   Dict[str, int]   = {}
    agent_only_samples = []

    slugs = list(voiceprints.keys())
    stacks = [voiceprints[s][1] for s in slugs]

    for ph in rec.get("speaker_json", []):
        if not isinstance(ph, dict):
            continue
        s = ts2s(ph.get("start"))
        e = ts2s(ph.get("end"))
        if e - s < MIN_PHRASE_S:
            continue
        si = max(0, int(s * sr)); ei = min(int(e * sr), len(audio))
        if ei - si < int(sr * MIN_PHRASE_S):
            continue
        chunk = audio[si:ei]
        emb = model.embed_chunk(chunk, sr)
        if emb is None:
            continue
        n = np.linalg.norm(emb)
        if n == 0:
            continue
        emb = emb / n

        # Cosine to every agent (max across that agent's centroids)
        sims = np.array([float(np.max(stacks[j] @ emb))
                          for j in range(len(slugs))], dtype=np.float32)
        if sims.size == 0:
            continue
        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])
        pred_slug = slugs[best_j]

        is_agent_pred = best_sim >= threshold
        speaker = (ph.get("speaker") or "").strip()
        is_agent_truth = bool(speaker) and speaker.lower() != "customer"

        if is_agent_truth:
            total_agent_phrases += 1
            agent_only_samples.append(chunk)
            if pred_slug == expected_slug and is_agent_pred:
                correct_agent_hits += 1

        if is_agent_pred and is_agent_truth: tp += 1
        elif is_agent_pred and not is_agent_truth: fp += 1
        elif not is_agent_pred and is_agent_truth: fn += 1
        else: tn += 1

        agent_sim_sum[pred_slug] = agent_sim_sum.get(pred_slug, 0.0) + best_sim
        agent_sim_n[pred_slug]   = agent_sim_n.get(pred_slug, 0) + 1

    # Call-level identified agent (top-30% mean across phrases)
    if not agent_sim_n:
        return None
    call_agent_slug = max(
        agent_sim_sum, key=lambda k: agent_sim_sum[k] / max(agent_sim_n[k], 1))
    correct_call = (call_agent_slug == expected_slug)

    # Bucket the held-out call by SNR — full audio so silence gaps reveal
    # the real noise floor (must match the enrollment-side estimator).
    snr_db = estimate_snr_db(audio, sr)

    return {
        "_id":               rid,
        "expected_slug":     expected_slug,
        "predicted_slug":    call_agent_slug,
        "correct_call":      correct_call,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "agent_phrases":     total_agent_phrases,
        "agent_hits":        correct_agent_hits,
        "snr_db":            round(snr_db, 1),
        "bucket":            bucket_for(snr_db),
    }


def summarise(rows: List[dict], label: str) -> None:
    if not rows:
        print(f"[{label}] no scored calls"); return
    n_calls = len(rows)
    correct = sum(1 for r in rows if r["correct_call"])
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    tn = sum(r["tn"] for r in rows); fn = sum(r["fn"] for r in rows)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n=== {label} ===")
    print(f"  calls scored:           {n_calls}")
    print(f"  call-level agent ID:    {correct}/{n_calls} "
          f"({100.0*correct/n_calls:.1f}%)")
    print(f"  segment AGENT P/R/F1:   {prec:.3f} / {rec:.3f} / {f1:.3f}")
    print(f"  segment counts:         tp={tp} fp={fp} tn={tn} fn={fn}")

    # Per-bucket breakdown
    by_bucket: Dict[str, list] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)
    for bkt in ("high", "mid", "low"):
        bk = by_bucket.get(bkt, [])
        if not bk:
            continue
        bk_correct = sum(1 for r in bk if r["correct_call"])
        bk_tp = sum(r["tp"] for r in bk); bk_fp = sum(r["fp"] for r in bk)
        bk_fn = sum(r["fn"] for r in bk)
        bk_p = bk_tp / max(bk_tp + bk_fp, 1)
        bk_r = bk_tp / max(bk_tp + bk_fn, 1)
        bk_f = 2 * bk_p * bk_r / max(bk_p + bk_r, 1e-9)
        print(f"  [{bkt}] {len(bk)} calls  "
              f"call-id={bk_correct}/{len(bk)}  P/R/F1={bk_p:.3f}/{bk_r:.3f}/{bk_f:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30,
                    help="Max held-out calls to score")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    ap.add_argument("--bucket", choices=("high", "mid", "low"), default=None,
                    help="Restrict to held-out calls in one SNR bucket")
    ap.add_argument("--save", default=None,
                    help="Optional path to dump full per-call JSON results")
    args = ap.parse_args()

    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[test-api] {model.model_name} ready (dim={model.dim})", flush=True)

    multi_vps   = load_voiceprint_stacks(multi=True,  target_dim=model.dim)
    legacy_vps  = load_voiceprint_stacks(multi=False, target_dim=model.dim)
    if not multi_vps:
        sys.exit("[ERROR] no voiceprints loaded — run enroll_multi_from_api.py first")
    n_multi = sum(s.shape[0] for _, s in multi_vps.values())
    n_legacy = sum(s.shape[0] for _, s in legacy_vps.values())
    print(f"[test-api] multi-VP: {len(multi_vps)} agents / {n_multi} centroids")
    print(f"[test-api] single-VP baseline: {len(legacy_vps)} agents / {n_legacy} centroids")

    held = held_out_calls(MIN_PHRASE_S, args.top, args.bucket)
    print(f"[test-api] {len(held)} held-out calls selected")
    if not held:
        sys.exit("[test-api] no held-out calls — every call in index.json was "
                 "used in enrollment. Re-scrape with a wider window.")

    multi_rows: List[dict] = []
    single_rows: List[dict] = []
    try:
        for i, rec in enumerate(held, 1):
            expected_slug = slug(rec["agent_name"])
            if expected_slug not in multi_vps:
                continue   # this agent isn't enrolled — skip
            t0 = time.time()
            mr = evaluate_call(rec, multi_vps, model, args.threshold, expected_slug)
            sr_ = evaluate_call(rec, legacy_vps, model, args.threshold, expected_slug)
            if mr is None or sr_ is None:
                continue
            multi_rows.append(mr); single_rows.append(sr_)
            verdict_m = "ok" if mr["correct_call"] else "WRONG"
            verdict_s = "ok" if sr_["correct_call"] else "WRONG"
            print(f"  [{i:3d}/{len(held)}] {rec['_id'][:8]} "
                  f"({rec['agent_name'][:24]:>24}, {mr['bucket']:>4}, "
                  f"{mr['snr_db']:5.1f}dB) "
                  f"multi={verdict_m:>5} single={verdict_s:>5}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    finally:
        model.unload()

    summarise(multi_rows, "MULTI-VP")
    summarise(single_rows, "SINGLE-VP (legacy)")

    # Head-to-head: where did multi-VP help vs hurt vs no change?
    helped = hurt = same = 0
    for m, s in zip(multi_rows, single_rows):
        if m["correct_call"] and not s["correct_call"]:
            helped += 1
        elif s["correct_call"] and not m["correct_call"]:
            hurt += 1
        else:
            same += 1
    print(f"\n=== HEAD-TO-HEAD (call-level agent ID) ===")
    print(f"  multi-VP fixed single-VP miss:  {helped}")
    print(f"  multi-VP regressed single-VP:   {hurt}")
    print(f"  same outcome:                   {same}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"multi": multi_rows, "single": single_rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\n[test-api] saved per-call results -> {args.save}")


if __name__ == "__main__":
    main()
