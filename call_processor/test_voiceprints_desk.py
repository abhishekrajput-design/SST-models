"""
test_voiceprints_desk.py — Sanity-check multi-voiceprint matching on noisy
desk recordings under testing-audio/{low,mid,high}/.

There's no phrase-level ground truth for these files, so the test reports:
  - The identified agent (top-30% mean cosine across detected speech windows).
  - Top-3 candidate agents with scores — useful when the call is borderline.
  - Estimated AGENT vs CUSTOMER time share (sanity: should be roughly 50/50
    on a real 1-on-1 call; a wildly skewed split usually means the matcher
    confused the two parties).
  - Estimated SNR of the recording (which bucket it would fall in if used
    for enrollment).

Writes a result JSON next to each MP3 for manual spot-checking.

Usage:
  python test_voiceprints_desk.py
  python test_voiceprints_desk.py --threshold 0.30
  python test_voiceprints_desk.py --dir ../testing-audio
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

from enroll_all_from_api import (  # type: ignore
    TARGET_SR, load_mp3_mono_16k, AGENTS_JSON,
)
from enroll_multi_from_api import estimate_snr_db, bucket_for  # type: ignore
from test_voiceprints_api import load_voiceprint_stacks  # type: ignore

WINDOW_S      = 1.5
STRIDE_S      = 0.75
VAD_FRAME_S   = 0.030
VAD_PCTL      = 60     # frames whose RMS exceeds the Pth percentile = speech


def vad_speech_mask(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return a boolean mask over WINDOW_S strides marking 'speech' windows.

    Crude energy VAD: a stride is speech if at least 30 % of its 30 ms frames
    sit above the recording-level RMS percentile. Good enough to drop pure
    silence/background-noise windows from the per-call score.
    """
    frame = int(sr * VAD_FRAME_S)
    if frame == 0:
        return np.ones(0, dtype=bool)
    n_frames = len(audio) // frame
    framed = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1) + 1e-12)
    if not rms.size:
        return np.ones(0, dtype=bool)
    thresh = float(np.percentile(rms, VAD_PCTL))

    win = int(sr * WINDOW_S); step = int(sr * STRIDE_S)
    if win == 0:
        return np.ones(0, dtype=bool)
    mask = []
    for s in range(0, len(audio) - win + 1, step):
        # Frame indices that fall inside this window
        fs = s // frame
        fe = (s + win) // frame
        if fe <= fs:
            mask.append(False)
            continue
        active = float((rms[fs:fe] >= thresh).mean())
        mask.append(active >= 0.30)
    return np.array(mask, dtype=bool)


def score_recording(
    mp3: Path,
    voiceprints: Dict[str, Tuple[str, np.ndarray]],
    model,
    threshold: float,
) -> dict:
    audio, sr = load_mp3_mono_16k(mp3)
    duration_s = len(audio) / sr
    snr_db = estimate_snr_db(audio, sr)

    win = int(sr * WINDOW_S); step = int(sr * STRIDE_S)
    speech_mask = vad_speech_mask(audio, sr)

    slugs = list(voiceprints.keys())
    stacks = [voiceprints[s][1] for s in slugs]
    names  = [voiceprints[s][0] for s in slugs]

    # Per-window: best agent + best score
    windows: List[dict] = []
    sim_sum: Dict[str, float] = {s: 0.0 for s in slugs}
    sim_n:   Dict[str, int]   = {s: 0   for s in slugs}

    idx = 0
    for s in range(0, len(audio) - win + 1, step):
        is_speech = bool(speech_mask[idx]) if idx < len(speech_mask) else False
        idx += 1
        if not is_speech:
            continue
        chunk = audio[s:s + win]
        emb = model.embed_chunk(chunk, sr)
        if emb is None:
            continue
        n = np.linalg.norm(emb)
        if n == 0:
            continue
        emb = emb / n
        sims = np.array([float(np.max(stacks[j] @ emb))
                          for j in range(len(slugs))], dtype=np.float32)
        if sims.size == 0:
            continue
        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])
        windows.append({
            "t":      round(s / sr, 2),
            "best":   slugs[best_j],
            "sim":    round(best_sim, 3),
            "label":  "AGENT" if best_sim >= threshold else "CUSTOMER",
        })
        sim_sum[slugs[best_j]] += best_sim
        sim_n[slugs[best_j]]   += 1

    # Pick the call's agent: top-30 % mean of windows whose own best matched
    # that slug — same aggregation rule diar_voiceprint uses.
    if windows:
        # Build per-agent score list (only windows where that agent was best)
        per_agent_scores: Dict[str, List[float]] = {s: [] for s in slugs}
        for w in windows:
            per_agent_scores[w["best"]].append(w["sim"])
        ranked: List[Tuple[str, float, int]] = []
        for slg, scores in per_agent_scores.items():
            if not scores:
                continue
            arr = np.sort(np.asarray(scores))
            k = max(3, int(len(arr) * 0.3))
            topk_mean = float(arr[-k:].mean())
            ranked.append((slg, topk_mean, len(scores)))
        ranked.sort(key=lambda t: -t[1])
        top3 = ranked[:3]
        identified_slug = top3[0][0] if top3 else ""
    else:
        top3 = []
        identified_slug = ""

    # AGENT/CUSTOMER time share — only count windows whose best agent is the
    # identified one as AGENT (everything else is treated as CUSTOMER).
    if windows:
        agent_windows = [w for w in windows if w["best"] == identified_slug
                                              and w["sim"] >= threshold]
        agent_share = len(agent_windows) / len(windows)
    else:
        agent_share = 0.0

    return {
        "file":             str(mp3),
        "duration_s":       round(duration_s, 1),
        "estimated_snr_db": round(snr_db, 1),
        "estimated_bucket": bucket_for(snr_db),
        "identified_slug":  identified_slug,
        "identified_name":  voiceprints[identified_slug][0] if identified_slug else "",
        "top3":             [
            {"slug": s, "name": voiceprints[s][0], "score": round(sc, 3),
             "n_windows": n}
            for (s, sc, n) in top3
        ],
        "n_speech_windows": len(windows),
        "agent_time_share": round(agent_share, 3),
        "windows":          windows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(SCRIPT_DIR.parent / "testing-audio"),
                    help="Root containing low/, mid/, high/ subdirs of MP3s")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--single-vp", action="store_true",
                    help="Use legacy single-VP matcher instead of multi-VP")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        sys.exit(f"[ERROR] {root} does not exist")

    targets: List[Path] = []
    for sub in ("low", "mid", "high"):
        d = root / sub
        if d.is_dir():
            for mp3 in sorted(d.glob("*.mp3")):
                targets.append(mp3)
    if not targets:
        for mp3 in sorted(root.glob("*.mp3")):
            targets.append(mp3)
    if not targets:
        sys.exit(f"[ERROR] no MP3s found under {root}")
    print(f"[test-desk] {len(targets)} desk recordings to test", flush=True)

    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[test-desk] {model.model_name} ready (dim={model.dim})", flush=True)

    voiceprints = load_voiceprint_stacks(multi=not args.single_vp,
                                          target_dim=model.dim)
    if not voiceprints:
        sys.exit("[ERROR] no voiceprints loaded — run enrollment first")
    n_centroids = sum(s.shape[0] for _, s in voiceprints.values())
    mode = "single-VP (legacy)" if args.single_vp else "multi-VP"
    print(f"[test-desk] {mode}: {len(voiceprints)} agents / {n_centroids} centroids",
          flush=True)

    results: List[dict] = []
    try:
        for mp3 in targets:
            t0 = time.time()
            try:
                r = score_recording(mp3, voiceprints, model, args.threshold)
            except Exception as e:
                print(f"  [{mp3.name}] failed: {e}", flush=True)
                continue
            results.append(r)
            top1 = r["top3"][0] if r["top3"] else {"name": "?", "score": 0.0}
            top2_str = ""
            if len(r["top3"]) > 1:
                top2_str = (f"   2nd: {r['top3'][1]['name'][:18]:>18} "
                            f"({r['top3'][1]['score']:.3f})")
            print(f"  [{mp3.parent.name:>4}/{mp3.name}] "
                  f"snr={r['estimated_snr_db']:5.1f}dB "
                  f"agent={top1['name'][:22]:>22} ({top1['score']:.3f})  "
                  f"agent_share={r['agent_time_share']:.2f}  "
                  f"({time.time()-t0:.1f}s){top2_str}", flush=True)

            # Save per-recording JSON next to the MP3
            out_path = mp3.with_suffix(".result.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
    finally:
        model.unload()

    print("\n=== Summary ===")
    for r in results:
        bucket = r["estimated_bucket"]
        ok = "✓" if r["identified_slug"] else "—"
        share = r["agent_time_share"]
        flag = "" if 0.20 <= share <= 0.80 else "  [share unusual]"
        print(f"  {ok} {bucket:>4}  {Path(r['file']).name:<48}  "
              f"-> {r['identified_name'][:30]:<30}  share={share:.2f}{flag}")


if __name__ == "__main__":
    main()
