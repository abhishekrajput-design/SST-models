"""
optimize_threshold.py — Find optimal threshold for 95%+ accuracy.

Tests multiple thresholds (0.25 to 0.55) on held-out API calls.
Reports accuracy, precision, recall per threshold.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from enroll_all_from_api import (
    INDEX_PATH, AUDIO_DIR, load_mp3_mono_16k, TARGET_SR, slug, ts2s,
)
from test_voiceprints_api import load_voiceprint_stacks, held_out_calls, evaluate_call

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
MIN_PHRASE_S = 0.4


def test_threshold(threshold: float, held: List[dict], voiceprints, model) -> dict:
    """Test all held-out calls with given threshold."""
    rows: List[dict] = []
    for rec in held:
        expected_slug = slug(rec["agent_name"])
        if expected_slug not in voiceprints:
            continue
        r = evaluate_call(rec, voiceprints, model, threshold, expected_slug)
        if r is None:
            continue
        rows.append(r)

    if not rows:
        return {"threshold": threshold, "calls": 0}

    n_calls = len(rows)
    correct = sum(1 for r in rows if r["correct_call"])
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    fn = sum(r["fn"] for r in rows)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    accuracy = correct / n_calls

    return {
        "threshold": threshold,
        "calls": n_calls,
        "accuracy": round(accuracy, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    print("[optimize] Loading model...")
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[optimize] {model.model_name} ready (dim={model.dim})")

    print("[optimize] Loading voiceprints...")
    vps = load_voiceprint_stacks(multi=True, target_dim=model.dim)
    if not vps:
        sys.exit("[ERROR] no voiceprints")

    print("[optimize] Loading held-out calls...")
    held = held_out_calls(MIN_PHRASE_S, 100, None)  # Use up to 100 calls
    print(f"[optimize] {len(held)} held-out calls available")

    if not held:
        sys.exit("[ERROR] no held-out calls")

    print(f"\n[optimize] Testing {len(THRESHOLDS)} thresholds...")
    print(f"{'Threshold':>10} {'Calls':>6} {'Accuracy':>10} {'P/R/F1':>20} {'tp/fp/tn/fn':>20}")
    print("=" * 70)

    results = []
    try:
        for threshold in THRESHOLDS:
            result = test_threshold(threshold, held, vps, model)
            results.append(result)
            if result["calls"] > 0:
                prec = result["precision"]
                rec = result["recall"]
                f1 = result["f1"]
                print(f"{result['threshold']:>10.2f} {result['calls']:>6d} "
                      f"{result['accuracy']*100:>9.1f}% "
                      f"{prec:.3f}/{rec:.3f}/{f1:.3f}  "
                      f"{result['tp']}/{result['fp']}/{result['tn']}/{result['fn']}")
    finally:
        model.unload()

    # Find best
    valid = [r for r in results if r["calls"] > 0]
    if valid:
        best = max(valid, key=lambda r: r["accuracy"])
        print(f"\n[BEST] threshold={best['threshold']:.2f} → "
              f"{best['accuracy']*100:.1f}% accuracy ({best['calls']} calls)")

    # Save results
    with open("threshold_optimization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[optimize] Results saved to threshold_optimization.json")


if __name__ == "__main__":
    main()
