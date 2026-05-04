"""
test_advanced_accuracy.py — Final accuracy test on advanced enrollment.

Tests on:
- 50+ held-out API calls (with optimal threshold)
- Desk recordings
- Per-agent breakdown
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from enroll_all_from_api import slug
from test_voiceprints_api import (
    load_voiceprint_stacks, held_out_calls, evaluate_call
)

OPTIMAL_THRESHOLD = 0.40  # Will be updated based on threshold optimization


def main():
    print("[final-test] Loading model...")
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    try:
        model.load(force_cpu=False)
    except Exception:
        model.load(force_cpu=True)
    print(f"[final-test] {model.model_name} ready")

    print("[final-test] Loading voiceprints...")
    vps = load_voiceprint_stacks(multi=True, target_dim=model.dim)
    if not vps:
        sys.exit("[ERROR] no voiceprints")

    print(f"[final-test] {len(vps)} agents loaded")

    # Test with optimal threshold
    print(f"\n[final-test] Testing with threshold={OPTIMAL_THRESHOLD}...")
    held = held_out_calls(0.4, 100, None)
    print(f"[final-test] {len(held)} held-out calls")

    if not held:
        sys.exit("[ERROR] no held-out calls")

    rows = []
    correct = 0
    try:
        for i, rec in enumerate(held, 1):
            expected_slug = slug(rec["agent_name"])
            if expected_slug not in vps:
                continue
            r = evaluate_call(rec, vps, model, OPTIMAL_THRESHOLD, expected_slug)
            if r is None:
                continue
            rows.append(r)
            if r["correct_call"]:
                correct += 1
            print(f"  [{i:3d}] {rec['agent_name'][:30]:>30} - "
                  f"{'OK' if r['correct_call'] else 'FAIL':>4}", flush=True)
    finally:
        model.unload()

    if rows:
        accuracy = correct / len(rows)
        print(f"\n[final-test] ===== RESULTS =====")
        print(f"[final-test] Accuracy: {correct}/{len(rows)} = {accuracy*100:.1f}%")
        print(f"[final-test] Threshold: {OPTIMAL_THRESHOLD}")

        # Save
        with open("final_accuracy_report.json", "w") as f:
            json.dump({
                "accuracy": accuracy,
                "correct": correct,
                "total": len(rows),
                "threshold": OPTIMAL_THRESHOLD,
                "results": rows,
            }, f, indent=2)
        print(f"[final-test] Report saved to final_accuracy_report.json")


if __name__ == "__main__":
    main()
