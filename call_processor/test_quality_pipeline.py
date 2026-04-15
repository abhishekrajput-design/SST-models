"""
Test script: run quality scoring + enhancement pipeline on all test audio files.
Usage:
    cd call_processor
    python test_quality_pipeline.py

Tests audio from:
    C:\\Users\\abhis\\Desktop\\SST-models\\testing-audio\\{high,mid,low}\\*.mp3
"""
from __future__ import annotations
import os
import sys
import time
import shutil
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Add call_processor to path ────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── Test audio sources ────────────────────────────────────────────────────────
TESTING_AUDIO = r"C:\Users\abhis\Desktop\SST-models\testing-audio"
OUTPUT_DIR    = os.path.join(HERE, "data", "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all audio files from high/mid/low
def find_test_files():
    files = []
    for tier_label in ["high", "mid", "low"]:
        folder = os.path.join(TESTING_AUDIO, tier_label)
        if not os.path.isdir(folder):
            logger.warning(f"Folder not found: {folder}")
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
                files.append((tier_label, os.path.join(folder, f)))
    return files


def run_test(tier_label: str, audio_path: str) -> dict:
    fname = os.path.basename(audio_path)
    logger.info("=" * 65)
    logger.info(f"FILE: {fname}  (folder: {tier_label})")
    logger.info("=" * 65)

    result = {
        "file":         fname,
        "folder":       tier_label,
        "path":         audio_path,
    }

    # ── Step 1: DNSMOS quality scoring ────────────────────────────────────────
    logger.info("Step 1: DNSMOS quality scoring…")
    t0 = time.time()
    from quality_scorer import score_audio, compute_enhancement_gain
    pre_score = score_audio(audio_path)
    logger.info(
        f"  Pre-MOS:  p808={pre_score['p808_mos']:.2f}  "
        f"sig={pre_score['mos_sig']:.2f}  bak={pre_score['mos_bak']:.2f}  "
        f"ovr={pre_score['mos_ovr']:.2f}  → {pre_score['tier_label']}"
    )
    result["pre_score"] = pre_score
    result["scoring_time_s"] = round(time.time() - t0, 2)

    # Check expected tier vs actual
    expected_tiers = {"high": 1, "mid": 2, "low": 3}
    expected = expected_tiers.get(tier_label)
    actual   = pre_score["tier"]
    match    = "✓" if expected and actual <= expected + 1 else "?"
    logger.info(
        f"  Expected folder '{tier_label}' ≈ Tier {expected or '?'}  "
        f"Actual: Tier {actual}  {match}"
    )

    # ── Step 2: Enhancement router ────────────────────────────────────────────
    logger.info(f"Step 2: ClearVoice Tier {actual} enhancement…")
    out_wav = os.path.join(
        OUTPUT_DIR,
        f"{tier_label}__{fname.replace('.mp3','').replace('.wav','')}_enhanced.wav"
    )
    t1 = time.time()
    try:
        from enhancement_router import route, ClearVoiceModels
        cv_result = route(
            input_path=audio_path,
            output_path=out_wav,
            quality=pre_score,
            status_cb=lambda s: logger.info(f"  [{s}]"),
        )
        enh_time = round(time.time() - t1, 2)
        result["enhancement"] = cv_result
        result["enhancement_time_s"] = enh_time
        result["output_wav"] = out_wav
        logger.info(f"  Pipeline: {cv_result.get('pipeline_used','?')}  ({enh_time}s)")
        if cv_result.get("separated_streams"):
            logger.info(f"  Separated streams: {cv_result['separated_streams']}")
    except ImportError as exc:
        logger.warning(f"  ClearVoice not available: {exc}")
        result["enhancement"] = {"error": str(exc)}
        out_wav = audio_path  # fallback: use original

    # ── Step 3: Post-enhancement DNSMOS ──────────────────────────────────────
    if os.path.isfile(out_wav):
        logger.info("Step 3: Post-enhancement DNSMOS re-score…")
        post_score = score_audio(out_wav)
        gain       = compute_enhancement_gain(pre_score, post_score)
        logger.info(
            f"  Post-MOS: p808={post_score['p808_mos']:.2f}  "
            f"sig={post_score['mos_sig']:.2f}  bak={post_score['mos_bak']:.2f}  "
            f"ovr={post_score['mos_ovr']:.2f}  → {post_score['tier_label']}"
        )
        logger.info(f"  Enhancement gain: {gain:+.3f} MOS points")
        result["post_score"]       = post_score
        result["enhancement_gain"] = gain

        if post_score["mos_ovr"] < 2.0:
            logger.warning("  ⚠ Post-MOS still < 2.0 — would flag for human review")
            result["needs_human_review"] = True
        else:
            result["needs_human_review"] = False

    # ── Step 4: Quality scorer quick test ─────────────────────────────────────
    logger.info("Step 4: Review flag check…")
    flags = []
    if result.get("needs_human_review"):
        flags.append(f"Post-MOS {result['post_score']['mos_ovr']:.2f} < 2.0")
    result["review_reasons"] = flags

    logger.info(
        f"DONE: {fname}  |  "
        f"Tier {pre_score['tier']} → post-ovr={result.get('post_score',{}).get('mos_ovr',0):.2f}  |  "
        f"Review: {'YES' if flags else 'no'}"
    )
    return result


def main():
    files = find_test_files()
    if not files:
        logger.error(f"No audio files found in {TESTING_AUDIO}")
        sys.exit(1)

    logger.info(f"\nFound {len(files)} test audio files:")
    for label, path in files:
        logger.info(f"  [{label}]  {os.path.basename(path)}")
    logger.info("")

    results = []
    for tier_label, audio_path in files:
        try:
            r = run_test(tier_label, audio_path)
            results.append(r)
        except Exception as exc:
            import traceback
            logger.error(f"FAILED: {audio_path}\n{traceback.format_exc()}")
            results.append({"file": os.path.basename(audio_path), "folder": tier_label, "error": str(exc)})

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("SUMMARY")
    logger.info("=" * 65)
    logger.info(f"{'File':<45} {'Folder':<7} {'Tier':<6} {'Pre':>5} {'Post':>5} {'Gain':>6} {'Review'}")
    logger.info("-" * 65)
    for r in results:
        if "error" in r and "pre_score" not in r:
            logger.info(f"{r['file']:<45} {r['folder']:<7} ERROR: {r['error']}")
            continue
        pre  = r.get("pre_score",  {}).get("mos_ovr", 0)
        post = r.get("post_score", {}).get("mos_ovr", 0)
        gain = r.get("enhancement_gain", 0)
        tier = r.get("pre_score",  {}).get("tier", "?")
        rev  = "YES" if r.get("needs_human_review") else "no"
        logger.info(
            f"{r['file']:<45} {r['folder']:<7} Tier {tier}  "
            f"{pre:>5.2f}  {post:>5.2f}  {gain:>+6.3f}  {rev}"
        )

    # Save JSON report
    report_path = os.path.join(OUTPUT_DIR, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFull report saved → {report_path}")
    logger.info(f"Enhanced WAVs    → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
