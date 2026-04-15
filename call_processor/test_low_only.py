"""
Quick focused test on the low-quality audio file only.
Skips DNSMOS scoring (slow) - directly runs ClearVoice Tier 3 pipeline.

Usage:
    python test_low_only.py                          # uses DEFAULT_LOW_FILE
    python test_low_only.py /path/to/audio.mp3       # explicit path
"""
from __future__ import annotations
import os, sys, time, logging, json
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_LOW_FILE = r"C:\Users\abhis\Desktop\SST-models\testing-audio\low\audio_04_12_2026_10_38_45_ldwibu.mp3"
OUT_DIR = os.path.join(HERE, "data", "test_results")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    LOW_FILE = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOW_FILE
    if not os.path.isfile(LOW_FILE):
        logger.error(f"File not found: {LOW_FILE}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"LOW FILE: {os.path.basename(LOW_FILE)}")
    logger.info("=" * 60)

    # ── Step 1: DNSMOS pre-score ────────────────────────────────────
    logger.info("Step 1: DNSMOS pre-score…")
    t0 = time.time()
    from quality_scorer import score_audio, compute_enhancement_gain
    pre = score_audio(LOW_FILE)
    logger.info(f"  Pre  → mos_ovr={pre['mos_ovr']:.2f}  {pre['tier_label']}  ({time.time()-t0:.1f}s)")

    # ── Step 2: ClearVoice enhancement ──────────────────────────────
    out_wav = os.path.join(OUT_DIR, "low_enhanced.wav")
    logger.info(f"Step 2: ClearVoice Tier {pre['tier']} enhancement…")
    t1 = time.time()
    from enhancement_router import route
    cv = route(
        input_path=LOW_FILE,
        output_path=out_wav,
        quality=pre,
        status_cb=lambda s: logger.info(f"  [{s}]"),
    )
    logger.info(f"  Pipeline: {cv.get('pipeline_used','?')}  ({time.time()-t1:.1f}s)")
    logger.info(f"  Output: {out_wav}")
    if cv.get("separated_streams"):
        logger.info(f"  Streams: {cv['separated_streams']}")

    # ── Step 3: DNSMOS post-score ───────────────────────────────────
    logger.info("Step 3: DNSMOS post-score…")
    t2 = time.time()
    post = score_audio(out_wav)
    gain = compute_enhancement_gain(pre, post)
    logger.info(f"  Post → mos_ovr={post['mos_ovr']:.2f}  {post['tier_label']}  ({time.time()-t2:.1f}s)")
    logger.info(f"  Gain: {gain:+.3f} MOS  |  Review: {'YES' if post['mos_ovr']<2.0 else 'no'}")

    # ── Save report ─────────────────────────────────────────────────
    report = {
        "file":     os.path.basename(LOW_FILE),
        "pre":      pre,
        "post":     post,
        "gain":     gain,
        "pipeline": cv.get("pipeline_used"),
        "output":   out_wav,
        "separated_streams": cv.get("separated_streams", []),
        "needs_human_review": post["mos_ovr"] < 2.0,
    }
    rp = os.path.join(OUT_DIR, "low_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved → {rp}")
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULT: {os.path.basename(LOW_FILE)}")
    logger.info(f"  Pre-MOS  : {pre['mos_ovr']:.2f}  ({pre['tier_label']})")
    logger.info(f"  Post-MOS : {post['mos_ovr']:.2f}  ({post['tier_label']})")
    logger.info(f"  Gain     : {gain:+.3f}")
    logger.info(f"  Pipeline : {cv.get('pipeline_used')}")
    logger.info(f"  Review   : {'YES ⚠' if post['mos_ovr']<2.0 else 'No ✓'}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
