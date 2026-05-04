"""
Compare UI diarization result (Omar identified) vs API ground truth (Mohamed Yasin-ali).
This tests whether the UI correctly identified the speaker or misidentified.
"""
import json
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).parent
RESULT_JSON = SCRIPT_DIR / "data/processed/enhanced_20260503T131905453_618398__parakeet-tdt-0.6b-v3/result.json"
INDEX_JSON = SCRIPT_DIR / "data/audiofy/_dataset/index.json"

def ts2s(ts):
    """Convert timestamp to seconds. Handle None, float, or string HH:MM:SS."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            parts = ts.split(":")
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return h * 3600 + m * 60 + s
            return float(ts)
        except:
            return 0.0
    return 0.0

def overlap_fraction(seg_start, seg_end, phrase_start, phrase_end):
    """Return fraction of segment overlapping with phrase (0 to 1)."""
    overlap_start = max(seg_start, phrase_start)
    overlap_end = min(seg_end, phrase_end)
    if overlap_end <= overlap_start:
        return 0.0
    overlap = overlap_end - overlap_start
    seg_dur = seg_end - seg_start
    if seg_dur == 0:
        return 0.0
    return overlap / seg_dur

def main():
    # Load UI result
    with open(RESULT_JSON, encoding="utf-8") as f:
        ui_result = json.load(f)
    ui_segments = ui_result.get("segments", [])

    # Load API ground truth
    with open(INDEX_JSON, encoding="utf-8") as f:
        index = json.load(f)
    api_rec = None
    for rec in index:
        if rec.get("_id") == "69efa352f91ac02559f7e936":
            api_rec = rec
            break

    if not api_rec:
        print("[ERROR] API call 69efa352f91ac02559f7e936 not found in index.json")
        return

    api_speaker_json = api_rec.get("speaker_json", [])
    api_agent_name = api_rec.get("agent_name", "Unknown")

    print(f"UI identified: Omar El Harchaoui")
    print(f"API ground truth: {api_agent_name}")
    print(f"UI segments: {len(ui_segments)}")
    print(f"API phrases: {len(api_speaker_json)}\n")

    # Match UI segments to API phrases by time overlap
    matches: List[dict] = []
    tp = fp = tn = fn = 0

    for seg in ui_segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        ui_label = seg.get("identified_speaker", "CUSTOMER")  # AGENT or CUSTOMER
        ui_agent = seg.get("agent_name", "")

        # Find best-overlapping API phrase
        best_overlap = 0.0
        best_phrase = None
        best_is_agent = False

        for phrase in api_speaker_json:
            if not isinstance(phrase, dict):
                continue
            ph_start = ts2s(phrase.get("start"))
            ph_end = ts2s(phrase.get("end"))
            speaker = (phrase.get("speaker") or "").strip()
            is_agent_truth = bool(speaker) and speaker.lower() != "customer"

            overlap = overlap_fraction(seg_start, seg_end, ph_start, ph_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_phrase = phrase
                best_is_agent = is_agent_truth

        # If no overlap found, skip this segment (silence gaps)
        if best_overlap < 0.01:
            continue

        # Compare UI label to API ground truth
        ui_is_agent = ui_label == "AGENT"
        api_is_agent = best_is_agent

        if ui_is_agent and api_is_agent:
            tp += 1
            verdict = "[ok] TP (both AGENT)"
        elif ui_is_agent and not api_is_agent:
            fp += 1
            verdict = "[FP] UI=AGENT, API=CUSTOMER"
        elif not ui_is_agent and api_is_agent:
            fn += 1
            verdict = "[FN] UI=CUSTOMER, API=AGENT"
        else:
            tn += 1
            verdict = "[ok] TN (both CUSTOMER)"

        matches.append({
            "start": seg_start,
            "end": seg_end,
            "overlap": round(best_overlap, 2),
            "ui_label": ui_label,
            "api_label": "AGENT" if api_is_agent else "CUSTOMER",
            "verdict": verdict,
            "ui_agent": ui_agent,
        })

    # Print per-segment summary
    print("=== PER-SEGMENT COMPARISON (top 20) ===")
    for i, m in enumerate(matches[:20]):
        print(f"  [{i+1:3d}] {m['start']:6.1f}-{m['end']:6.1f}s (overlap={m['overlap']}) "
              f"{m['ui_label']:>8} vs {m['api_label']:<8}  {m['verdict']}")
    if len(matches) > 20:
        print(f"  ... and {len(matches) - 20} more")

    # Compute metrics
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    print(f"\n=== ACCURACY METRICS ===")
    print(f"  Segments scored: {len(matches)}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision (AGENT): {prec:.3f}")
    print(f"  Recall (AGENT):    {rec:.3f}")
    print(f"  F1 (AGENT):        {f1:.3f}")
    print(f"  Overall Accuracy:  {accuracy:.3f}")

    # Interpretation
    print(f"\n=== INTERPRETATION ===")
    if f1 > 0.7:
        print(f"  [OK] GOOD: Speaker identification aligns well with ground truth")
    elif f1 > 0.5:
        print(f"  [~]  OKAY: Some misidentification; check where the errors cluster")
    else:
        print(f"  [XX] POOR: High mismatch - possible wrong speaker or audio swap")

    # Error breakdown
    if fp > 0 or fn > 0:
        print(f"\n  Errors: {fp} false positives (UI=AGENT when API=CUSTOMER)")
        print(f"          {fn} false negatives (UI=CUSTOMER when API=AGENT)")

    print(f"\n=== SPEAKER IDENTITY CHECK ===")
    print(f"  UI identified: Omar El Harchaoui")
    print(f"  API ground truth: {api_agent_name}")
    if "omar" in ui_agent.lower() and "yasin" in api_agent_name.lower():
        print(f"  [WARN] MISMATCH: Audio may be from a different call or")
        print(f"         the uploaded audio may be mislabeled in the UI")
    elif "omar" in ui_agent.lower() and "omar" in api_agent_name.lower():
        print(f"  [OK]  MATCH: Both identify as Omar")
    else:
        print(f"  [?]   CHECK: UI={ui_agent}, API={api_agent_name}")

if __name__ == "__main__":
    main()
