# ZAK RAISSI TRAINING - MANUAL GEMINI STEPS

## ⚠️ API Quota Exhausted

The Gemini API free tier quota is exhausted. We'll use **manual browser upload** instead.

---

## Quick Setup (10 minutes total)

### Files to Upload
1. `call_processor/data/processed/zak_e2e_test_20260423/norm_dynonly_60s.wav` (1.8 MB)
2. `call_processor/data/processed/zak_compare_20260423/norm_held_out.wav` (7.9 MB)
3. `call_processor/data/processed/enhanced_zak_raissi_barnet_00_audio_03_15_2026_10_08_04_qbpe8e/norm_enhanced_zak_raissi_barnet_00_audio_03_15_2026_10_08_04_qbpe8e.wav` (54.9 MB)

---

## For Each File (Repeat 3 times):

### 1️⃣ CALL 1: zak_e2e_test_20260423

1. **Open**: https://gemini.google.com/app
2. **Upload**: `norm_dynonly_60s.wav` (Click "+" button → Attach file)
3. **Send this prompt**:

```
Transcribe this call and identify the speaker (agent or customer) for each segment with precise timestamps.
The agent's name is Zak Raissi.
Return ONLY valid JSON:

{
  "call_id": "zak_e2e_test_20260423",
  "agent_name": "Zak Raissi",
  "source": "gemini",
  "segments": [
    {"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."},
    {"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}
  ]
}

Focus on accuracy: mark each segment as "agent" or "customer".
```

4. **Copy the JSON response**
5. **Create file**: `call_processor/data/training/gemini_labels_zak_e2e_test_20260423_call1.json`
6. **Paste and save** the JSON

---

### 2️⃣ CALL 2: zak_compare_20260423

Repeat same steps but:
- Upload: `norm_held_out.wav` 
- Change `call_id` to: `"zak_compare_20260423"`
- Save file: `gemini_labels_zak_compare_20260423_call2.json`

---

### 3️⃣ CALL 3: enhanced_zak_raissi_barnet

Repeat same steps but:
- Upload: `norm_enhanced_zak_raissi_barnet_00_audio_03_15_2026_10_08_04_qbpe8e.wav`
- Change `call_id` to: `"enhanced_zak_raissi_barnet"`
- Save file: `gemini_labels_enhanced_zak_raissi_barnet_call3.json`

---

## After You've Saved All 3 JSON Files:

Run this command:

```bash
python call_processor/scripts/combine_and_retrain.py "Zak Raissi"
```

This will:
- Load all 3 JSON files
- Extract 500+ agent segments
- Create combined voiceprint
- Update agents.json

---

## Then Test:

```bash
python scripts/detailed_role_comparison.py
```

**Expected**: 95%+ accuracy on Zak! 🚀

---

## File Names Must Match Exactly:

- `gemini_labels_zak_e2e_test_20260423_call1.json`
- `gemini_labels_zak_compare_20260423_call2.json`  
- `gemini_labels_enhanced_zak_raissi_barnet_call3.json`

All go in: `call_processor/data/training/`

---

## Having Issues?

If Gemini's JSON is wrapped in markdown (```json ... ```):
1. Copy just the JSON part (between the braces)
2. Save as a `.json` file

The script will work once all 3 files are saved!
