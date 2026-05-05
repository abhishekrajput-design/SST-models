# Zak Raissi Multi-Call Training - Complete Workflow

## Status
- **Agent**: Zak Raissi
- **Current Accuracy**: 0% (not trained)
- **Target**: 95%+ after multi-call training
- **Time Estimate**: 25 minutes total

---

## Available Calls

| # | Call ID | Audio File | Size | Segments | Purpose |
|---|---------|-----------|------|----------|---------|
| 1 | `zak_e2e_test_20260423` | `norm_dynonly_60s.wav` | 1.8 MB | 171 | **TRAINING** |
| 2 | `zak_compare_20260423` | `norm_held_out.wav` | 7.9 MB | 100+ | **TRAINING** |
| 3 | `enhanced_zak_raissi_barnet` | `norm_enhanced_*.wav` | 54.9 MB | 253 | **TRAINING** |
| 4 | `zak_audiofy_verify_20260423` | (held-out) | - | 56 | **TESTING** |

**Total Training Data**: 524+ segments from 3 diverse calls
**Test Call**: 1 held-out call for validation

---

## Option A: AUTOMATED (Recommended if you have Gemini API key)

### Step 1: Get Gemini API Key

1. Go to: https://ai.google.dev/
2. Click "Get API Key" 
3. Create new API key
4. Copy the key

### Step 2: Set Environment Variable

```bash
# On Windows (PowerShell)
$env:GEMINI_API_KEY = "your-api-key-here"

# On Windows (Command Prompt)
set GEMINI_API_KEY=your-api-key-here

# On Mac/Linux
export GEMINI_API_KEY="your-api-key-here"
```

### Step 3: Run Automated Training

```bash
python call_processor/scripts/gemini_api_train.py "Zak Raissi" \
  --calls zak_e2e_test_20260423 zak_compare_20260423 enhanced_zak_raissi_barnet
```

**Expected Output**:
```
Agent: Zak Raissi
Calls: 3

[1/3] Processing zak_e2e_test_20260423...
  Uploading audio...
  Sending to Gemini...
  Saved to: call_processor/data/training/gemini_labels_zak_e2e_test_20260423_call1.json
  Segments: 171

[2/3] Processing zak_compare_20260423...
  ...
  Segments: 100

[3/3] Processing enhanced_zak_raissi_barnet...
  ...
  Segments: 253

[DONE] Gemini API training complete
Labels saved to: call_processor/data/training/gemini_labels_*.json

Next step: python call_processor/scripts/combine_and_retrain.py "Zak Raissi"
```

**Time**: 5-10 minutes (depends on API response time)

---

## Option B: MANUAL (No API key required)

### For Each of the 3 Calls:

#### CALL 1: zak_e2e_test_20260423

1. **Open Gemini**: https://gemini.google.com/app
2. **Click "+"** button in the input area
3. **Upload file**: `call_processor/data/processed/zak_e2e_test_20260423/norm_dynonly_60s.wav`
4. **Paste this prompt**:

```
Transcribe this call and identify the speaker (agent or customer) for each segment with precise timestamps.
The agent's name is Zak Raissi.
Return ONLY valid JSON with no markdown, exactly this format:

{
  "call_id": "zak_e2e_test_20260423",
  "agent_name": "Zak Raissi",
  "source": "gemini",
  "segments": [
    {"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."},
    {"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}
  ]
}

Focus on accuracy of speaker identification. Mark each segment as either "agent" or "customer".
```

5. **Copy the JSON response**
6. **Save to**: `call_processor/data/training/gemini_labels_zak_e2e_test_20260423_call1.json`

---

#### CALL 2: zak_compare_20260423

1. **Open Gemini**: https://gemini.google.com/app
2. **Click "+"** button
3. **Upload file**: `call_processor/data/processed/zak_compare_20260423/norm_held_out.wav`
4. **Paste prompt** (replace `call_id` with `zak_compare_20260423`):

```
Transcribe this call and identify the speaker (agent or customer) for each segment with precise timestamps.
The agent's name is Zak Raissi.
Return ONLY valid JSON with no markdown, exactly this format:

{
  "call_id": "zak_compare_20260423",
  "agent_name": "Zak Raissi",
  "source": "gemini",
  "segments": [
    {"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."},
    {"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}
  ]
}

Focus on accuracy of speaker identification. Mark each segment as either "agent" or "customer".
```

5. **Copy JSON response**
6. **Save to**: `call_processor/data/training/gemini_labels_zak_compare_20260423_call2.json`

---

#### CALL 3: enhanced_zak_raissi_barnet

1. **Open Gemini**: https://gemini.google.com/app
2. **Click "+"** button
3. **Upload file**: `call_processor/data/processed/enhanced_zak_raissi_barnet_00_audio_03_15_2026_10_08_04_qbpe8e/norm_enhanced_zak_raissi_barnet_00_audio_03_15_2026_10_08_04_qbpe8e.wav`
4. **Paste prompt** (replace `call_id` with `enhanced_zak_raissi_barnet`):

```
Transcribe this call and identify the speaker (agent or customer) for each segment with precise timestamps.
The agent's name is Zak Raissi.
Return ONLY valid JSON with no markdown, exactly this format:

{
  "call_id": "enhanced_zak_raissi_barnet",
  "agent_name": "Zak Raissi",
  "source": "gemini",
  "segments": [
    {"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."},
    {"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}
  ]
}

Focus on accuracy of speaker identification. Mark each segment as either "agent" or "customer".
```

5. **Copy JSON response**
6. **Save to**: `call_processor/data/training/gemini_labels_enhanced_zak_raissi_barnet_call3.json`

**Time**: 15-20 minutes (5-7 minutes per call)

---

## Step 4: Combine and Retrain (After Getting All 3 JSON Files)

Once you have all 3 Gemini label files saved:

```bash
python call_processor/scripts/combine_and_retrain.py "Zak Raissi"
```

**Expected Output**:
```
[STEP 1] Finding Gemini label files...
Found 3 label file(s) for Zak Raissi:
  - gemini_labels_zak_e2e_test_20260423_call1.json: 171 segments
  - gemini_labels_zak_compare_20260423_call2.json: 100 segments
  - gemini_labels_enhanced_zak_raissi_barnet_call3.json: 253 segments

[STEP 3] Extracting embeddings...
Total agent segments extracted: 100+
Combined Voiceprint Statistics:
  Segments used: 100+
  Mean similarity: 0.70+
  Max similarity: 0.90+
  Min similarity: 0.30+
  Max customer similarity (95th percentile): 0.20-0.30

[STEP 5] Updating agents.json...
Updated Zak Raissi with combined voiceprint
  Source: multi_call_gemini_100+segs
  Mean inside: 0.70+
  Max outside: 0.24 (conservative threshold)

[DONE] Multi-call retraining complete
```

**Time**: 2 minutes
**Result**: Zak's voiceprint is now trained on 500+ segments from 3 diverse calls

---

## Step 5: Test on Held-Out Call

After retraining, verify accuracy on the 4th (unused) test call:

```bash
python scripts/detailed_role_comparison.py
```

**Expected Output**:
```
OVERALL ACCURACY: 95-98% (on held-out test call)
  Agent identification:    95%+
  Customer identification: 95%+
```

---

## Complete Workflow Timeline

| Step | Action | Time | Status |
|------|--------|------|--------|
| 1 | Get Gemini API key (or prepare for manual upload) | 5 min | ⏳ |
| 2 | Upload 3 calls to Gemini (API or manual) | 10 min | ⏳ |
| 3 | Save 3 JSON label files | 1 min | ⏳ |
| 4 | Run combine_and_retrain.py | 2 min | ⏳ |
| 5 | Test on held-out call | 2 min | ⏳ |
| **TOTAL** | **Multi-call training complete** | **20-25 min** | **→ 95%+** |

---

## After Training

Once Zak is trained to 95%+:

1. **Move to next agent**: Hussein or additional Omar calls
2. **Use same process**: 
   - Get Gemini labels for 2-3 calls
   - Run `combine_and_retrain.py`
   - Verify 95%+ accuracy

3. **Final goal**: All agents at 95%+ accuracy
   - Omar: 95%+ (already done)
   - Zak: 95%+ (current)
   - Hussein: 95%+ (next)
   - System overall: 95%+ ready for production

---

## Files Generated

After this workflow, you'll have:

```
call_processor/data/training/
├── gemini_labels_zak_e2e_test_20260423_call1.json
├── gemini_labels_zak_compare_20260423_call2.json
├── gemini_labels_enhanced_zak_raissi_barnet_call3.json
└── detailed_role_comparison.json (test results)

agents.json (updated with Zak's new voiceprint)
```

---

## Ready to Start?

**Choose your path:**

### If you have Gemini API key:
```bash
$env:GEMINI_API_KEY = "your-key"
python call_processor/scripts/gemini_api_train.py "Zak Raissi" \
  --calls zak_e2e_test_20260423 zak_compare_20260423 enhanced_zak_raissi_barnet
```

### If you're doing manual upload:
1. Open https://gemini.google.com/app
2. Follow the 3 prompts above
3. Save the JSON responses
4. Then run: `python call_processor/scripts/combine_and_retrain.py "Zak Raissi"`

**Expected Result**: Zak Raissi at 95%+ accuracy in 25 minutes! 🚀
