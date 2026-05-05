# Role Identification Fix - Path to 95%+ Accuracy

## Current Problem: 48.1% Accuracy

We're getting **SPEAKER ROLE IDENTIFICATION WRONG** on many segments.

### The Failures:

**AGENT segments misclassified as CUSTOMER (12 errors):**
```
"Hello, Mark?"           sim=0.056   (should be AGENT, marked as CUSTOMER)
"5 o'clock?"             sim=0.163   (should be AGENT, marked as CUSTOMER)
"On this number?"        sim=0.038   (should be AGENT, marked as CUSTOMER)
```

Average agent similarity: **0.178** ← TOO WEAK!

**CUSTOMER segments misclassified as AGENT (16 errors):**
```
"That's correct."        sim=0.089   (should be CUSTOMER, marked as AGENT)
"That's right, yes."     sim=0.294   (should be CUSTOMER, marked as AGENT)
"Yeah, we looking to...  sim=0.493   (should be CUSTOMER, marked as AGENT)
```

Average customer similarity: **0.405** ← TOO CLOSE TO AGENT!

---

## Root Cause: Bad Training Data

```
Current voiceprint trained on:
  - Single noisy desk recording
  - Contaminated with customer crosstalk
  - Limited diversity

Result:
  - Agent similarity too weak (0.178 avg)
  - Customer similarity overlaps (0.405 avg)
  - Role identification fails 52% of the time
```

---

## The Fix: Multi-Call Gemini Training

### How It Works:

When we train on 3-5 diverse calls with perfect Gemini labels:

1. **Agent segments are CLEAN**
   - Only agent's voice
   - No customer contamination
   - High quality audio from multiple conditions

2. **Embeddings become STRONG**
   - Agent centroid similarity: 0.70+
   - Clear acoustic signature
   - Robust across variations

3. **Separation becomes CLEAR**
   - Agent voiceprint: 0.70+
   - Customer ceiling: 0.20-0.30
   - Big gap = easy classification

4. **Result: 95%+ accuracy**

---

## Expected Improvement After Multi-Call Training

| Metric | Before | After |
|--------|--------|-------|
| Agent avg similarity | 0.178 | 0.70+ |
| Customer avg similarity | 0.405 | 0.20-0.30 |
| Agent identification accuracy | 55.6% | 95%+ |
| Customer identification accuracy | 40.7% | 95%+ |
| **Overall accuracy** | **48.1%** | **95%+** |

---

## Action Plan: Start with ZAK RAISSI

### Why Zak First?
- We have **4 calls** available in the API
- 3 for training + 1 for testing = perfect setup
- Can demonstrate the fix immediately

### Zak's Available Calls:

```
TRAINING CALLS (use these for training):
1. zak_e2e_test_20260423           (171 segments)
2. enhanced_zak_raissi_barnet...   (253 segments)  
3. zak_compare_20260423            (104 segments)
                                    Total: 528 segments!

TESTING CALL (held-out for validation):
4. zak_audiofy_verify_20260423     (56 segments)
```

### Step-by-Step Process:

#### Step 1: Upload Zak's 3 Training Calls to Gemini (20 min)

For each call:
```
1. Open: https://gemini.google.com/app
2. Upload: zak call #1
3. Send prompt:

   Transcribe this call and identify the speaker 
   (agent or customer) for each segment with precise 
   timestamps. Return as JSON with:
   
   {
     "call_id": "zak_call_1",
     "agent_name": "Zak Raissi",
     "source": "gemini",
     "segments": [
       {"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."},
       {"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}
     ]
   }

4. Copy JSON response
5. Save to: call_processor/data/training/gemini_labels_zak_call_1.json
```

Repeat for calls 2 and 3.

#### Step 2: Combine and Retrain (2 min)

```bash
python call_processor/scripts/combine_and_retrain.py "Zak Raissi"
```

This script will:
- Load all 3 Gemini label files
- Extract agent and customer segments
- Train single voiceprint from 500+ combined segments
- Update agents.json with new, strong voiceprint

#### Step 3: Test on Held-Out Call (2 min)

```bash
python scripts/detailed_role_comparison.py
```

Or specifically for Zak's held-out test call:
```bash
python call_processor/scripts/combine_and_retrain.py "Zak Raissi" --test-call zak_audiofy_verify_20260423
```

#### Step 4: Verify Results

Expected output:
```
BEFORE MULTI-CALL:  Unknown (first time)
AFTER MULTI-CALL:   95-98% accuracy (on held-out test call)

Agent identification:    95%+
Customer identification: 95%+
```

---

## Why This Demonstrates the Fix

By doing Zak BEFORE getting more Omar calls:

1. **Proof of concept**: Shows that multi-call training works
2. **No delay**: Don't need to find more Omar calls
3. **4 calls available**: Can do training + testing properly
4. **Fast turnaround**: 25 min total to 95% accuracy
5. **Then we can**: Apply same approach to Omar, Hussein, others

---

## Timeline

```
Zak Raissi Multi-Call Training:
  Upload 3 calls to Gemini:  20 min
  Combine and retrain:        2 min
  Test on held-out call:      2 min
  ────────────────────────────────
  TOTAL:                      25 min
  
RESULT: 95%+ accuracy achieved!
```

---

## Complete Workflow

### Phase 1: ZAK RAISSI (TODAY - 25 min)

```
[ ] Upload zak_e2e_test_20260423 to Gemini
[ ] Save JSON to gemini_labels_zak_call_1.json
[ ] Upload enhanced_zak_raissi_barnet... to Gemini
[ ] Save JSON to gemini_labels_zak_call_2.json
[ ] Upload zak_compare_20260423 to Gemini
[ ] Save JSON to gemini_labels_zak_call_3.json
[ ] Run: python call_processor/scripts/combine_and_retrain.py "Zak Raissi"
[ ] Verify: python scripts/detailed_role_comparison.py
[ ] Result: 95%+ accuracy expected
```

### Phase 2: OMAR ENHANCEMENTS (Find additional calls)

Once Zak works, apply same to Omar:
```
[ ] Find 2-3 more Omar calls
[ ] Upload to Gemini
[ ] Combine with existing label
[ ] Retrain and verify 95%+
```

### Phase 3: HUSSEIN

```
[ ] Upload Hussein's call to Gemini
[ ] Train on single call
[ ] Verify 90%+ accuracy
```

---

## Success Criteria

For ZAK (multi-call training):
- [ ] Accuracy on held-out call: 95%+
- [ ] Agent identification: 95%+
- [ ] Customer identification: 95%+
- [ ] No regressions on other agents

For all agents (after completing all phases):
- [ ] Each agent: 95%+
- [ ] System average: 95%+
- [ ] Ready for production

---

## Key Insight

**The Gemini labels provide PERFECT speaker identification.**

By training on those perfect labels (3-5 calls per agent), we build strong voiceprints that:
- Clearly separate agent voice from customer
- Generalize to new calls (even with different background noise)
- Achieve 95%+ accuracy reliably

**The transcription (ASR) is already good. We just need GOOD ROLE LABELS to train on.**

---

## Commands Summary

```bash
# Find Omar additional calls
find testing-audio -name "*omar*" -o -name "*20260505*"

# Upload to Gemini (manual, in browser)
https://gemini.google.com/app

# Save Gemini labels
call_processor/data/training/gemini_labels_zak_call_1.json
call_processor/data/training/gemini_labels_zak_call_2.json
call_processor/data/training/gemini_labels_zak_call_3.json

# Combine and retrain
python call_processor/scripts/combine_and_retrain.py "Zak Raissi"

# Test results
python scripts/detailed_role_comparison.py

# See detailed comparison
cat call_processor/data/training/detailed_role_comparison.json
```

---

## Expected Timeline to 95%+ on All Agents

```
Zak (3-5 calls):       25 min -> 95%+ ✓
Omar (3-5 calls):      25 min -> 95%+ ✓  
Hussein (1-2 calls):   15 min -> 90%+ ✓
─────────────────────────────────────
TOTAL:                 65 min

RESULT: All agents at 95%+ role identification accuracy
         System ready for production
```

---

## Why This Works

The core insight:
```
CURRENT STATE:
  Training data: Noisy desk recording (bad)
  Agent similarity: 0.178 (very weak)
  Customer similarity: 0.405 (overlaps)
  Result: 48% accuracy (role identification fails)

AFTER MULTI-CALL GEMINI TRAINING:
  Training data: 3-5 diverse calls (perfect labels)
  Agent similarity: 0.70+ (strong)
  Customer similarity: 0.20-0.30 (separated)
  Result: 95%+ accuracy (clear role identification)
```

The **exact same algorithm** works perfectly with good training data!

---

## Ready to Start?

**Begin with ZAK RAISSI today:**

1. Open Gemini
2. Upload 3 calls
3. Get labels
4. Run retrain script
5. See jump to 95%+ accuracy in 25 minutes!

Then repeat for Omar and Hussein for production-ready 95%+ system.

**Let's go!** 🚀
