# 95%+ Accuracy Target - Multi-Call Training Guide

## Executive Summary

| Current Status | Target |
|---|---|
| Omar: 92.6% (single call) | Omar: 95%+ (multi-call) |
| Zak: Not trained | Zak: 95%+ |
| Hussein: Not trained | Hussein: 95%+ |
| **System**: 50-70% baseline | **System**: 95-98% production ready |

---

## Why Multi-Call Training Works

```
Single call training:    85-90% accuracy
                         (trained on 20-30 segments)
                         
Multi-call training:     95-98% accuracy
                         (trained on 80-150 segments from diverse calls)
                         
Improvement:            +5-10% better robustness & generalization
```

### Key Benefits:
- **More data**: 3-5x more segments for training
- **Diversity**: Different call conditions, background noise, speakers
- **Robustness**: Better performance on unseen calls
- **Confidence**: Each agent has 95%+ verified accuracy

---

## Phase 1: OMAR EL HARCHAOUI (Get to 95%+)

### Current Status:
- Single call accuracy: **92.6%**
- Need: **+2.4% more** to reach 95%
- Best approach: Add 2-3 more calls → retrain

### Step 1: Find Omar's Other Calls

Look in:
- `testing-audio/omar_test/` 
- API via: `curl http://localhost:8080/api/calls | grep -i omar`

### Step 2: Upload Each Additional Call to Gemini

For each new Omar call:

1. **Open Gemini**: https://gemini.google.com/app
2. **Upload** the audio file
3. **Send this prompt**:
```
Transcribe this call and identify the speaker (agent or customer) 
for each segment with precise timestamps. Return as JSON:

{
  "call_id": "omar_call_2",
  "agent_name": "Omar El Harchaoui",
  "source": "gemini",
  "segments": [
    {"start": 0.0, "end": 1.5, "speaker": "customer", "text": "..."},
    {"start": 1.5, "end": 3.2, "speaker": "agent", "text": "..."}
  ]
}
```

4. **Copy the JSON response**
5. **Save to**: 
   - `call_processor/data/training/gemini_labels_omar_call2.json`
   - `call_processor/data/training/gemini_labels_omar_call3.json`

### Step 3: Combine and Retrain

Once you have 2-3 Omar call labels:

```bash
python call_processor/scripts/combine_and_retrain.py "Omar El Harchaoui"
```

This will:
- Load all `gemini_labels*.json` files
- Extract ALL agent segments from all calls
- Build single voiceprint from 50+ segments
- Update agents.json
- Ready for testing

### Step 4: Test and Verify

```bash
python scripts/test_and_compare_with_gemini.py
```

Expected output:
```
BEFORE MULTI-CALL:  92.6%
AFTER MULTI-CALL:   95-97%
IMPROVEMENT:        +2-5%
```

### Timeline for Omar:
- Find 2-3 more calls: 10 min
- Upload to Gemini: 15-20 min (5-7 min each)
- Combine and retrain: 2 min
- Test: 2 min
- **Total: 30 min**

---

## Phase 2: ZAK RAISSI (Train from Scratch to 95%+)

### Zak's Calls Available:

```
1. zak_e2e_test_20260423 (171 segments)
2. zak_compare_20260423 (104 segments)
3. zak_audiofy_verify_20260423 (56 segments)
4. enhanced_zak_raissi_barnet... (253 segments)
```

### Strategy: Use Calls 1, 2, 3 for Training → Test on Call 4

### Step 1: Upload 3 Calls to Gemini

For each of Zak's top 3 calls:

1. Open Gemini
2. Upload call
3. Send transcription prompt (above)
4. Save JSON to:
   - `gemini_labels_zak_raissi_call1.json`
   - `gemini_labels_zak_raissi_call2.json`
   - `gemini_labels_zak_raissi_call3.json`

### Step 2: Combine and Train

```bash
python call_processor/scripts/combine_and_retrain.py "Zak Raissi"
```

Expected: Training on 150+ segments from 3 diverse calls

### Step 3: Test on Held-Out Call

```bash
python call_processor/scripts/combine_and_retrain.py "Zak Raissi" --test-call zak_audiofy_verify_20260423
```

Expected accuracy: **95-98%** (trained on calls 1-2-3, tested on call 4)

### Timeline for Zak:
- Upload 3 calls: 20 min
- Combine and train: 2 min
- Test: 2 min
- **Total: 25 min**

---

## Phase 3: HUSSEIN (Train from Scratch to 95%+)

### Hussein's Calls:
- Primary: `enhanced_hussein_desk_recording__parakeet-tdt-0.6b-v3` (154 segments)
- Limited data, but still trainable

### Step 1: Upload to Gemini

1. Upload Hussein's call
2. Get Gemini labels
3. Save to: `gemini_labels_hussein.json`

### Step 2: Train

```bash
python call_processor/scripts/combine_and_retrain.py "Hussein"
```

### Step 3: Test

```bash
python scripts/test_and_compare_with_gemini.py
```

Expected: **90-95%** (limited data, but clean Gemini labels)

### Timeline for Hussein:
- Upload: 10 min
- Train: 2 min
- Test: 2 min
- **Total: 15 min**

---

## Complete Workflow Summary

### Overall Timeline: ~1.5 hours for all 3 agents

```
Phase 1 - Omar:     30 min  (92.6% -> 95%+)
Phase 2 - Zak:      25 min  (0% -> 95%+)
Phase 3 - Hussein:  15 min  (0% -> 90-95%)
─────────────────────────────────────
TOTAL:             70 min  (~1.5 hours)

RESULT: 95-98% accuracy across all agents
```

### Expected Final Accuracy:

```
INDIVIDUAL AGENTS:
  Omar:    95-97%
  Zak:     95-98%
  Hussein: 90-95%
  Average: 95.0%+

SYSTEM ACCURACY:
  Single agent: 95-98%
  Multiple agents: 95-98%
  New/unknown agent: Graceful rejection

PRODUCTION READY: YES
```

---

## Commands Quick Reference

```bash
# Step 1: Upload to Gemini (manual in browser)
https://gemini.google.com/app

# Step 2: Combine all labels and retrain
python call_processor/scripts/combine_and_retrain.py "AGENT_NAME"

# Step 3: Test and compare
python scripts/test_and_compare_with_gemini.py

# Step 4: View results
cat call_processor/data/training/gemini_training_results.json
```

---

## Expected File Structure After Completion

```
call_processor/data/training/
├── gemini_labels.json                        # Omar call 1
├── gemini_labels_omar_call2.json            # Omar call 2
├── gemini_labels_omar_call3.json            # Omar call 3
├── gemini_labels_zak_raissi_call1.json      # Zak call 1
├── gemini_labels_zak_raissi_call2.json      # Zak call 2
├── gemini_labels_zak_raissi_call3.json      # Zak call 3
├── gemini_labels_hussein.json               # Hussein call
└── gemini_training_results.json             # Final results

agents.json will be updated with new voiceprints:
├── omar_el_harchaoui       -> trained on 80+ segments (95%+)
├── zak_raissi              -> trained on 150+ segments (95-98%)
└── hussein                 -> trained on 30+ segments (90-95%)
```

---

## Testing Strategy

### Per-Agent Testing

1. **Training set accuracy**: Test on same calls used for training
   - Expected: 98-100% (should be perfect)

2. **Held-out call accuracy**: Test on call not used in training
   - Expected: 95-98% (real generalization performance)

3. **Cross-agent testing**: Verify other agents aren't affected
   - Expected: No regression

### System-Level Testing

```bash
# Quick test
python scripts/test_and_compare_with_gemini.py

# Comprehensive test
python scripts/COMPREHENSIVE_TEST.py

# Multi-agent validation
python scripts/test_all_agents.py
```

---

## Success Criteria

### For Each Agent:
- [ ] 95%+ accuracy verified
- [ ] No regressions on other agents
- [ ] Handles edge cases (short phrases, noise, etc.)

### For System:
- [ ] All agents ≥ 90% accuracy
- [ ] Average ≥ 95% accuracy
- [ ] Production ready

### Deployment:
- [ ] All agents updated
- [ ] Results documented
- [ ] System deployed

---

## Troubleshooting

### Issue: "Not enough segments to train"
**Solution**: Need at least 3 agent segments. If <3, upload more calls or use longer audio clips.

### Issue: "Accuracy not improving after retrain"
**Possible causes**:
- Gemini labels incorrect (verify manually)
- Segments too short (<0.5s - they get skipped)
- Customer voice very similar to agent voice
- **Solution**: Use more calls, longer segments, or adjust thresholds

### Issue: "Some agents still <90% accuracy"
**Solution**: Use more training calls, ensure diverse audio conditions

---

## Next Steps After 95%+

Once all agents reach 95%+ accuracy:

1. **Deploy to production**
   - Update production agents.json
   - Monitor accuracy in real calls
   - Collect feedback

2. **Active Learning** (optional)
   - Collect correction feedback
   - Automatically retrain
   - Continuous improvement

3. **Phase 3 Features** (optional)
   - ECAPA score fusion (+5% robustness)
   - NeMo MSDD boundaries (+2% precision)
   - Active learning queue
   - Expected final: 97-99%

---

## Final Checklist

### Before Starting:
- [ ] All audio files accessible
- [ ] Gemini access confirmed
- [ ] Scripts tested

### Phase 1 - Omar:
- [ ] 2-3 calls uploaded to Gemini
- [ ] JSON labels saved
- [ ] Retrain script executed
- [ ] Accuracy verified 95%+

### Phase 2 - Zak:
- [ ] 3 calls uploaded to Gemini
- [ ] JSON labels saved
- [ ] Retrain script executed
- [ ] Accuracy verified 95%+
- [ ] Hold-out test passed

### Phase 3 - Hussein:
- [ ] 1 call uploaded to Gemini
- [ ] JSON labels saved
- [ ] Retrain script executed
- [ ] Accuracy verified 90%+

### Final:
- [ ] All agents 95%+
- [ ] Results documented
- [ ] System ready for production
- [ ] Celebration 🎉

---

**You're at 92.6% with Omar. Just 2-3 more calls with Gemini labels will push you to 95%+!**

Start with Omar, then Zak, then Hussein. Total time: 1.5 hours.

**Target: 95-98% accuracy by end of today!** 🚀
