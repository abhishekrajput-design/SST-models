# Speaker Identification System - Current State & Improvement Plan

**Date**: 2026-05-05  
**Status**: Hotfixes A-D implemented and tested

---

## Current Performance

### Test Call: 6-minute Omar El Harchaoui + Customer Mark call
- **Duration**: 6:11 (372 seconds)
- **Segments**: 132 (from Parakeet ASR)
- **Speakers**: 2 (AGENT: Omar, CUSTOMER: Mark)

### Baseline Accuracy (Before Hotfixes)
- Omar identification: ~38.6%
- Root cause: Embedding contamination + short segment failures

### Current Accuracy (With Hotfixes A-D)
- Hotfix A (embedding-failed segment protection): ✓ Implemented
- Hotfix B (Omar re-enrollment from GT): ✓ Implemented (but circular - trained on same call)
- Hotfix C (threshold lowering 0.40→0.36): ✓ Implemented
- Hotfix D (cluster_first protection): ✓ Implemented
- **Result**: ~41.7% (marginal improvement, as expected)

### Why Hotfixes Alone Aren't Enough

The fundamental issue is **embedding quality gap**:

```
AGENT segments:    High quality: 0.70-0.79 (✓ 100% correct)
                   Low quality:  0.03-0.34 (✗ 0% correct)

CUSTOMER segments: All range:    0.50-0.76 (⚠ 92% misidentified as AGENT)
```

When customer similarity overlaps with poor-quality agent segments, no threshold can separate them without introducing false positives.

---

## Problem Root Causes

1. **Enrollment Data Quality**
   - Original Omar voiceprints trained on noisy desk recordings
   - Contained customer voice crosstalk
   - Only 0.5-2 minutes of training data

2. **Circular Re-enrollment**
   - Re-enrollment script extracted segments from THIS SAME TEST CALL
   - Created overfitting to call-specific acoustics
   - No improvement on independent calls

3. **Short Segment Weakness**
   - Embedding extraction fails for <0.5s phrases
   - "Five o'clock." (0.5s) gets sim=0.028
   - Hotfix A protects but can't create good embeddings from insufficient audio

---

## Solutions & Expected Improvements

### Phase 1: Quick Wins (This Week) - Estimated +5-10%

✅ **Already Done:**
- Hotfixes A-D (✓ code complete)
- CAM++ 512-dim embeddings (✓ production ready)

**To Implement:**
- ECAPA-TDNN fusion (complementary embeddings)
  - Better robustness to channel noise
  - Different error patterns than CAM++
  - Expected: +5-8%

- Temporal voting (10-second windows)
  - Use neighbor context to fix isolated errors
  - Expected: +3-5%

**Result**: ~50-55% → ~60-70% on Omar's call

---

### Phase 2: Proper Re-enrollment (2 Weeks) - Estimated +15-25%

**Current Problem**: Only trained on 8 segments from 1 call

**Solution**: Re-enroll from independent call recordings
- Extract 50-100 agent segments from 5-10 different calls
- Each call provides diverse acoustic conditions
- Avoid circular training (don't use test call for training)
- Expected: +15-25%

**Result**: ~60-70% → ~85-90% on Omar's call

---

### Phase 3: Production Upgrade (4 Weeks) - Estimated +5-10%

**Full 10-step plan:**
1. NeMo MSDD for better speaker boundaries
2. Consensus boundary selection (MSDD + Pyannote)
3. ECAPA + CAM++ fusion
4. Temporal voting
5. Unknown speaker rejection
6. Confidence gating
7. Active learning queue
8. Unknown rejection gates
9. Threshold optimization
10. Production deployment

**Result**: ~85-90% → ~95-98% across all call durations

---

## Data Available

**API Contains**: 100+ processed calls with:
- Parakeet TDT v3 ASR transcriptions
- Segment boundaries and text
- Speaker counts (mostly 2-speaker calls)
- Call durations: 20s - 30+ minutes
- Quality: Mix of call center, desk recordings, clean calls

**Limitation**: No ground truth speaker labels
- Can use system predictions as weak labels
- Can manually verify select calls for validation

---

## Recommended Action Plan

1. **This Week**
   - [ ] Implement ECAPA-TDNN fusion
   - [ ] Add temporal voting (10s windows)
   - [ ] Test on Omar's call
   - [ ] Target: ~65-70% accuracy

2. **Next Week**
   - [ ] Extract 50+ agent segments from API calls (5-10 different calls)
   - [ ] Re-enroll all agents with diverse data
   - [ ] Test accuracy improvement
   - [ ] Target: ~85-90% accuracy

3. **Week 3-4**
   - [ ] Implement NeMo MSDD + consensus boundaries
   - [ ] Add confidence gating + unknown rejection
   - [ ] Full system test across multiple call types
   - [ ] Target: ~95%+ accuracy

---

## Technical Details

### Hotfix A: Embedding-Failed Segment Protection
- **File**: `call_processor/src/diar_multi.py`
- **Lines**: 1150-1152 (flag), 1438-1446 (protection)
- **Status**: ✓ Implemented
- **Effect**: Prevents short segments from false demotion when embeddings fail

### Hotfix B: Omar Re-enrollment from GT
- **File**: `call_processor/scripts/enroll_omar_from_gt.py`
- **Status**: ✓ Implemented (but circular - needs independent data)
- **Result**: mean_inside_sim=0.6921, max_outside_sim=0.4039

### Hotfix C: Threshold Calibration
- **File**: `call_processor/src/diar_multi.py` line 30
- **Change**: PER_AGENT_THRESH_CAP: 0.40 → 0.36
- **Status**: ✓ Implemented

### Hotfix D: cluster_first Mode Protection
- **File**: `call_processor/src/diar_multi.py` lines 474-503
- **Status**: ✓ Implemented
- **Effect**: Protects unembeddable segments in cluster_first mode

---

## Files Modified

### Core Diarization
- ✓ `call_processor/src/diar_multi.py` - All hotfixes + cluster_first protection

### Scripts Created
- ✓ `call_processor/scripts/enroll_omar_from_gt.py` - Omar re-enrollment
- ✓ `call_processor/scripts/bulk_reenroll_from_api.py` - Bulk re-enrollment framework
- ✓ `call_processor/scripts/test_accuracy_on_api_calls.py` - API testing

### Test Results
- ✓ `c:/Users/abhis/Desktop/SST-models/testing-audio/omar_test/realtrancription.md`
- ✓ `c:/Users/abhis/Desktop/SST-models/testing-audio/omar_test/our_system_trancription.md`
- ✓ `c:/Users/abhis/Desktop/SST-models/testing-audio/omar_test/COMPARISON_AND_ANALYSIS.md`

---

## Next Immediate Steps

1. **Clarify Training Data**: 
   - Do you have access to manually-corrected transcriptions for calls in the API?
   - Can you provide 5-10 calls with verified speaker labels for re-enrollment?

2. **Pick Implementation Focus**:
   - Quick wins (ECAPA + temporal voting) for immediate improvement?
   - Or invest time in proper re-enrollment first?
   - Or both in parallel?

3. **Deployment Target**:
   - Current 16GB GPU hardware - all proposed solutions fit within memory
   - Production deployment timeline?
   - Accuracy threshold requirement?

---

## Expected Timeline to Production

| Target | Effort | Expected Accuracy |
|--------|--------|------------------|
| This week | 2-3 hours | 65-70% (ECAPA + temporal voting) |
| Next week | 5-8 hours | 85-90% (proper re-enrollment) |
| Week 3-4 | 15-20 hours | 95%+ (full upgrade + testing) |

---

**Generated**: 2026-05-05  
**System**: SST-models Speaker Identification Pipeline  
**Status**: Ready for next phase of improvements
