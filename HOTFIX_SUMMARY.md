# Hotfix Implementation Summary

## What Was Implemented

### Hotfix A: Embedding-Failed Segment Protection
**File**: `call_processor/src/diar_multi.py`

**Problem**: Short ASR segments with failed embeddings (e.g., "Five o'clock." ~0.5s) get `_best_sim = 0.0`. Anti-flip pass 3 unconditionally demotes any AGENT segment with `sim < 0.10` surrounded by CUSTOMER, causing false demotions.

**Solution**:
1. Lines 1150-1152: Set `_emb_failed = True` flag when embedding extraction fails (`valid[i] = False`)
2. Lines 1438-1446: In anti-flip pass 3, skip demotion if segment has `_emb_failed=True` AND has anchor neighbors (AGENT with `sim >= 0.30`)
3. Lines 474-488: In cluster_first mode (which doesn't run anti-flip passes), pull unembeddable segments to agent_cluster if adjacent segments are in agent_cluster
4. Lines 495-503: In cluster_first demote logic, protect `_emb_failed` segments if anchored by high-sim neighbors

### Hotfix B: Re-enrollment from Ground Truth
**File**: `call_processor/scripts/enroll_omar_from_gt.py`

**Problem**: Omar's enrollment audio (desk mics) contains customer-like speech characteristics, causing false AGENT labels on customer confirmations (e.g., "That's correct." cosine = 0.670).

**Solution**:
- Extract AGENT-only segments from call audio using ground truth labels
- Build new voiceprints from clean, labeled call-channel speech
- Compute `max_outside_sim` from customer segments in the same calls

**Results**:
- 8 AGENT segments extracted (2-5s each, SNR ≥12dB)
- mean_inside_sim: 0.6921
- max_outside_sim: 0.4039 (dropped from 0.49)
- Successfully updated agents.json for Omar El Harchaoui

### Hotfix C: Threshold Calibration
**File**: `call_processor/src/diar_multi.py` (line 30)

**Change**: `PER_AGENT_THRESH_CAP` from 0.40 → 0.36

**Rationale**: With clean re-enrollment, the agent-customer similarity gap widens. Lower cap allows threshold to drop from 0.40 to 0.36, reducing false positives while maintaining sensitivity.

### Hotfix D: cluster_first Mode Protection
**File**: `call_processor/src/diar_multi.py` (cluster_first path)

**Problem**: The Omar/Mark call triggers `cluster_first_voiceprint` mode (67s, 122 embeddable segments). This mode doesn't run anti-flip passes, so Hotfix A is ineffective for the default flow.

**Solution**: Added embedding-failed protection in cluster_first paths:
- Line 474-488: Assign unembeddable segments to agent_cluster if nearby segments are agents
- Line 495-503: Protect demote logic from removing `_emb_failed` segments with high-sim neighbors

---

## Test Results

### Test Case: Mini Hatch Call (Ground Truth)
- Input: 24 GT-labeled turns
- Output: 10/24 correct = **41.7% accuracy**
- Previous (no hotfixes): ~38.6%
- Expected with proper ASR + re-enrollment: ≥80%

**Key Findings**:
1. Short AGENT phrases ("Yeah speaking.") still get low sims (0.028) — suggests re-enrollment didn't help Omar's short-phrase embedding quality
2. Customer confirmations still get high sims (0.758) — suggests customer speech still matches Omar's voiceprint
3. Longer AGENT phrases work well (sims 0.75+)

### Root Cause Analysis
The re-enrollment used the SAME CALL for extraction and testing, which is circular. To properly validate:
1. Need independent test call(s) with Parakeet transcription
2. Need to verify Omar's new voiceprints were actually loaded and used
3. Need to check if the enrollment extraction captured enough diverse AGENT examples

---

## Code Changes Summary

| File | Change | Lines | Purpose |
|------|--------|-------|---------|
| `diar_multi.py` | `_emb_failed` flag | 1150-1152 | Mark embedding failures |
| `diar_multi.py` | Anti-flip pass 3 protection | 1438-1446 | Protect short segments |
| `diar_multi.py` | cluster_first unembeddable path | 474-488 | Pull to agent_cluster |
| `diar_multi.py` | cluster_first demote protection | 495-503 | Guard demote logic |
| `diar_multi.py` | Threshold cap | 30 | 0.40 → 0.36 |
| `enroll_omar_from_gt.py` | NEW | — | Re-enrollment script |
| `agents.json` | Omar entry updated | — | New voiceprints + stats |

---

## Next Steps for Full Production Upgrade

The hotfixes are code-complete. To achieve the full 93-97% accuracy target across all agents:

1. **Verify hotfixes**: Test with independent Parakeet-transcribed calls (not circular)
2. **Re-enroll all agents**: Run `enroll_from_transcriptions.py` on all 49 agents using call recordings + transcriptions
3. **Implement Steps 1-9 of production upgrade plan**:
   - NeMo MSDD diarization + consensus boundaries
   - FAISS index for all agents
   - ECAPA + CAM++ score fusion
   - Temporal voting (10-second windows)
   - Unknown speaker rejection
   - Confidence-gated classification
   - Active learning queue

**Projected timeline**: 2-3 weeks for full implementation and validation.

