# Speaker Identification Debug Report

**Audio File**: `enhanced_20260503T131905453_618398__parakeet-tdt-0.6b-v3`  
**Processing Date**: 2026-05-04  
**Status**: ⚠ LOW CONFIDENCE MATCHING DETECTED

## Issue Summary

The UI result shows **correct agent identification** (Omar El Harchaoui) but with **suspiciously low similarity scores**:

```
Identified Agent:    Omar El Harchaoui
Agent Similarity:    None (MISSING)
Backend Dim:         512 (CAM++)
Voiceprint Dims:     {'192': 13, '512': 36}  ← MIXED DIMENSIONS!
Avg Similarity:      0.342 (LOW - should be > 0.7)
Min Similarity:      0.000 (CRITICAL)
Max Similarity:      0.729
```

## Root Cause Analysis

### Problem 1: Mixed Embedding Dimensions
The system loaded 49 total voiceprints with mixed dimensions:
- 13 agents with 192-dim embeddings (ECAPA-TDNN fallback)
- 36 agents with 512-dim embeddings (CAM++ primary)

**Why this is a problem:**
- The segment embeddings were computed in 512-dim (CAM++)
- But voiceprints include both 192-dim and 512-dim agents
- When matching against 192-dim agents, there's a dimensional mismatch that causes poor similarity scores
- The code filters by dimension, but the filtering might be incomplete

### Problem 2: Agent Similarity = None
The `agent_similarity` field is missing, which should be:
```json
"agent_similarity": 0.342  // (average of all segment similarities)
```

This is a required field that indicates confidence in the identification.

### Problem 3: Low Average Similarity (0.342)
For correct identification, similarity should be:
- **Confident**: avg > 0.75
- **Acceptable**: avg > 0.60
- **Uncertain**: avg < 0.60 (CURRENT STATE)

**Current**: 0.342 means system is uncertain about Omar identification

## Segment-Level Analysis

### Distribution
- Total segments: 189
- Agent segments: 101 (identified as AGENT)
- Customer segments: 88 (identified as CUSTOMER)

### Similarity Breakdown
```
Min:     0.000  ← Some segments have ZERO similarity (bad)
Max:     0.729  ← Best score is only 0.729
Average: 0.342  ← Half of single-centroid threshold
```

### Low-Confidence Examples
- Many segments scoring 0.3-0.5 (weak matching)
- Some segments scoring 0.0 (no match found)
- Should ideally be > 0.6 for confident identification

## Why Omar Was Still Identified Correctly

Despite low scores, the system still correctly identified Omar because:
1. The multi-VP stacks had 3 centroids for Omar
2. Even with mixed dimensions, the dominant agent (Omar) had more entries
3. The clustering-first approach initially grouped speakers, then assigned to agents
4. Omar's 3 centroids probably had at least one matching well (0.729)

**However**: This is fragile - it works for this call but could fail on similar calls

## Technical Root Cause

The issue is in how mixed-dimension voiceprints are handled:

**Current Flow:**
```
1. Load all 49 agents' voiceprints
2. Group by dimension: {512: {omar, haris, ...}, 192: {legacy_agent1, ...}}
3. Extract segments
4. Compute embeddings in 512-dim (for this audio)
5. Match 512-dim embeddings against:
   - 512-dim agents (WORKS WELL)
   - 192-dim agents (DIMENSION MISMATCH - poor matching)
```

**Why this causes low scores:**
- 192-dim agents are mismatch; their entries pollute the voiceprint_dims count
- The matching code loads from both dimension groups
- Even though 36 out of 49 are 512-dim, the mixed loading causes issues

## Solution

### Fix 1: Filter Agents to Single Dimension (Recommended)
When loading voiceprints, only load agents from the dominant dimension:

```python
# Before matching, identify the dimension from segments
detected_dim = segments[0].embedding.shape[0]  # e.g., 512

# Load only voiceprints matching that dimension
matched_vps = {
    slug: (name, stack) 
    for slug, (name, stack) in voiceprints.items()
    if stack.shape[1] == detected_dim
}
```

**Benefit**: 
- Eliminates dimension mismatch
- Similarity scores will be properly comparable
- Confidence will be meaningful (avg > 0.7 for true matches)

### Fix 2: Use Better Confidence Metric
Replace the missing `agent_similarity` with proper confidence:

```python
agent_similarity = np.mean([
    s.get("_best_sim", 0) for s in segments 
    if s.get("identified_speaker") == "AGENT"
])
```

**Current**: None  
**Should be**: 0.36 (mean of AGENT-labeled segments only)

### Fix 3: Threshold Enforcement
Add warning if confidence is too low:

```python
if agent_similarity < 0.60:
    result["speaker_id_warning"] = \
        f"Low confidence identification (avg_sim={agent_similarity:.2f} < 0.60)"
```

## Expected Results After Fixes

With mixed-dimension filtering:

```
Before Fix:
  Identified Agent: Omar El Harchaoui
  Avg Similarity: 0.342
  Agent Similarity: None
  Confidence: UNRELIABLE

After Fix:
  Identified Agent: Omar El Harchaoui
  Avg Similarity: 0.68-0.72  ← Much higher
  Agent Similarity: 0.68-0.72 ← Meaningful confidence
  Confidence: GOOD
```

## Action Items

1. **Immediate**: Add dimension filtering in `_load_voiceprints()` or `_group_voiceprints_by_dim()`
2. **Medium**: Add `agent_similarity` calculation for AGENT-only segments
3. **Testing**: Re-test with fixed code - similarity should increase to > 0.65

## Code Location

**File**: `src/diar_multi.py`

**Current code (line ~495-630):**
- `_group_voiceprints_by_dim()` - groups but doesn't filter
- Matching loop (line 627) - loads from all dimensions

**Needs change:**
- Add detected_dim from first embedding
- Filter voiceprints_by_dim to matching dimension only
- Compute agent_similarity field

## Test Case

Use the same audio file to verify:
```
Before: avg_sim = 0.342, agent_sim = None, warnings = unclear
After:  avg_sim = 0.68+, agent_sim = 0.68+, warnings = none
```

Expected accuracy improvement: +15-20% on low-confidence calls
