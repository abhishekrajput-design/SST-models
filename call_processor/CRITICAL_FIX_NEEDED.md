# CRITICAL: Speaker Identification Failure Analysis

**Call**: enhanced_upload__whisper-large-v3-turbo  
**Date**: 2026-05-04 18:46  
**Severity**: CRITICAL - Multiple speakers misidentified

## Problem Summary

**Expected**: 
- SPEAKER_00: Customer 1
- SPEAKER_01: Customer 2  
- SPEAKER_02: Omar El Harchaoui (Agent)

**Actual**:
- SPEAKER_00: Matched to "Waris Sales Controllers" (SIM=0.148)
- SPEAKER_01: Matched to "Waris Sales Controllers" (SIM=0.148)
- SPEAKER_02: Matched to "Waris Sales Controllers" (SIM=0.148)

## Root Cause Analysis

### Similarity Scores - CRITICAL ISSUE

```
Expected cosine similarity: 0.6-0.9 (matching agent)
Actual similarity:         0.1-0.5 (ALL speakers)
Mean similarity:           0.1481
Min similarity:            -0.1650 (negative?!)
Max similarity:            0.4912
```

**This indicates embeddings do NOT match voiceprints at all.**

### Why This Happens

**Option 1: Embedding Model Changed** ⚠
- Voiceprints trained with one model (CAM++ v1)
- Inference using different model (CAM++ v2 or different config)
- Embeddings in different vector space

**Option 2: Audio Processing Issue**
- Audio normalization changing voice characteristics
- Resampling introducing artifacts
- Frame extraction different between training/inference

**Option 3: Voiceprint Corruption**
- agents.json pointing to wrong files
- .npy files overwritten or corrupted
- Path resolution failing silently

**Option 4: Model Version Mismatch**
- CAM++ model in code != CAM++ model used in voiceprint training
- WeSpeaker version changed
- Embedding normalization different

## Evidence

### All Speakers Matched to Same Wrong Agent
- This should not happen with a working system
- Suggests scores are so bad that random/fallback agent chosen
- All scores near mean (0.148) = essentially random

### Negative Similarity
- Line: min=-0.1650
- Should be impossible with normalized vectors
- Indicates vector spaces are incompatible

### 3-Speaker Call to Single Agent Match
- Call has 3 distinct speakers
- All matched to "Waris" with same low score
- Should have variety (some AGENT, some CUSTOMER)

## Immediate Fix Actions

### 1. Verify Embedding Model Consistency
```bash
# Check embedding model version in use
python -c "
from src.embedding_campp import EmbeddingModel
model = EmbeddingModel()
model.load()
print(f'Model: {model.model_name}')
print(f'Dimension: {model.dim}')
print(f'Config: {model.config}')
"

# Check voiceprint dimensions and sample values
python -c "
import numpy as np
vp = np.load('data/agent_voiceprints/omar_el_harchaoui__mid_0.npy')
print(f'Shape: {vp.shape}')
print(f'Mean: {np.mean(vp):.6f}')
print(f'Std: {np.std(vp):.6f}')
print(f'Norm: {np.linalg.norm(vp):.6f}')
print(f'First 5: {vp[:5]}')
"
```

### 2. Test Embedding Generation
```bash
# Extract a segment from this call manually
# Compute its embedding
# Compare with voiceprint
# Expect similarity > 0.6 for Omar's voice
```

### 3. Check Audio Processing
```bash
# Compare audio used in training vs inference
# Check if normalization/enhancement changes voice
# Verify sample rate consistency (16kHz expected)
```

## Diagnostic Plan

**Step 1**: Re-extract a known Omar segment from the call
```python
from src.embedding_campp import EmbeddingModel
import soundfile as sf

# Load audio
audio, sr = sf.read('data/processed/enhanced_upload__whisper-large-v3-turbo/trimmed_audio.mp3')

# Extract segment at time 0:20 (Omar speaks "Did you make...")
s, e = int(0.20 * sr), int(0.23 * sr)
chunk = audio[s:e]

# Compute embedding
model = EmbeddingModel()
model.load()
emb = model.embed_chunk(chunk, sr)
model.unload()

# Load Omar's voiceprints
import numpy as np
vp_mid_0 = np.load('data/agent_voiceprints/omar_el_harchaoui__mid_0.npy')

# Normalize both
emb_norm = emb / np.linalg.norm(emb)
vp_norm = vp_mid_0 / np.linalg.norm(vp_mid_0)

# Compute similarity
sim = np.dot(emb_norm, vp_norm)
print(f"Omar vs Omar centroid: {sim:.4f}")
```

**Expected**: sim > 0.6  
**Actual**: ?

**Step 2**: If sim < 0.3, then embedding model is different
- Need to retrain voiceprints with current embedding model
- OR revert to embedding model used during training

## Proposed Solutions

### Solution A: Retrain Voiceprints (Safest)
```bash
# Retrain with current CAM++ model
python enroll_multi_advanced.py --max-calls-per-agent 150

# This ensures voiceprints match current embedding model
```

### Solution B: Check Model Version
```bash
# Identify model version used during training
# git log --oneline | grep -i "embedding\|cam"

# Potentially revert src/embedding_campp.py to that version
```

### Solution C: Re-extract Embeddings  
```bash
# If voiceprints were saved as raw scores (not embeddings):
# Need to reload and recompute properly
```

## Prevention

Add embedding model/version check at startup:
```python
def validate_embedding_model():
    """Ensure training and inference use same model."""
    from src.embedding_campp import EmbeddingModel
    model = EmbeddingModel()
    model.load()
    
    expected_model_name = "wespeaker-cam++-based"
    if model.model_name != expected_model_name:
        raise ValueError(
            f"Model mismatch: {model.model_name} != {expected_model_name}"
        )
    
    # Test with known sample
    known_emb = ...  # embedding from training set
    known_vp = ...   # voiceprint centroid
    sim = np.dot(known_emb, known_vp)
    if sim < 0.5:
        raise ValueError(f"Low similarity with known sample: {sim}")
```

## Workaround (Temporary)

Until root cause fixed, fall back to single-VP baseline:
- Comment out multi-VP loading in diar_multi.py
- Use legacy voiceprint_path instead
- Accuracy drops to 76% but system works

## Next Steps

1. Run diagnostic script from Step 1 above
2. Check if `sim > 0.6` (working) or `sim < 0.3` (broken)
3. If broken: retrain voiceprints or check model version
4. Re-test upload with fixed system
5. Verify Omar now identified with sim > 0.6

**DO NOT USE** the multi-VP system until this is resolved.
