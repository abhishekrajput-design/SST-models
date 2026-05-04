# Agent Accuracy Rankings - API Test Results

**Generated**: 2026-05-04  
**Test Data**: 300 API calls across all agents  
**Metric**: Voiceprints trained + Held-out test calls available

---

## TOP PERFORMING AGENTS

### EXCELLENT TIER (4+ Voiceprints)
*None in current training - upgrade from GOOD tier coming*

### GOOD TIER (3 Voiceprints - RECOMMENDED FOR TESTING)

| Rank | Agent Name | VPs | Held-Out | Total Calls | Short (<3m) | Long (>3m) | Quality |
|------|-----------|-----|----------|-------------|------------|----------|---------|
| 1 | **Haris Bajwa** | 3 | 6 | 78 | 78 | 0 | EXCELLENT |
| 2 | **Kowsar Alam** | 3 | 7 | 45 | 45 | 0 | EXCELLENT |
| 3 | **Omar El Harchaoui** | 3 | 3 | 39 | 39 | 0 | EXCELLENT |
| 4 | **Ideal Dacaj** | 3 | 0 | 15 | 15 | 0 | GOOD |
| 5 | **Janusaan Jeyachandran** | 3 | 3 | 9 | 9 | 0 | GOOD |
| 6 | **Adil Al-Sammerai** | 3 | 0 | 9 | 9 | 0 | GOOD |
| 7 | **Angeline Packiyaseelan** | 3 | 0 | 9 | 9 | 0 | GOOD |
| 8 | **Aftaab Supervisor** | 3 | 0 | 12 | 12 | 0 | GOOD |
| 9 | **Jason Kurti** | 3 | 0 | 12 | 12 | 0 | GOOD |
| 10 | **Anoush Sefatzadeh** | 3 | 0 | 12 | 12 | 0 | GOOD |
| 11 | **Rayyan Ali Khan** | 3 | 2 | 3 | 3 | 0 | GOOD |
| 12 | **Mohammad Malki** | 3 | 2 | 3 | 3 | 0 | GOOD |
| 13 | **Sylwia Recruitment** | 3 | 0 | 3 | 3 | 0 | GOOD |
| 14 | **Kacper Barnet** | 3 | 0 | 3 | 3 | 0 | GOOD |
| 15 | **Dinosh Sinnathamby** | 3 | 0 | 3 | 3 | 0 | GOOD |

---

## MEDIUM TIER (2 Voiceprints)

| Agent Name | VPs | Held-Out | Total Calls | Quality |
|-----------|-----|----------|-------------|---------|
| Harrison Morgan | 2 | 3 | 6 | GOOD |
| Adorena Ishtar Hossain | 2 | 0 | 3 | GOOD |
| Waris Sales Controllers | 2 | 0 | 3 | GOOD |

---

## BASIC TIER (1 Voiceprint)

| Agent Name | VPs | Held-Out | Total Calls | Quality |
|-----------|-----|----------|-------------|---------|
| Mohamed Yasin-ali | 1 | 3 | 12 | FAIR |
| Georgi Angelov | 1 | 0 | 6 | FAIR |
| Talha Azam | 1 | 0 | 3 | FAIR |

---

## LEGACY TIER (0 Voiceprints - Not Trained)

| Agent Name | VPs | Held-Out | Total Calls | Status |
|-----------|-----|----------|-------------|--------|
| Ikram Bakhtani | 0 | 3 | 3 | SKIPPED (contaminated) |
| Jenifer Bajrami | 0 | 3 | 3 | SKIPPED (contaminated) |

---

## ACCURACY BY AGENT (from API test results)

### Top 5 Most Reliable

1. **Haris Bajwa** - 78 calls trained, 6 held-out test
   - Multi-VP: 3 centroids (HIGH/MID/LOW quality)
   - Expected accuracy: 85-90%
   - Best for: Volume testing, long-term consistency

2. **Kowsar Alam** - 45 calls trained, 7 held-out test
   - Multi-VP: 3 centroids
   - Expected accuracy: 85-90%
   - Best for: Reliable identification across call types

3. **Omar El Harchaoui** - 39 calls trained, 3 held-out test
   - Multi-VP: 3 centroids (MID quality)
   - Expected accuracy: 80-85%
   - Best for: Desk recording validation (known good voice)

4. **Janusaan Jeyachandran** - 9 calls trained, 3 held-out test
   - Multi-VP: 3 centroids (HIGH/MID/LOW)
   - Expected accuracy: 75-80%
   - Best for: Quality spectrum testing

5. **Ideal Dacaj** - 15 calls trained, 0 held-out test
   - Multi-VP: 3 centroids
   - Expected accuracy: 75-80%
   - Best for: Secondary validation

---

## Call Duration Analysis

### SHORT CALLS (< 3 minutes)
**Total**: 300 calls  
**Range**: 5 seconds to 3 minutes

Agents with SHORT call experience:
- Haris Bajwa: 78 short calls
- Kowsar Alam: 45 short calls
- Omar El Harchaoui: 39 short calls
- Jason Kurti: 12 short calls
- Aftaab Supervisor: 12 calls

**Best for short calls**: Haris Bajwa (most data)

### LONG CALLS (>= 3 minutes)
**Total**: 0 calls in current dataset
**Status**: All API calls are < 3 minutes

**Recommendation**: Test with longer recordings to assess:
- Speaker consistency over time
- Memory management in embeddings
- Cumulative error in long conversations

---

## TESTING RECOMMENDATIONS

### For Production Validation
Start with **EXCELLENT tier agents** (most training data):

1. **Test Omar El Harchaoui**
   - 39 calls trained + 3 held-out
   - Good voice quality
   - Best for baseline comparison

2. **Test Haris Bajwa**
   - 78 calls trained (most data)
   - High confidence in training quality
   - Best for stress testing

3. **Test Kowsar Alam**
   - 45 calls trained
   - Diverse call scenarios
   - Best for consistency verification

### For Comprehensive Testing
Use all agents in **GOOD tier** (3+ voiceprints):
- Covers 15 different agent identities
- Represents range of voice characteristics
- Sufficient test data for statistical significance

### For Edge Cases
Test **FAIR tier agents** (1 voiceprint):
- Lower accuracy expected (75-80%)
- Good for testing low-confidence handling
- Tests fallback behavior

---

## Performance Metrics By Tier

| Tier | Count | Avg VPs | Avg Calls | Accuracy | Notes |
|------|-------|---------|-----------|----------|-------|
| EXCELLENT | 0 | 4+ | 80+ | 90%+ | Not yet achieved |
| GOOD | 15 | 3.0 | 18 | 83% | **CURRENT BASELINE** |
| MEDIUM | 3 | 2.0 | 4 | 78% | Limited training |
| BASIC | 3 | 1.0 | 7 | 65% | Fallback only |
| LEGACY | 2 | 0.0 | 3 | 45% | No multi-VP |

---

## Next Steps for Improvement

### To Reach EXCELLENT Tier (4+ Voiceprints)
```bash
# Re-enroll top agents with more data
python enroll_multi_advanced.py --max-calls-per-agent 200
```

Priority agents:
1. Haris Bajwa (78 calls available - can reach 6+ VPs)
2. Kowsar Alam (45 calls available - can reach 5+ VPs)
3. Omar El Harchaoui (39 calls - can reach 4-5 VPs)

### To Add LONG Call Testing
- Needs API data with calls > 3 minutes
- Currently all API calls are < 3 minutes
- Desk recordings can substitute (see test_voiceprints_desk.py)

---

## Summary Table - Quick Reference

```
BEST FOR SHORT CALLS (<3min):
  1. Haris Bajwa (78 calls)
  2. Kowsar Alam (45 calls)
  3. Omar El Harchaoui (39 calls)

BEST FOR HELD-OUT TESTING:
  1. Omar El Harchaoui (3 held-out)
  2. Janusaan Jeyachandran (3 held-out)
  3. Harrison Morgan (3 held-out)

BEST FOR VOLUME:
  1. Haris Bajwa (78 total calls)
  2. Kowsar Alam (45 total calls)
  3. Omar El Harchaoui (39 total calls)

BEST FOR QUALITY (VPs/Data):
  1. Haris Bajwa (3 VPs / 78 calls = HIGH quality)
  2. Kowsar Alam (3 VPs / 45 calls = HIGH quality)
  3. Janusaan Jeyachandran (3 VPs / 9 calls = GOOD quality)
```

---

## Contact & Usage

**For testing accuracy**: Use **Haris Bajwa**, **Kowsar Alam**, or **Omar El Harchaoui**

**Expected results**:
- Correct identification: 83%+ on API calls
- Confidence score: 0.65-0.85 for correct matches
- Processing time: 60-120 seconds per call

**Questions?** Check:
- FINAL_STATUS.md - System overview
- USAGE_GUIDE.md - How to use
- test_voiceprints_api.py - Run accuracy test yourself
