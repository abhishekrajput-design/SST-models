#!/usr/bin/env python
"""
Test diarization accuracy on calls from the API.
Shows baseline accuracy across multiple real calls.
"""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, "call_processor")

from src.diar_multi import diarize_multi
from src.voiceprints import load_agents_index

API_ENDPOINT = "http://localhost:8080/api/calls"
AGENTS_JSON = "call_processor/data/agent_voiceprints/agents.json"

print("=" * 100)
print("TESTING DIARIZATION ACCURACY ON API CALLS")
print("=" * 100)

# Get all processed calls
print("\nFetching processed calls from API...")
try:
    response = requests.get(API_ENDPOINT, timeout=10)
    all_calls = response.json()
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Filter to calls with segments (Parakeet-processed)
valid_calls = [
    c for c in all_calls
    if c.get("model") == "parakeet-tdt-0.6b-v3"
    and c.get("segments", 0) >= 20  # At least 20 segments for meaningful test
    and c.get("segments", 0) <= 200  # Not too long
]

print(f"Found {len(valid_calls)} valid calls (20-200 segments)")

# Load agents
agents = load_agents_index(AGENTS_JSON)
print(f"Loaded {len(agents)} agents")

# Test on first 10 calls
print("\n" + "-" * 100)
print("Testing first 10 calls...")
print("-" * 100 + "\n")

results = []

for i, call_info in enumerate(valid_calls[:10]):
    call_id = call_info.get("id", "")
    audio_file = call_info.get("orig_file", "")
    n_segments = call_info.get("segments", 0)

    # Get call details
    try:
        response = requests.get(f"http://localhost:8080/api/call/{call_id}", timeout=10)
        call_data = response.json()
        segments = call_data.get("segments", [])
    except:
        print(f"{i+1}. {call_id[:30]:30s} - ERROR fetching details")
        continue

    if not segments or not Path(audio_file).exists():
        print(f"{i+1}. {call_id[:30]:30s} - SKIP (no segments or file)")
        continue

    # Prepare segments for diarization
    diar_segments = [
        {
            "start": float(s.get("start", 0)),
            "end": float(s.get("end", 0)),
            "text": s.get("text", ""),
            "speaker": s.get("speaker", "SPEAKER_00"),
            "identified_speaker": "UNKNOWN",
            "confidence": 0.0,
        }
        for s in segments
    ]

    # Run diarization
    try:
        result = diarize_multi(
            diar_segments,
            norm_wav=audio_file,
            threshold=0.25,
            agents_index_path=AGENTS_JSON,
            force_cpu=False,
        )
        out_segments = result.get("segments", diar_segments)
        agent_name = result.get("agent_name", "Unknown")
        mode = result.get("speaker_mode", "unknown")

        # Count matches (assume every other segment should be AGENT in 2-speaker calls)
        # This is a rough estimate without ground truth
        agent_count = sum(1 for s in out_segments if "AGENT" in s.get("identified_speaker", ""))
        speaker_ratio = agent_count / len(out_segments) if out_segments else 0

        results.append({
            "call_id": call_id,
            "n_segments": len(out_segments),
            "agent_identified": agent_name,
            "mode": mode,
            "agent_ratio": speaker_ratio,
        })

        print(f"{i+1}. {call_id[:30]:30s} | {len(out_segments):3d} segs | "
              f"Agent: {agent_name[:20]:20s} | Mode: {mode:20s} | Agent%: {speaker_ratio*100:5.1f}%")

    except Exception as e:
        print(f"{i+1}. {call_id[:30]:30s} - ERROR: {str(e)[:50]}")
        continue

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

if results:
    print(f"\nProcessed {len(results)} calls successfully\n")

    avg_agent_ratio = np.mean([r["agent_ratio"] for r in results])
    identified_agents = [r["agent_identified"] for r in results if r["agent_identified"] != "Unknown"]

    print(f"Average Agent % across calls: {avg_agent_ratio*100:.1f}%")
    print(f"Identified agents: {set(identified_agents) if identified_agents else 'None'}")

    # Show calls by identified agent
    print("\nBy Agent:")
    for agent in set(r["agent_identified"] for r in results):
        agent_calls = [r for r in results if r["agent_identified"] == agent]
        avg_ratio = np.mean([r["agent_ratio"] for r in agent_calls])
        print(f"  {agent}: {len(agent_calls)} calls, avg agent%={avg_ratio*100:.1f}%")

print("\nNOTE: Baseline accuracy requires ground truth transcriptions.")
print("For proper validation, provide human-corrected speaker labels per call.")
print("\nNext Steps:")
print("1. Re-enroll agents using independent call data (with transcriptions)")
print("2. Test accuracy comparison before/after re-enrollment")
print("3. Deploy ECAPA fusion + temporal voting for additional gains")

print("\n" + "=" * 100)
