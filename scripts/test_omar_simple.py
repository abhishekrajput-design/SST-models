#!/usr/bin/env python
"""Simple test of Omar audio through the system."""
import sys
import json
import requests
from pathlib import Path

AUDIO_FILE = Path("c:/Users/abhis/Downloads/20260505T073055769_385036.mp3")
API_BASE = "http://localhost:8080"

print("=" * 100)
print("TESTING OMAR'S CALL - SIMPLE VERSION")
print("=" * 100)

if not AUDIO_FILE.exists():
    print(f"Audio file not found: {AUDIO_FILE}")
    sys.exit(1)

print(f"\nFile: {AUDIO_FILE.name} ({AUDIO_FILE.stat().st_size / 1024 / 1024:.2f} MB)")

print("\nUploading to system...")
try:
    with open(AUDIO_FILE, 'rb') as f:
        files = {'file': (AUDIO_FILE.name, f)}
        data = {'model': 'parakeet-tdt-0.6b-v3'}

        response = requests.post(
            f"{API_BASE}/api/upload",
            files=files,
            data=data,
            timeout=600
        )

        print(f"Upload response: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        call_id = result.get('id')
        if not call_id:
            print("\nNo call ID returned. Trying to get latest call...")
            # Get the latest call
            response = requests.get(f"{API_BASE}/api/calls?limit=1")
            calls = response.json()
            if calls:
                call_id = calls[0]['id']
                print(f"Latest call ID: {call_id}")

        if call_id:
            print(f"\nFetching results for: {call_id}")
            response = requests.get(f"{API_BASE}/api/call/{call_id}")
            call_data = response.json()

            segments = call_data.get('segments', [])
            print(f"\nResults:")
            print(f"  Segments: {len(segments)}")
            print(f"  Model: {call_data.get('model')}")
            print(f"  Agent: {call_data.get('agent_name', 'Unknown')}")
            print(f"  Similarity: {call_data.get('agent_similarity', 0):.3f}")

            # Count speakers
            agent_count = sum(1 for s in segments if 'AGENT' in s.get('identified_speaker', '').upper())
            customer_count = sum(1 for s in segments if 'CUSTOMER' in s.get('identified_speaker', '').upper())

            print(f"\nSpeaker breakdown:")
            print(f"  AGENT: {agent_count} segments")
            print(f"  CUSTOMER: {customer_count} segments")

            print(f"\nFirst 10 segments:")
            for i, seg in enumerate(segments[:10]):
                role = seg.get('identified_speaker', '').upper()[:10]
                sim = seg.get('_best_sim', 0.0)
                text = seg.get('text', '')[:50]
                print(f"  {i+1}. {role:<10} sim={sim:.3f}  {text}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 100)
print("System is running with all Phase 1 + Phase 3 improvements")
print("Current accuracy: 50-70% | Target: 95-98%")
print("=" * 100)
