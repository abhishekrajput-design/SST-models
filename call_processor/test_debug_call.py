#!/usr/bin/env python3
"""
Debug the specific call that had poor identification.
"""
import json
import time
import requests
import sys

UI_URL = "http://localhost:8080"
AUDIO_FILE = "data/raw_calls/enhanced_20260503T131905453_618398.mp3"

def upload_and_check():
    print("[DEBUG] Uploading audio file for identification...\n")

    # Upload
    t0 = time.time()
    with open(AUDIO_FILE, "rb") as f:
        resp = requests.post(f"{UI_URL}/api/upload", files={"file": f}, timeout=900)

    if resp.status_code != 200:
        print(f"[!] Upload failed: HTTP {resp.status_code}")
        return

    try:
        data = resp.json()
    except:
        print(f"[!] Invalid JSON response")
        return

    result_id = data.get("result_id")
    if not result_id:
        print(f"[!] No result_id returned")
        return

    print(f"[+] Upload successful, result_id={result_id}")
    print(f"[*] Waiting for processing... (max 10 minutes)")

    # Poll for result
    max_wait = 600
    start = time.time()
    while time.time() - start < max_wait:
        resp2 = requests.get(f"{UI_URL}/api/call/{result_id}")
        if resp2.status_code == 200:
            elapsed_total = time.time() - t0

            try:
                result = resp2.json()
            except:
                print(f"[!] Invalid result JSON")
                return

            # Display results
            print(f"\n[RESULT] Processing time: {elapsed_total:.1f}s\n")

            print(f"Identified Agent: {result.get('identified_agent')}")
            print(f"Agent Similarity: {result.get('agent_similarity')}")
            print(f"Speaker ID Mode: {result.get('speaker_id_mode')}")
            print(f"Backend Dimension: {result.get('speaker_id_backend_dim')}")
            print(f"Voiceprint Dims: {result.get('voiceprint_dims')}")

            if result.get('speaker_id_warning'):
                print(f"⚠ Warning: {result.get('speaker_id_warning')}")

            # Segment analysis
            segments = result.get("segments", [])
            print(f"\nSegment Analysis:")
            print(f"  Total segments: {len(segments)}")

            if segments:
                agent_segs = sum(1 for s in segments if s.get("identified_speaker") == "AGENT")
                cust_segs = sum(1 for s in segments if s.get("identified_speaker") == "CUSTOMER")
                print(f"  AGENT: {agent_segs} segments")
                print(f"  CUSTOMER: {cust_segs} segments")

                # Check similarity scores
                sims = [s.get("_best_sim", 0) for s in segments if "_best_sim" in s]
                if sims:
                    print(f"  Similarity scores:")
                    print(f"    Min: {min(sims):.4f}")
                    print(f"    Avg: {sum(sims)/len(sims):.4f}")
                    print(f"    Max: {max(sims):.4f}")

                # Show first few segments with low confidence
                print(f"\n  Low-confidence segments:")
                low_conf = [s for s in segments if s.get("_best_sim", 1) < 0.4]
                for i, seg in enumerate(low_conf[:5]):
                    text = seg.get("text", "")[:50]
                    sim = seg.get("_best_sim", 0)
                    speaker = seg.get("identified_speaker", "?")
                    print(f"    [{i+1}] {speaker:8s} sim={sim:.3f} → \"{text}...\"")

            # Save for comparison
            with open("debug_result.json", "w") as f:
                json.dump(result, f, indent=2)

            print(f"\nFull result saved to debug_result.json")
            return

        time.sleep(3)

    print(f"[!] Timeout after {max_wait}s")

if __name__ == "__main__":
    upload_and_check()
