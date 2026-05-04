#!/usr/bin/env python3
"""
Single upload test - check multi-voiceprint matching via UI.
"""
import json
import time
import os
import subprocess
import sys
import requests
from pathlib import Path

UI_URL = "http://localhost:8080"
API_DATA_DIR = "data/audiofy/_dataset"

def start_ui():
    """Start UI server."""
    print("[*] Starting UI server...")
    proc = subprocess.Popen(
        [sys.executable, "ui.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    time.sleep(5)  # Wait for startup
    return proc

def upload_call(call_id, agent_name, audio_path):
    """Upload single call and measure time."""
    print(f"[*] Uploading {call_id[:8]} ({agent_name})...")

    t0 = time.time()
    with open(audio_path, "rb") as f:
        resp = requests.post(f"{UI_URL}/api/upload", files={"file": f}, timeout=600)

    if resp.status_code != 200:
        print(f"[!] Upload failed: {resp.status_code}")
        return None

    upload_time = time.time() - t0
    data = resp.json()
    result_id = data.get("result_id")

    print(f"[+] Upload done in {upload_time:.1f}s, result_id={result_id}")
    return result_id

def poll_result(result_id, max_wait=600):
    """Poll for result."""
    print(f"[*] Polling for result...")
    start = time.time()

    while time.time() - start < max_wait:
        resp = requests.get(f"{UI_URL}/api/call/{result_id}")
        if resp.status_code == 200:
            elapsed = time.time() - start
            print(f"[+] Result ready in {elapsed:.1f}s")
            return resp.json(), elapsed

        time.sleep(3)
        pct = (time.time() - start) / max_wait * 100
        print(f"  ... {pct:.0f}% ({time.time() - start:.1f}s)")

    return None, None

def main():
    print("[test] Single upload UI accuracy test\n")

    # Start UI
    ui_proc = start_ui()

    try:
        # Load API data
        with open(f"{API_DATA_DIR}/index.json") as f:
            api_data = json.load(f)

        # Pick a known good call - Omar El Harchaoui with high-SNR call
        call = next((c for c in api_data if c.get("_id") == "69efb80e"), None)

        if not call:
            print("[!] Test call not found")
            return

        call_id = call.get("_id")
        agent_name = call.get("agent_name")
        audio_path = f"{API_DATA_DIR}/audio/{call_id}.mp3"

        if not os.path.exists(audio_path):
            print(f"[!] Audio file not found: {audio_path}")
            return

        print(f"Test call: {call_id}")
        print(f"Agent: {agent_name}")
        print(f"Audio: {os.path.getsize(audio_path) / 1024 / 1024:.1f} MB\n")

        # Upload
        result_id = upload_call(call_id, agent_name, audio_path)
        if not result_id:
            return

        # Wait for result
        result, wait_time = poll_result(result_id)
        if not result:
            print("[!] Timeout waiting for result")
            return

        # Total time
        total_time = wait_time
        print(f"\n[RESULTS]")
        print(f"  Total processing time: {total_time:.1f}s")
        print(f"  Identified agent: {result.get('identified_agent')}")
        print(f"  Speaker ID backend dim: {result.get('speaker_id_backend_dim')}")
        print(f"  Voiceprint dims: {result.get('voiceprint_dims')}")
        print(f"  Total segments: {len(result.get('segments', []))}")

        # Extract segment labels
        segments = result.get("segments", [])
        agent_segs = sum(1 for s in segments if s.get("speaker") == "AGENT")
        cust_segs = sum(1 for s in segments if s.get("speaker") == "CUSTOMER")

        print(f"  Agent segments: {agent_segs}")
        print(f"  Customer segments: {cust_segs}")

        # Check accuracy
        identified = (result.get("identified_agent") or "").lower().replace(" ", "_")
        expected = (agent_name or "").lower().replace(" ", "_")

        if identified == expected or expected in identified:
            print(f"\n[OK] Correctly identified as {identified}")
        else:
            print(f"\n[FAIL] Expected {expected}, got {identified}")

        # Save result
        with open("single_upload_result.json", "w") as f:
            json.dump({
                "call_id": call_id,
                "agent_name": agent_name,
                "total_time_s": total_time,
                "identified_agent": result.get("identified_agent"),
                "correct": identified == expected or expected in identified,
                "result": result
            }, f, indent=2)

        print(f"\nFull result saved to single_upload_result.json")

    finally:
        # Kill UI
        print("\n[*] Stopping UI...")
        ui_proc.terminate()
        ui_proc.wait(timeout=5)

if __name__ == "__main__":
    main()
