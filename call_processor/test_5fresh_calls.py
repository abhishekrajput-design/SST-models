#!/usr/bin/env python3
"""
Process 5 fresh audio calls with Parakeet, test agent ID and transcription vs API.
"""
import json
import os
from pathlib import Path
import numpy as np
import random

from enroll_all_from_api import slug, load_mp3_mono_16k, ts2s
from src.embedding_campp import EmbeddingModel
from src.transcribers import get_transcriber
from src.diar_multi import diarize_multi

api_file = "data/audiofy/_dataset/index.json"
audio_dir = Path("data/audiofy/_dataset/audio")

with open(api_file) as f:
    api_data = json.load(f)

with open("data/agent_voiceprints/agents.json") as f:
    agents_data = json.load(f)

def load_voiceprints(target_dim=512):
    out = {}
    for slg, info in agents_data.items():
        if not isinstance(info, dict):
            continue
        paths = []
        if isinstance(info.get("voiceprints"), list):
            for entry in info["voiceprints"]:
                p = entry.get("path") if isinstance(entry, dict) else entry
                if p:
                    paths.append(p)
        if not paths:
            legacy = info.get("voiceprint_path") or info.get("voiceprint")
            if legacy:
                paths.append(legacy)

        loaded = []
        for raw in paths:
            try:
                vp = np.load(raw).astype(np.float32).squeeze()
            except:
                continue
            if vp.ndim != 1 or vp.shape[0] != target_dim:
                continue
            n = np.linalg.norm(vp)
            if n > 0:
                vp = vp / n
            loaded.append(vp)

        if loaded:
            stacked = np.array(loaded, dtype=np.float32)
            out[slg] = (info.get("agent_name", slg), stacked)

    return out

# Load models
print("[SETUP] Loading embedding model...")
model = EmbeddingModel()
try:
    model.load(force_cpu=False)
except:
    model.load(force_cpu=True)

print("[SETUP] Loading transcriber (Parakeet)...")
transcriber = get_transcriber('parakeet-tdt-0.6b-v3')

vps = load_voiceprints(target_dim=model.dim)
print(f"[SETUP] Loaded {len(vps)} agents\n")

# Select 5 calls (3-4 minutes)
test_calls = []
for call in api_data:
    call_id = call.get('_id')
    audio_file = audio_dir / f"{call_id}.mp3"
    if not audio_file.exists():
        continue
    
    speaker_json = call.get('speaker_json', [])
    if not speaker_json:
        continue
    
    max_time = max((ts2s(s.get('end', '0:0:0')) for s in speaker_json), default=0)
    
    if 180 <= max_time <= 240:
        has_agent = any((s.get('speaker') or '').lower().strip() not in ('customer', '')
                        for s in speaker_json)
        agent_slug = slug(call.get('agent_name', ''))
        if agent_slug in vps and has_agent:
            test_calls.append(call)

random.seed(42)
test_calls = random.sample(test_calls, min(5, len(test_calls)))

print(f"Testing {len(test_calls)} calls\n")
print("="*100)
print("FRESH CALL TEST - Parakeet Transcription + Agent ID vs API")
print("="*100 + "\n")

results = []
for idx, call in enumerate(test_calls, 1):
    call_id = call.get('_id')
    agent_name = call.get('agent_name')
    agent_slug = slug(agent_name)
    
    audio_file = audio_dir / f"{call_id}.mp3"
    print(f"[{idx}] {agent_name:30s} ({call_id[:8]}) ", end="", flush=True)
    
    try:
        # Load audio
        audio, sr = load_mp3_mono_16k(audio_file)
        
        # Transcribe with Parakeet
        segments = transcriber.transcribe(audio)
        
        # Diarize
        result = diarize_multi(segments, str(audio_file))
        
        # Compare with API
        api_agent_count = sum(1 for p in call.get('speaker_json', [])
                              if (p.get('speaker') or '').lower().strip() not in ('customer', ''))
        api_customer_count = sum(1 for p in call.get('speaker_json', [])
                                 if (p.get('speaker') or '').lower().strip() in ('customer', ''))
        
        our_agent = sum(1 for s in result.get('segments', [])
                       if s.get('identified_speaker') == 'AGENT')
        our_customer = sum(1 for s in result.get('segments', [])
                          if s.get('identified_speaker') == 'CUSTOMER')
        
        # Calculate accuracy
        agent_match = our_agent >= (api_agent_count * 0.7)
        customer_match = our_customer >= (api_customer_count * 0.7)
        
        accuracy = 'OK' if (agent_match and customer_match) else 'MISS'
        
        print(f"{accuracy} | API:(A={api_agent_count},C={api_customer_count}) | Ours:(A={our_agent},C={our_customer})")
        
        results.append({'agent': agent_name, 'accuracy': accuracy})
        
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")

print("\n" + "="*100)
print("SUMMARY - 5 Fresh Calls with Parakeet + Agent ID")
print("="*100)
if results:
    correct = sum(1 for r in results if r['accuracy'] == 'OK')
    print(f"Calls tested: {len(results)}")
    print(f"Correct identification: {correct}/{len(results)} ({100*correct//len(results)}%)")
