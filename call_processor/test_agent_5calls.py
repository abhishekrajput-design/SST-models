#!/usr/bin/env python3
"""
Test 5 calls with agent speech - compare transcription and diarization vs API.
"""
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import random

from enroll_all_from_api import slug, load_mp3_mono_16k, ts2s
from src.embedding_campp import EmbeddingModel

# Load API dataset
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

print("[TEST] Loading models...")
model = EmbeddingModel()
try:
    model.load(force_cpu=False)
except:
    model.load(force_cpu=True)

vps = load_voiceprints(target_dim=model.dim)
print(f"[TEST] Loaded {len(vps)} agents, embedding dim={model.dim}")

# Find calls with agent speech
calls_with_agent = []
for call in api_data:
    agent_name = call.get('agent_name', '')
    agent_slug = slug(agent_name)
    speaker_json = call.get('speaker_json', [])
    
    has_agent = any((s.get('speaker') or '').lower().strip() not in ('customer', '')
                    for s in speaker_json)
    
    if agent_slug in vps and has_agent and speaker_json:
        calls_with_agent.append(call)

print(f"[TEST] Found {len(calls_with_agent)} calls with agent speech")

# Select 5 random calls
random.seed(42)
selected = random.sample(calls_with_agent, min(5, len(calls_with_agent)))

print("\n" + "="*100)
print("AGENT IDENTIFICATION & TRANSCRIPTION TEST - 5 Calls with Agent Speech")
print("="*100 + "\n")

results = []
THRESHOLD = 0.35

for idx, call in enumerate(selected, 1):
    call_id = call.get('_id')
    agent_name = call.get('agent_name')
    agent_slug = slug(agent_name)
    
    audio_file = audio_dir / f"{call_id}.mp3"
    if not audio_file.exists():
        continue
    
    print(f"[{idx}] {agent_name:30s} (API: {call_id[:8]}) ", end="", flush=True)
    
    try:
        audio, sr = load_mp3_mono_16k(audio_file)
        
        # Count API agent vs customer
        api_agent_phrases = sum(1 for p in call.get('speaker_json', [])
                                if (p.get('speaker') or '').lower().strip() not in ('customer', ''))
        api_customer_phrases = sum(1 for p in call.get('speaker_json', [])
                                   if (p.get('speaker') or '').lower().strip() in ('customer', ''))
        
        # Diarization
        tp = fp = tn = fn = 0
        our_agent_count = 0
        
        for phrase in call.get('speaker_json', []):
            s = ts2s(phrase.get('start'))
            e = ts2s(phrase.get('end'))
            if e - s < 0.4:
                continue
            
            si = max(0, int(s * sr))
            ei = min(int(e * sr), len(audio))
            if ei - si < int(sr * 0.4):
                continue
            
            chunk = audio[si:ei]
            emb = model.embed_chunk(chunk, sr)
            if emb is None:
                continue
            
            n = np.linalg.norm(emb)
            if n == 0:
                continue
            emb = emb / n
            
            slugs = list(vps.keys())
            stacks = [vps[s][1] for s in slugs]
            sims = np.array([float(np.max(stacks[j] @ emb)) for j in range(len(slugs))], dtype=np.float32)
            best_sim = float(np.max(sims))
            
            is_agent_pred = best_sim >= THRESHOLD
            speaker = (phrase.get('speaker') or '').strip()
            is_agent_truth = bool(speaker) and speaker.lower() != 'customer'
            
            if is_agent_pred:
                our_agent_count += 1
            
            if is_agent_pred and is_agent_truth: tp += 1
            elif is_agent_pred and not is_agent_truth: fp += 1
            elif not is_agent_pred and is_agent_truth: fn += 1
            else: tn += 1
        
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        
        print(f"F1={f1:.3f} | API:(A={api_agent_phrases},C={api_customer_phrases}) Ours:A={our_agent_count}")
        results.append({
            'agent': agent_name,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'api_agent': api_agent_phrases,
            'our_agent': our_agent_count
        })
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*100)
print("SUMMARY - Agent Identification on 5 Agent Calls")
print("="*100)
print(f"Calls tested: {len(results)}")
if results:
    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"Average F1: {avg_f1:.3f}")
    print(f"\nDetailed Results:")
    for r in results:
        print(f"  {r['agent']:30s} | F1={r['f1']:.3f} | API Agent phrases={r['api_agent']:3d} | Our detection={r['our_agent']:3d}")
