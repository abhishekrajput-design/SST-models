#!/usr/bin/env python3
"""
Test real API calls - compare our identification against API ground truth transcription.
"""
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import random

from enroll_all_from_api import slug, load_mp3_mono_16k, ts2s
from src.embedding_campp import EmbeddingModel
from src.voiceprints import resolve_voiceprint_path
from enroll_multi_from_api import estimate_snr_db

api_file = "data/audiofy/_dataset/index.json"
audio_dir = Path("data/audiofy/_dataset/audio")

with open(api_file) as f:
    api_data = json.load(f)

with open("data/agent_voiceprints/agents.json") as f:
    agents_data = json.load(f)

def load_voiceprint_stacks(target_dim=512):
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
            r = resolve_voiceprint_path(raw, "data/agent_voiceprints/agents.json")
            if not r or not Path(r).is_file():
                continue
            try:
                vp = np.load(r).astype(np.float32).squeeze()
            except:
                continue
            if vp.ndim != 1:
                continue
            if vp.shape[0] != target_dim:
                continue
            n = np.linalg.norm(vp)
            if n > 0:
                vp = vp / n
            loaded.append(vp)

        if loaded:
            stacked = np.array(loaded, dtype=np.float32)
            out[slg] = (info.get("agent_name", slg), stacked)

    return out

print("[TEST] Loading embedding model...")
model = EmbeddingModel()
try:
    model.load(force_cpu=False)
except:
    model.load(force_cpu=True)

print("[TEST] Model ready (dim={})".format(model.dim))

vps = load_voiceprint_stacks(target_dim=model.dim)
print("[TEST] Loaded {} agents".format(len(vps)))

print()
print("=" * 100)
print("REAL API CALL ACCURACY TEST - Comparing Against API Ground Truth")
print("=" * 100)
print()

random.seed(42)

testable_calls = []
for call in api_data:
    agent_name = call.get('agent_name', '')
    agent_slug = slug(agent_name)
    if agent_slug in vps and call.get('speaker_json'):
        testable_calls.append(call)

selected = random.sample(testable_calls, min(10, len(testable_calls)))

results = []
THRESHOLD = 0.35

for idx, call in enumerate(selected, 1):
    call_id = call.get('_id')
    agent_name = call.get('agent_name')
    agent_slug = slug(agent_name)

    audio_file = audio_dir / "{}.mp3".format(call_id)
    if not audio_file.exists():
        continue

    print("[{:2d}] {:30s} (API: {:8s}) ".format(idx, agent_name, call_id[:8]), end="", flush=True)

    try:
        audio, sr = load_mp3_mono_16k(audio_file)

        api_agent_phrases = 0
        api_customer_phrases = 0

        for phrase in call.get('speaker_json', []):
            speaker = (phrase.get('speaker') or '').strip().lower()
            if speaker != 'customer' and speaker:
                api_agent_phrases += 1
            else:
                api_customer_phrases += 1

        tp = fp = tn = fn = 0
        agent_sims = defaultdict(list)
        slugs = list(vps.keys())
        stacks = [vps[s][1] for s in slugs]

        our_agent_count = 0
        our_customer_count = 0

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

            sims = np.array([float(np.max(stacks[j] @ emb)) for j in range(len(slugs))], dtype=np.float32)
            best_j = int(np.argmax(sims))
            best_sim = float(sims[best_j])
            pred_slug = slugs[best_j]

            is_agent_pred = best_sim >= THRESHOLD
            speaker = (phrase.get('speaker') or '').strip()
            is_agent_truth = bool(speaker) and speaker.lower() != 'customer'

            if is_agent_pred:
                our_agent_count += 1
            else:
                our_customer_count += 1

            if is_agent_pred and is_agent_truth: tp += 1
            elif is_agent_pred and not is_agent_truth: fp += 1
            elif not is_agent_pred and is_agent_truth: fn += 1
            else: tn += 1

            agent_sims[pred_slug].append(best_sim)

        if agent_sims:
            call_agent_slug = max(agent_sims, key=lambda k: sum(agent_sims[k]) / len(agent_sims[k]))
            correct = (call_agent_slug == agent_slug)

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)

            avg_sim = sum(agent_sims[agent_slug]) / len(agent_sims[agent_slug]) if agent_slug in agent_sims else 0

            snr_db = estimate_snr_db(audio, sr)

            status = "OK" if correct else "WRONG"
            print("{} | F1={:.3f} | Sim={:.3f} | API:(A={},C={}) Ours:(A={},C={}) | SNR={:.1f}dB".format(
                status, f1, avg_sim, api_agent_phrases, api_customer_phrases,
                our_agent_count, our_customer_count, snr_db))

            results.append({
                'agent': agent_name,
                'correct': correct,
                'f1': f1,
                'similarity': avg_sim,
                'api_agent': api_agent_phrases,
                'api_customer': api_customer_phrases,
                'our_agent': our_agent_count,
                'our_customer': our_customer_count,
                'snr_db': snr_db
            })
        else:
            print("SKIP - no segments")

    except Exception as e:
        print("ERROR: {}".format(str(e)[:60]))

model.unload()

print()
print("=" * 100)
print("ACCURACY SUMMARY - Real API Calls vs Ground Truth")
print("=" * 100)

if results:
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_sim = sum(r['similarity'] for r in results) / len(results)

    print("\nCalls tested:           {}".format(total))
    print("Correctly identified:   {}/{} ({:.1f}%)".format(correct, total, 100*correct/total))
    print("Avg F1 (segment level): {:.3f}".format(avg_f1))
    print("Avg similarity score:   {:.3f}".format(avg_sim))
    print()

    print("Per-call breakdown:")
    print("-" * 100)
    for r in results:
        status = "[OK]" if r['correct'] else "[WRONG]"
        print("  {} {:30s} | F1={:.3f} | Sim={:.3f} | API:(A={:2d},C={:2d}) Ours:(A={:2d},C={:2d}) | SNR={:5.1f}dB".format(
            status, r['agent'], r['f1'], r['similarity'],
            r['api_agent'], r['api_customer'], r['our_agent'], r['our_customer'], r['snr_db']))
