"""
Compare Omar's multi-VP accuracy vs the misidentified upload.
Test Omar on his enrolled calls and on the uploaded call.
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from enroll_all_from_api import load_mp3_mono_16k, AUDIO_DIR, TARGET_SR, ts2s
from test_voiceprints_api import load_voiceprint_stacks
from src.embedding_campp import EmbeddingModel

AGENTS_JSON = Path(__file__).parent / "data/agent_voiceprints/agents.json"
INDEX_JSON = Path(__file__).parent / "data/audiofy/_dataset/index.json"

model = EmbeddingModel()
try:
    model.load(force_cpu=False)
except:
    model.load(force_cpu=True)
print(f"[test] {model.model_name} ready (dim={model.dim})")

vps = load_voiceprint_stacks(multi=True, target_dim=model.dim)
if "omar_el_harchaoui" not in vps:
    print("[ERROR] Omar not in voiceprints")
    sys.exit(1)

omar_name, omar_stack = vps["omar_el_harchaoui"]
print(f"\nOmar voiceprints: {omar_stack.shape} = {omar_stack.shape[0]} centroids")

# Test on Mohamed Yasin-ali's call (the one that matched the uploaded audio)
with open(INDEX_JSON, encoding='utf-8') as f:
    index = json.load(f)
moha_rec = None
for rec in index:
    if rec.get("_id") == "69efa352f91ac02559f7e936":
        moha_rec = rec
        break

if moha_rec:
    rid = moha_rec["_id"]
    mp3 = AUDIO_DIR / f"{rid}.mp3"
    if mp3.exists():
        print(f"\n=== Testing on Mohamed's call {rid[:8]} ===")
        try:
            audio, sr = load_mp3_mono_16k(mp3)
            print(f"  Loaded {len(audio)/sr:.1f}s audio")

            best_sims = []
            for phrase in moha_rec.get("speaker_json", []):
                s = ts2s(phrase.get("start"))
                e = ts2s(phrase.get("end"))
                if e - s < 0.4:
                    continue
                si = max(0, int(s * sr))
                ei = min(int(e * sr), len(audio))
                chunk = audio[si:ei]
                emb = model.embed_chunk(chunk, sr)
                if emb is None:
                    continue
                n = np.linalg.norm(emb)
                if n == 0:
                    continue
                emb = emb / n
                best_sim = float(np.max(omar_stack @ emb))
                best_sims.append(best_sim)
                speaker = phrase.get("speaker", "")
                print(f"    {s:5.1f}–{e:5.1f}s [{speaker:>8}] Omar cosine = {best_sim:.3f}")

            if best_sims:
                print(f"  Average Omar cosine: {np.mean(best_sims):.3f}")
                print(f"  Max Omar cosine: {np.max(best_sims):.3f}")
        except Exception as e:
            print(f"  Error: {e}")

model.unload()
