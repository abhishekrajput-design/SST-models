# Agent Voiceprints — Handover Doc

Last verified: 2026-05-20 against working copy on Abhishek's machine.

This doc tells you exactly **which files** make up the enrolled agent voiceprints,
**what format** they are in, and **how the pipeline loads** them — so you can
copy them to another machine and they will just work.

---

## 1. TL;DR — What to copy

To move the voiceprints to another machine, copy these two things:

```
call_processor/data/agent_voiceprints/agents.json     # index, ~60 KB
call_processor/data/agent_voiceprints/*.npy           # 219 files, ~876 KB total
```

Total payload ≈ **940 KB**. Drop them into the same relative path
(`call_processor/data/agent_voiceprints/`) on the target machine and the
pipeline picks them up automatically.

**Do NOT copy these subfolders** — they are local-only state:

```
call_processor/data/agent_voiceprints/_candidates/           # in-progress experiments
call_processor/data/agent_voiceprints/backup_20260504T151730/ # old backup
call_processor/data/agent_voiceprints/daily_reports/         # local training reports
call_processor/data/agent_voiceprints/agents.backup.*.json   # ~30 backup JSONs
call_processor/data/agent_voiceprints/agents.enhanced_backup.*.json
```

---

## 2. File layout

```
call_processor/
└── data/
    └── agent_voiceprints/
        ├── agents.json              <-- THE INDEX (single source of truth)
        ├── <slug>.npy               <-- legacy single voiceprint (Tier 3)
        ├── <slug>__<bucket>_<n>.npy <-- multi-VP, SNR-bucketed (Tier 2)
        └── <slug>_pure_campp_v<n>.npy / <slug>_clean_campp_<n>.npy
                                     <-- multi-VP, CAM++ (Tier 1)
```

- **`agents.json`** maps `slug → {agent_name, voiceprint paths, metadata}`. The
  pipeline reads this first; .npy files are only loaded if referenced here.
- **`.npy` files** are single 1-D `float32` numpy arrays, **L2-normalised**.
  Each file is one centroid (i.e. one voiceprint).
- Two embedding dims exist in the codebase:
  - **192-dim** → ECAPA-TDNN (SpeechBrain `spkrec-ecapa-voxceleb`)
  - **512-dim** → CAM++ (wespeaker)
  Within a single agent's centroid list, mismatched dims are dropped — dominant
  dim wins. See `_load_voiceprints` in `call_processor/src/diar_multi.py:160`.

---

## 3. `agents.json` schema — two shapes

### Shape A — legacy single voiceprint (Tier 3)

```json
"sarah_aziz": {
  "agent_name": "Sarah Aziz",
  "voiceprint_path": "C:\\...\\sarah_aziz.npy",
  "n_clips": 74,
  "total_seconds": 225.6,
  "used_calls": 4,
  "mean_inside_sim": 0.657,
  "max_outside_sim": 0.625,
  "source": "audiofy_api_bulk_tight_20260424"
}
```

### Shape B — multi-voiceprint (Tier 1 + Tier 2)

```json
"omar_el_harchaoui": {
  "agent_name": "Omar El Harchaoui",
  "voiceprint_path": "omar_el_harchaoui_pure_campp_v1.npy",   // legacy fallback
  "voiceprints": [
    { "path": "omar_el_harchaoui_pure_campp_v1.npy",
      "source": "pure_api_agent_segments",
      "embedding_model": "cam++", "embedding_dim": 512 },
    { "path": "omar_el_harchaoui_pure_campp_v2.npy", ... },
    { "path": "omar_el_harchaoui_pure_campp_v3.npy", ... }
  ],
  "n_voiceprints": 3,
  "embedding_model": "cam++",
  "embedding_dim": 512,
  "mean_inside_sim": 0.756,
  "max_outside_sim": 0.4569
}
```

### Path resolution — important

Some legacy entries store an **absolute Windows path**
(`C:\Users\abhis\Desktop\SST-models\...`). On a different machine these will
not exist. The loader (`call_processor/src/voiceprints.py:24` —
`resolve_voiceprint_path`) handles this: if the absolute path is missing it
falls back to the file's **basename** beside `agents.json`. So as long as you
copy the .npy files into the same directory as `agents.json`, paths resolve
correctly — no need to rewrite the JSON.

---

## 4. Enrollment tiers — quality summary

There are **52 agents** in `agents.json` as of 2026-05-20, across three tiers:

### Tier 1 — Production-grade, CAM++ 512-dim, multi-VP (6 agents)

| Slug | Display name | # Centroids | Source |
|---|---|---:|---|
| `omar_el_harchaoui`   | Omar El Harchaoui   | 3  | `pure_api_agent_segments` (mean_inside=0.756, max_outside=0.457) |
| `amandeep_nandra`     | Amandeep Nandra     | 3  | pure CAM++ |
| `zak_raissi_barnet`   | Zak Raissi Barnet   | 12 | desk CAM++ |
| `hussein_mohamed`     | Hussein Mohamed     | 14 | pure + desk CAM++ |
| `aayush`              | Aayush              | 5  | `clean_single_speaker_recording` |
| `anil`                | Anil                | 4  | CAM++ |

### Tier 2 — Multi-VP ECAPA, SNR-bucketed (21 agents)

Format: `<slug>__<bucket>_<n>.npy` where bucket ∈ {low, mid, high} = SNR bucket.

`haris_bajwa`, `allan_johnson`, `talha_azam`, `mohammad_malki`,
`sylwia_recruitment`, `ideal_dacaj`, `harrison_morgan`, `adil_al_sammerai`,
`angeline_packiyaseelan`, `jason_kurti`, `anoush_sefatzadeh`, `kowsar_alam`,
`georgi_angelov`, `janusaan_jeyachandran`, `aftaab_supervisor`,
`mohamed_yasin_ali`, `rayyan_ali_khan`, `adorena_ishtar_hossain`,
`waris_sales_controllers`, `kacper_barnet`, `dinosh_sinnathamby`

### Tier 3 — Legacy single-vector ECAPA (25 agents)

One `.npy` per agent. Lowest quality tier — kept for coverage. Tagged
`source: audiofy_api_bulk_tight_20260424` or similar.

`zak_local_20260423` (Zak Raissi), `mohammed_al_russell`, `sarah_aziz`,
`rebeca_cazan`, `rajan_singh`, `rebecca_murphy`, `nevethan_krishnamohan`,
`shuahib_miah`, `albjon_vokshi`, `qaim_ravji`, `tulay_finance_consultant`,
`mashrur_rahman`, `mohammed_malik`, `mohammed_hussein_al_khwildi`,
`sababa_hossain`, `liza_mae_esguerra`, `benjamin_ahmadi`, `niloufar_dastbaz`,
`rafik_saleh`, `gabriel_bighiu`, `nirvan_nagra`, `arfat_barnet`,
`dilayda_barnet`, `kleo_gurra`, `jenifer_bajrami`

---

## 5. How the pipeline consumes these files

Two callers load voiceprints — both go through `agents.json`:

1. **`call_processor/src/diar_multi.py`** → `_load_voiceprints()` (line 160).
   Returns `{slug: (display_name, stack)}` where `stack` is shape `(N, dim)`
   of L2-normalised centroids. Used during mono diarisation for cosine matching.
2. **`call_processor/src/speaker_role.py`** → `_match_enrolled()`. Loads the
   correct embedding model (ECAPA for 192-dim, CAM++ for 512-dim) based on
   the voiceprint's shape.

**Minimal loading snippet** (copy-pasteable, no project code needed):

```python
import json, numpy as np
from pathlib import Path

INDEX = Path("call_processor/data/agent_voiceprints/agents.json")
DIR   = INDEX.parent

with INDEX.open() as f:
    agents = json.load(f)

voiceprints = {}  # slug -> (name, ndarray(N, dim))
for slug, info in agents.items():
    paths = [vp["path"] for vp in info.get("voiceprints", [])]
    if not paths and info.get("voiceprint_path"):
        paths = [info["voiceprint_path"]]
    vecs = []
    for p in paths:
        # try absolute first, fall back to basename beside agents.json
        candidate = Path(p)
        if not candidate.is_file():
            candidate = DIR / Path(p).name
        if not candidate.is_file():
            continue
        v = np.load(candidate).astype(np.float32).squeeze()
        v = v / (np.linalg.norm(v) + 1e-9)
        vecs.append(v)
    if vecs:
        voiceprints[slug] = (info.get("agent_name", slug), np.stack(vecs))
```

After this, cosine similarity between a query embedding `q` (same dim as the
stack) and an agent's centroids is `voiceprints[slug][1] @ q`.

---

## 6. How to add a new agent (for reference)

Enrollment scripts live in `call_processor/scripts/`:

- `auto_enroll_agent.py` — **the main gated entry point**. Wraps everything with
  leave-one-call-out validation + activation gates (≥95% accuracy minimum).
- `train_omar_pure_embeddings.py` / `train_zak_pure_embeddings.py` —
  agent-specific CAM++ training with purity filtering (Tier 1 path).
- `enroll_multi_from_api.py` (in `call_processor/`) — multi-VP ECAPA with SNR
  bucketing (Tier 2 path).
- `enroll_all_from_api.py` (in `call_processor/`) — bulk ECAPA single-vector
  enrollment (Tier 3 path).

Source training audio: `traning_data/<agent>/call_NN/{audio.mp3, data.json}`
(note: directory name is "traning_data" with the typo).

---

## 7. Gotchas

- **Path style**: `agents.json` contains a mix of absolute Windows paths and
  bare filenames. The loader handles both. If you maintain `agents.json`
  manually, prefer bare filenames — they survive moves.
- **Backups will not be loaded**: the loader only reads `agents.json`. The
  `agents.backup.*.json` files are inert — safe to keep or delete.
- **Mismatched dims within one agent are silently dropped**: if you mix
  192-dim and 512-dim centroids under the same slug, the loader keeps only
  the dominant dim. Don't mix.
- **Live server path**: on the AWS box (`13.42.127.218:8080`) these files live
  at the same relative path. Linux ignores backslashes in absolute Windows
  paths, so the basename fallback in `resolve_voiceprint_path` is what makes
  it work cross-platform.
- **Some entries set `use_for_segment_role: true`** with thresholds
  (`segment_role_min_similarity`, `segment_role_min_margin`). These tune
  per-segment role decisions in `diar_multi.py`. Preserve them when copying
  `agents.json`.

---

## 8. Quick sanity check on the receiving machine

```bash
python -c "
import sys; sys.path.insert(0, 'call_processor')
from src.voiceprints import voiceprint_inventory
import json; print(json.dumps(voiceprint_inventory(), indent=2))
"
```

Expected output: `agent_count: 52`, `missing_count: 0`, `voiceprint_dims`
containing both `192` and `512` keys.
