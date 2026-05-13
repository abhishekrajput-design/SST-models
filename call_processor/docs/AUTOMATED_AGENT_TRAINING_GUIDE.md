# Automated Agent Training Pipeline Guide

**Last Updated:** May 2026

This document details the Automated Agent Training Pipeline that seamlessly collects, filters, and trains new agent voiceprints (using CAM++ embeddings) from production Audiofy API data. 

The goal of this system is to replace the manual agent enrollment process with a data-driven, self-improving pipeline that surfaces metrics and status directly via the unified UI Dashboard.

---

## 1. System Components

The automated training ecosystem consists of three primary modules:

1. **`daily_training_daemon.py` (The Orchestrator)**  
   A background daemon designed to be run periodically (e.g., via Task Scheduler or manually triggered via the UI). It connects to the Audiofy API to download recent call recordings and JSON labels, groups them by agent, and triggers the enrollment process.

2. **`train_agent_from_api_labels.py` (The Engine)**  
   A generalized voiceprint clustering engine. It extracts the raw CAM++ (512-dim) embeddings from the recordings, validates ground-truth data, performs K-Means clustering, and validates the resulting voiceprints using a rigorous Leave-One-Call-Out (LOCO) verification method.

3. **`ui.py` & `index.html` (The Dashboard)**  
   A front-end interface built directly into the existing UI. It interacts with background threads via new API routes to monitor agent performance stats and manually trigger training daemon runs.

---

## 2. Pipeline Execution Flow

### Phase 1: Data Scraping & Filtering
When `daily_training_daemon.py` is invoked, it queries the Audiofy API for calls placed within the target window (default: last 7 days). 
- **Purity Filters:** It parses the `speaker_json` and enforces strict criteria. A call is only eligible for training if the agent speaks enough valid, high-confidence, non-system agent segments.
- **Role Normalization:** Audiofy may label speakers as `Customer`, `Customer_1`, `Customer_2`, or `Agent Name_1`. The daemon maps every `Customer*` label to customer, maps only labels matching the requested agent name to agent, and skips unknown labels. This prevents customer speech from being enrolled as agent speech.
- **Data Staging:** Recordings and JSON labels are saved to the `traning_data/_daily_auto` directory.

### Phase 2: K-Means Clustering & Extraction
For each eligible agent, `train_agent_from_api_labels.py` loads their audio data and computes pure CAM++ embeddings for every labeled agent speech segment.
- **K-Means:** Because acoustic environments vary wildly (headset vs laptop mic, quiet vs noisy background), the system clusters the embeddings into `K=min(3, num_calls)`. 
- **Multi-Voiceprint Storage:** A single agent may have multiple "centroids" representing their different acoustic environments.

### Phase 3: LOCO Validation & Activation Gates
Before committing a new voiceprint to production, the system evaluates its quality.
- **LOCO Verification:** The system iterates over the calls, temporarily removing one call's data from the centroid pool, and then tests if the remaining centroids can correctly classify the held-out call.
- **Activation Gates:** A newly trained voiceprint will **only** overwrite the previous version in `agents.json` if all same-data role accuracies pass the configured threshold, LOCO validation passes that same threshold, `mean_inside_sim >= 0.60`, and `max_outside_sim <= 0.42`.

### Phase 4: UI Updates
The `agents.json` file is updated to include the new embedding vectors alongside metadata (`n_voiceprints`, `mean_inside_sim`, `total_training_seconds`, etc). The frontend UI polls these endpoints to dynamically update the agent statistics grid.

---

## 3. UI Dashboard Integration

You can monitor and trigger the automated training directly from the UI.

### New API Endpoints (`ui.py`)
- `GET /api/agents`: Returns a parsed list of all enrolled agents and their performance metrics.
- `GET /api/training-history`: (Future/WIP) Returns historical performance trends per agent.
- `POST /api/auto-train`: Spawns an isolated background thread that runs the `daily_training_daemon.py` subprocess.
- `GET /api/auto-train-status`: Polls the active daemon's `stdout` tail to display a live terminal feed in the browser.

### Using the Training Dashboard
1. Open the UI (`http://localhost:8080`).
2. Click the **Training** tab in the sidebar navigation.
3. Review the **Enrolled Agents** grid. Agents flagged in **Yellow/Amber** have low confidence metrics and may require more training audio. Agents flagged in **Green** are performing exceptionally.
4. Click **Run Daily Training** to manually invoke the scraping and training daemon. A terminal window will appear below the button, streaming the real-time progress of the API scraping, clustering, and voiceprint validation.

---

## 4. Configuration & Advanced Use

### Environment Variables
Ensure the following variables are present in your `.env` file for the scraper to authenticate:
```env
AUDIOFY_API_TOKEN=your_token_here
```

### Manual CLI Execution
If you prefer not to use the UI, you can invoke the daemon manually via CLI:
```bash
# Run for the last 7 days and activate the voiceprints in production
python call_processor/scripts/daily_training_daemon.py --days 7 --activate

# Run for a specific agent as a dry-run (won't save to agents.json)
python call_processor/scripts/daily_training_daemon.py --days 30 --agents "Zak Raissi Barnet" --dry-run

# Run a precise Audiofy dataset window for one agent
python call_processor/scripts/daily_training_daemon.py ^
  --start-time "2026-03-27T00:12:00.000Z" ^
  --end-time "2026-03-28T23:59:59.999Z" ^
  --user-name "Omar El Harchaoui" ^
  --agents "Omar El Harchaoui" ^
  --min-calls 1 --max-calls-total 10 --max-calls-per-agent 10 --dry-run
```

### Verified Local Zak Test - 2026-05-12
The safe local validation path is currently:
```bash
python call_processor/scripts/daily_training_daemon.py --skip-scrape --work-dir traning_data --agents Zak --min-calls 1 --max-calls-per-agent 2 --clusters 2 --dry-run
```

Latest verified result for `zak_raissi_barnet` from `traning_data/zak_raissi`:
- Calls loaded: 2 clean calls (`call_14`, `call_17`)
- Pure training rows: 20
- Same-data accuracy: 94.12% overall, 95.65% agent, 90.91% customer
- LOCO accuracy: 80.0%
- Mean inside similarity: 0.7662
- Max outside similarity: 0.2832
- Activation result: gated, not activated, because LOCO is below the 85% activation threshold

The report artifact is written to:
`call_processor/data/agent_voiceprints/daily_reports/zak_raissi_barnet.last_training_report.json`

### Verified Omar API Dataset Test - 2026-05-12
Using the exact dataset request:
```json
{
  "start_time": "2026-03-27T00:12:00.000Z",
  "end_time": "2026-03-28T23:59:59.999Z",
  "limit": 10,
  "user_name": "Omar El Harchaoui"
}
```

The API returned 10 records. After purity filtering, 9 calls were staged in:
`traning_data/_daily_auto/omar_el_harchaoui`

The filtered staging data contains:
- 351 agent-labelled segments
- 242 customer-labelled segments
- `Customer_1`, `Customer_2`, `Customer_3`, and `Customer_4` correctly mapped to customer
- The voicemail/system-only call excluded from training

Dry-run training result:
- Same-data accuracy at calibrated threshold: 76.82% overall, 64.43% agent, 98.81% customer
- Best same-data threshold: 88.63% overall, 93.62% agent, 79.76% customer
- LOCO accuracy: 75.32%
- Activation result: gated, not activated

Live UI upload check:
- Input: `traning_data/_daily_auto/omar_el_harchaoui/call_02/audio.mp3`
- Endpoint: `POST /api/upload?filename=omar_api_69c840836a2041f487a6ac20_call_02.mp3&model=parakeet-tdt-0.6b-v3&agent_slug=omar_el_harchaoui`
- Result id: `omar_api_69c840836a2041f487a6ac20_call_02__parakeet-tdt-0.6b-v3`
- Segment accuracy against API labels: 90.00% overall, 88.89% agent, 90.91% customer
- Verification artifact: `call_processor/data/agent_voiceprints/daily_reports/omar_call02_ui_verify_20260327_20260328.json`

### Current Live API Scrape Status
The UI route and daemon execution path are working. A fresh Audiofy login token is required at runtime; do not commit tokens into `.env` or source files. If scraping returns HTTP 401 (`Access Denied Invalid Token`), refresh the token from the Audiofy login and rerun the daemon.

### Automatic Backups
Every time `agents.json` is modified by the daemon, a timestamped backup copy is saved in `call_processor/data/agent_voiceprints/`. If an automated update ever degrades accuracy, you can safely revert to a previous backup state.
