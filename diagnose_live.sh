#!/usr/bin/env bash
# diagnose_live.sh — dump full state so we can debug a failed pipeline run.
# Usage:  bash diagnose_live.sh [result_id]
# If no result_id is given, uses the last one reported by /api/status.

set +e

REPO_DIR="/home/ubuntu/projects/SST-models"
APP_DIR="$REPO_DIR/call_processor"
LOG="/var/log/callproc/server.log"

RESULT_ID="${1:-}"
if [ -z "$RESULT_ID" ]; then
    RESULT_ID=$(curl -s --max-time 5 http://localhost:8080/api/status \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('result_id') or '')" 2>/dev/null)
fi

echo "=============================================================="
echo "  LIVE PIPELINE DIAGNOSTIC"
echo "  Result ID: ${RESULT_ID:-<none>}"
echo "  Time:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

echo
echo "── 1. DISK SPACE ─────────────────────────────────────────────"
df -h /
echo
echo "── 2. MEMORY + GPU ───────────────────────────────────────────"
free -h
echo
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -5

echo
echo "── 3. SERVER STATUS ──────────────────────────────────────────"
curl -s --max-time 5 http://localhost:8080/api/status
echo

echo
echo "── 4. EXPECTED RESULT FOLDER ─────────────────────────────────"
if [ -n "$RESULT_ID" ]; then
    echo "Path: $APP_DIR/data/processed/$RESULT_ID/"
    ls -la "$APP_DIR/data/processed/$RESULT_ID/" 2>&1
else
    echo "(no result_id available)"
fi

echo
echo "── 5. ALL PROCESSED FOLDERS FOR THIS AUDIO ───────────────────"
if [ -n "$RESULT_ID" ]; then
    # strip off __<model> suffix to find all variants for this audio
    BASE="${RESULT_ID%__*}"
    ls -la "$APP_DIR/data/processed/" 2>/dev/null | grep "$BASE" || echo "(nothing matched $BASE)"
fi

echo
echo "── 6. ENHANCED AUDIO FILE ON DISK ────────────────────────────"
if [ -n "$RESULT_ID" ]; then
    BASE="${RESULT_ID%__*}"
    ls -la "$APP_DIR/data/raw_calls/$BASE.mp3" "$APP_DIR/data/raw_calls/${BASE#enhanced_}.mp3" 2>&1
fi

echo
echo "── 7. SERVER PROCESS ─────────────────────────────────────────"
ps -ef | grep -E "python.*ui\.py" | grep -v grep | head -3

echo
echo "── 8. SERVER LOG — LAST 200 LINES ────────────────────────────"
tail -200 "$LOG" 2>&1

echo
echo "── 9. SERVER LOG — TRACE FOR THIS RUN ────────────────────────"
if [ -n "$RESULT_ID" ]; then
    BASE="${RESULT_ID%__*}"
    # Grab pipeline trace lines: model loads, errors, result-save prints, etc.
    grep -n -E "vrcta2|${BASE#enhanced_}|$RESULT_ID|\[UI\]|Transcrib|result\.json|Pipeline complete|Traceback|No space|OSError|errno|FAILED|CUDA|OOM" "$LOG" 2>/dev/null | tail -120
fi

echo
echo "── 10. GIT STATUS ────────────────────────────────────────────"
cd "$REPO_DIR" && git log --oneline -3
echo
git status --short
echo
echo "=============================================================="
echo "  Paste the entire output above to Claude."
echo "=============================================================="
