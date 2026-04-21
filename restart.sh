#!/usr/bin/env bash
# restart.sh — Pull latest code and restart the Call Processor server
# Usage: bash restart.sh

set -e

REPO_DIR="/home/ubuntu/projects/SST-models"
APP_DIR="$REPO_DIR/call_processor"
PYTHON="/opt/miniconda3/envs/callproc/bin/python"
LOG="/var/log/callproc/server.log"

echo "==> Pulling latest code..."
cd "$REPO_DIR"
git pull origin main

echo "==> Killing existing server on port 8080..."
sudo fuser -k 8080/tcp 2>/dev/null || true
sleep 2

echo "==> Truncating old log to clear stale buffered traces..."
: > "$LOG"

echo "==> Starting server (unbuffered stdout so [UI] prints flush in real time)..."
cd "$APP_DIR"
# -u forces stdout/stderr unbuffered so we can tail -f the log mid-pipeline
# PYTHONUNBUFFERED=1 also covers any subprocess Python calls
PYTHONUNBUFFERED=1 nohup "$PYTHON" -u ui.py >> "$LOG" 2>&1 &
PID=$!
echo "    PID=$PID  Log=$LOG"

sleep 3
STATUS=$(curl -s --max-time 5 http://localhost:8080/api/status 2>/dev/null || echo "not responding")
echo "==> Status: $STATUS"
echo "==> Done. Open http://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}'):8080"
