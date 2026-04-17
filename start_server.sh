#!/usr/bin/env bash
# Re-exec with bash if invoked via sh (Ubuntu sh = dash, no pipefail support)
if [ -z "$BASH_VERSION" ]; then exec bash "$0" "$@"; fi
# =============================================================================
#  start_server.sh — Start / restart the Call Processor dashboard
#
#  Usage:
#    bash start_server.sh              # start in background (default)
#    bash start_server.sh --fg         # run in foreground (see logs live)
#    bash start_server.sh --test       # run 30-second smoke test
#    bash start_server.sh --status     # show service / process status
#    bash start_server.sh --stop       # stop all server processes
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$REPO_DIR/call_processor"
CONDA_ENV="callproc"
MINICONDA_DIR="/opt/miniconda3"
LOG_FILE="/var/log/callproc/server.log"
PORT=8080

# Resolve Python: prefer conda env, fall back to system python
if [[ -x "$MINICONDA_DIR/envs/$CONDA_ENV/bin/python" ]]; then
    PYTHON="$MINICONDA_DIR/envs/$CONDA_ENV/bin/python"
elif command -v conda &>/dev/null; then
    # conda is on PATH but installed elsewhere
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
    if [[ -n "$CONDA_BASE" && -x "$CONDA_BASE/envs/$CONDA_ENV/bin/python" ]]; then
        PYTHON="$CONDA_BASE/envs/$CONDA_ENV/bin/python"
    else
        PYTHON="$(command -v python3 || command -v python)"
    fi
else
    PYTHON="$(command -v python3 || command -v python)"
fi

# Fallback log location if /var/log/callproc not writable
if [[ ! -w "$(dirname "$LOG_FILE")" ]] 2>/dev/null; then
    LOG_FILE="$REPO_DIR/server.log"
fi

# ─── Load .env ───────────────────────────────────────────────────────────────
if [[ -f "$REPO_DIR/.env" ]]; then
    set -o allexport; source "$REPO_DIR/.env"; set +o allexport
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────
is_running() {
    curl -s --max-time 2 "http://localhost:$PORT/api/status" &>/dev/null
}

stop_server() {
    echo "Stopping existing server on port $PORT..."
    if systemctl is-active --quiet callproc 2>/dev/null; then
        sudo systemctl stop callproc
        echo "  Stopped callproc.service"
        return
    fi
    PIDS=$(lsof -ti tcp:$PORT 2>/dev/null || true)
    if [[ -n "$PIDS" ]]; then
        kill -9 $PIDS 2>/dev/null || true
        echo "  Killed PIDs: $PIDS"
    else
        echo "  No process found on port $PORT"
    fi
}

show_status() {
    echo "── Python ──────────────────────────────────────────"
    echo "  $PYTHON ($($PYTHON --version 2>&1))"
    echo ""
    echo "── Service status ──────────────────────────────────"
    if systemctl is-active --quiet callproc 2>/dev/null; then
        systemctl status callproc --no-pager | head -20
    else
        PIDS=$(lsof -ti tcp:$PORT 2>/dev/null || true)
        [[ -n "$PIDS" ]] && echo "  Background process PIDs: $PIDS" || echo "  No process on port $PORT"
    fi
    echo ""
    echo "── API status ──────────────────────────────────────"
    curl -s "http://localhost:$PORT/api/status" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Server not responding"
    echo ""
    echo "── Recent logs ─────────────────────────────────────"
    [[ -f "$LOG_FILE" ]] && tail -20 "$LOG_FILE" || echo "  No log file at $LOG_FILE"
}

smoke_test() {
    echo "── Smoke test (30s timeout) ────────────────────────"
    START=$SECONDS
    while ! is_running; do
        if (( SECONDS - START > 30 )); then
            echo "  FAIL: server did not respond within 30s"
            exit 1
        fi
        echo "  waiting... ($((SECONDS - START))s)"
        sleep 2
    done

    echo "  Server responding on port $PORT"

    STATUS=$(curl -s "http://localhost:$PORT/api/status")
    echo "  /api/status: $STATUS"

    CALLS=$(curl -s "http://localhost:$PORT/api/calls" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} calls')" 2>/dev/null || echo "parse error")
    echo "  /api/calls:  $CALLS"

    echo "  PASS"
}

# ─── Argument parsing ────────────────────────────────────────────────────────
MODE="bg"
for arg in "$@"; do
    case $arg in
        --fg)     MODE="fg"     ;;
        --test)   MODE="test"   ;;
        --status) MODE="status" ;;
        --stop)   MODE="stop"   ;;
    esac
done

case $MODE in
    stop)
        stop_server
        exit 0
        ;;
    status)
        show_status
        exit 0
        ;;
    test)
        smoke_test
        exit 0
        ;;
esac

# ─── Start server ────────────────────────────────────────────────────────────
echo "Starting Call Processor server on port $PORT..."
echo "  Python: $PYTHON"

# Prefer systemd if service exists
if systemctl is-enabled --quiet callproc 2>/dev/null; then
    sudo systemctl restart callproc
    sleep 2
    if systemctl is-active --quiet callproc; then
        echo "  callproc.service restarted OK"
        smoke_test
    else
        echo "  Service failed — check: sudo journalctl -u callproc -n 50"
        exit 1
    fi
    exit 0
fi

# No systemd — kill any existing process and start fresh
stop_server 2>/dev/null || true

mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || LOG_FILE="$REPO_DIR/server.log"

if [[ "$MODE" == "fg" ]]; then
    echo "  Running in foreground. Ctrl+C to stop."
    cd "$APP_DIR"
    exec "$PYTHON" ui.py
else
    cd "$APP_DIR"
    nohup "$PYTHON" ui.py >> "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "  PID: $SERVER_PID  |  Log: $LOG_FILE"
    smoke_test
    IP=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null \
         || hostname -I | awk '{print $1}')
    echo ""
    echo "  Dashboard: http://${IP}:${PORT}"
    echo "  Tail logs: tail -f $LOG_FILE"
fi
