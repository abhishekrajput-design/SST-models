#!/bin/sh
set -eu

APP_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
PID_FILE="${PID_FILE:-$APP_DIR/ui.pid}"
STDOUT_LOG="${STDOUT_LOG:-$APP_DIR/ui_stdout.log}"
STDERR_LOG="${STDERR_LOG:-$APP_DIR/ui_stderr.log}"
STATUS_URL="${STATUS_URL:-http://127.0.0.1:8080/api/status}"
TAIL_LINES="${TAIL_LINES:-120}"

find_python() {
    if [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "$PYTHON"
    elif [ -x "$APP_DIR/.venv/bin/python" ]; then
        printf '%s\n' "$APP_DIR/.venv/bin/python"
    elif [ -x "$APP_DIR/../.venv/bin/python" ]; then
        printf '%s\n' "$APP_DIR/../.venv/bin/python"
    elif [ -x "$APP_DIR/../venv/bin/python" ]; then
        printf '%s\n' "$APP_DIR/../venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        echo "python not found. Activate your venv or set PYTHON=/path/to/python." >&2
        exit 1
    fi
}

pid_value() {
    if [ -f "$PID_FILE" ]; then
        tr -d '[:space:]' < "$PID_FILE"
    fi
}

is_running() {
    pid="$(pid_value)"
    [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1
}

status() {
    if is_running; then
        echo "UI server running: pid $(pid_value)"
    else
        echo "UI server not running"
    fi

    if command -v curl >/dev/null 2>&1; then
        curl -fsS "$STATUS_URL" || true
        echo
    else
        echo "curl not found; skipped $STATUS_URL"
    fi
}

start() {
    if is_running; then
        echo "UI server already running: pid $(pid_value)"
        status
        return 0
    fi

    python_bin="$(find_python)"
    mkdir -p "$APP_DIR/data/raw_calls" "$APP_DIR/data/processed"
    touch "$STDOUT_LOG" "$STDERR_LOG"

    cd "$APP_DIR"
    nohup "$python_bin" -u ui.py > "$STDOUT_LOG" 2> "$STDERR_LOG" &
    echo "$!" > "$PID_FILE"

    sleep 3
    if is_running; then
        echo "Started UI server: pid $(pid_value)"
        echo "Logs:"
        echo "  stdout: $STDOUT_LOG"
        echo "  stderr: $STDERR_LOG"
        status
    else
        echo "UI server failed to start. Recent logs:"
        tail -n 80 "$STDOUT_LOG" "$STDERR_LOG" || true
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop() {
    if ! is_running; then
        echo "UI server not running"
        rm -f "$PID_FILE"
        return 0
    fi

    pid="$(pid_value)"
    echo "Stopping UI server: pid $pid"
    kill "$pid" >/dev/null 2>&1 || true

    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            rm -f "$PID_FILE"
            echo "Stopped"
            return 0
        fi
        sleep 1
    done

    echo "Process did not exit after 20s; forcing stop"
    kill -9 "$pid" >/dev/null 2>&1 || true
    rm -f "$PID_FILE"
}

logs() {
    touch "$STDOUT_LOG" "$STDERR_LOG"
    tail -n "$TAIL_LINES" -f "$STDOUT_LOG" "$STDERR_LOG"
}

case "${1:-start}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 2
        ;;
esac
