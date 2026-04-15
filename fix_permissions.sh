#!/usr/bin/env bash
# Re-exec with bash if invoked via sh
if [ -z "$BASH_VERSION" ]; then exec bash "$0" "$@"; fi
# =============================================================================
#  fix_permissions.sh — Fix model directory ownership after sudo aws_setup.sh
#
#  Usage:
#    bash fix_permissions.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$REPO_DIR/call_processor"
SERVICE_USER="${SUDO_USER:-ubuntu}"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${BOLD}${GREEN}[FIX]${NC} $*"; }

# ─── Fix model directory ownership ───────────────────────────────────────────
step "Fixing model directory ownership -> $SERVICE_USER"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR/models/" 2>/dev/null \
    && echo "  chown OK: $APP_DIR/models/" \
    || echo "  chown skipped (already correct)"

# ─── Remove stale lock files ──────────────────────────────────────────────────
step "Removing stale .lock files"
LOCKS=$(find "$APP_DIR/models" -name "*.lock" 2>/dev/null || true)
if [[ -n "$LOCKS" ]]; then
    echo "$LOCKS" | while read -r f; do
        sudo rm -f "$f" && echo "  Removed: $f"
    done
else
    echo "  No lock files found"
fi

# ─── Fix .env ownership ───────────────────────────────────────────────────────
step "Fixing .env ownership"
if [[ -f "$REPO_DIR/.env" ]]; then
    sudo chown "$SERVICE_USER:$SERVICE_USER" "$REPO_DIR/.env"
    sudo chmod 600 "$REPO_DIR/.env"
    echo "  .env OK"
else
    echo "  .env not found — creating empty template"
    cat > "$REPO_DIR/.env" <<'ENVEOF'
# Fill in your API keys, then restart: sudo systemctl restart callproc

# Required for pyannote diarization (https://huggingface.co/settings/tokens)
HF_TOKEN=

# Required for Deepgram Nova-3 (https://console.deepgram.com)
DEEPGRAM_API_KEY=
ENVEOF
    sudo chown "$SERVICE_USER:$SERVICE_USER" "$REPO_DIR/.env"
    sudo chmod 600 "$REPO_DIR/.env"
    echo "  Template .env created at $REPO_DIR/.env"
fi

# ─── Restart service ─────────────────────────────────────────────────────────
step "Restarting callproc service"
sudo systemctl restart callproc
sleep 3
if systemctl is-active --quiet callproc; then
    echo "  callproc.service is RUNNING"
else
    echo "  callproc.service failed to start — check: sudo journalctl -u callproc -n 30"
    exit 1
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Done! Permissions fixed and service restarted.${NC}"
echo ""
IP=$(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null \
     || hostname -I | awk '{print $1}')
echo -e "  Dashboard: http://${IP}:8080"
echo -e "  Logs:      sudo journalctl -u callproc -f"
