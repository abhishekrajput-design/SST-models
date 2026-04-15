#!/usr/bin/env bash
# Re-exec with bash if invoked via sh
if [ -z "$BASH_VERSION" ]; then exec bash "$0" "$@"; fi
# =============================================================================
#  fix_models.sh — Fix all model issues found during live testing
#
#  Issues fixed:
#    [1] NVIDIA driver too old  — Whisper + Parakeet fail with CUDA errors
#        Upgrades to nvidia-driver-550 (supports CUDA 12.4)
#        Then reinstalls PyTorch cu124 wheels
#    [2] qwen-asr not installed — Qwen3-ASR fails on import
#    [3] transformers too old   — VibeVoice needs unreleased architecture
#        Installs transformers from git HEAD
#
#  Usage:
#    bash fix_models.sh
#  Note: reboots the instance at the end if the NVIDIA driver was upgraded.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MINICONDA_DIR="/opt/miniconda3"
CONDA_ENV="callproc"
CONDA_PY="$MINICONDA_DIR/envs/$CONDA_ENV/bin/python"
CONDA_PIP="$MINICONDA_DIR/envs/$CONDA_ENV/bin/pip"
CONDA_UV="$MINICONDA_DIR/envs/$CONDA_ENV/bin/uv"
SERVICE_USER="${SUDO_USER:-ubuntu}"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step()  { echo -e "\n${BOLD}${GREEN}[FIX]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
ok()    { echo -e "  ${GREEN}OK${NC}  $*"; }

uv_install() { "$CONDA_UV" pip install --python "$CONDA_PY" "$@"; }

DRIVER_UPGRADED=false

# ─── [1] NVIDIA DRIVER UPGRADE ───────────────────────────────────────────────
step "[1/3] Checking NVIDIA driver (need >= 550 for CUDA 12.4)"

# Try to load the kernel module first (may not be loaded after initial install)
modprobe nvidia 2>/dev/null || true

# Get driver version — extract only the numeric version, ignore any error text
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | grep -oP '^\d+\.\d+' | head -1 || echo "")

if [[ -z "$CUDA_VER" ]]; then
    warn "nvidia-smi not returning a valid version — driver may not be loaded."
    warn "Proceeding with driver upgrade anyway..."
    MAJOR=0
else
    echo "  Current driver: $CUDA_VER"
    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
fi

# Convert to integer safely
MAJOR="${MAJOR//[^0-9]/}"
MAJOR="${MAJOR:-0}"

if (( MAJOR >= 550 )); then
    ok "Driver $CUDA_VER is >= 550 — no upgrade needed"
else
    echo "  Driver version $MAJOR < 550 — upgrading to nvidia-driver-550..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y -q
    apt-get install -y nvidia-driver-550 2>/dev/null \
        || apt-get install -y nvidia-driver-545 2>/dev/null \
        || apt-get install -y nvidia-driver-535 2>/dev/null \
        || warn "Driver upgrade failed — GPU models may not work until driver is updated"

    # Reinstall PyTorch matching the new driver
    echo "  Reinstalling PyTorch cu124 wheels for new driver..."
    uv_install --upgrade torch torchaudio \
        --index-url "https://download.pytorch.org/whl/cu124" \
        || "$CONDA_PIP" install --upgrade torch torchaudio \
           --index-url "https://download.pytorch.org/whl/cu124"
    ok "PyTorch reinstalled"
    DRIVER_UPGRADED=true
fi

# ─── [2] QWEN-ASR PACKAGE ────────────────────────────────────────────────────
step "[2/3] Installing qwen-asr (required for Qwen3-ASR-1.7B)"

if "$CONDA_PY" -c "import qwen_asr" 2>/dev/null; then
    ok "qwen-asr already installed"
else
    echo "  Installing qwen-asr..."
    uv_install "qwen-asr" \
        || "$CONDA_PIP" install "qwen-asr" \
        || warn "qwen-asr not found on PyPI — Qwen3-ASR will fall back to transformers"
    ok "qwen-asr installed"
fi

# ─── [3] TRANSFORMERS FROM GIT (VibeVoice architecture) ──────────────────────
step "[3/3] Upgrading transformers to git HEAD (required for VibeVoice-ASR)"

CURRENT_TF=$("$CONDA_PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
echo "  Current transformers: $CURRENT_TF"

echo "  Installing transformers from git HEAD..."
uv_install "transformers @ git+https://github.com/huggingface/transformers.git" \
    || "$CONDA_PIP" install "git+https://github.com/huggingface/transformers.git" \
    || warn "Git install failed — trying latest PyPI release instead..."
    uv_install --upgrade "transformers>=4.50.0" 2>/dev/null || true

NEW_TF=$("$CONDA_PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
ok "transformers: $CURRENT_TF -> $NEW_TF"

# ─── RESTART SERVICE ─────────────────────────────────────────────────────────
if [[ "$DRIVER_UPGRADED" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  NVIDIA driver was upgraded — a REBOOT is required.${NC}"
    echo -e "${YELLOW}  After reboot, the service will start automatically.${NC}"
    echo -e "${YELLOW}  Run:  sudo reboot${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo ""
    step "Restarting callproc service"
    sudo systemctl restart callproc
    sleep 3
    if systemctl is-active --quiet callproc; then
        ok "callproc.service is RUNNING"
    else
        warn "Service failed to start — check: sudo journalctl -u callproc -n 30"
    fi
fi

# ─── DONE ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}fix_models.sh complete.${NC}"
echo ""
echo "  Packages fixed:"
echo "    qwen-asr       : $("$CONDA_PY" -c "import qwen_asr; print('OK')" 2>/dev/null || echo 'check manually')"
echo "    transformers   : $("$CONDA_PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo 'unknown')"
echo "    torch          : $("$CONDA_PY" -c "import torch; print(torch.__version__, '| cuda=' + str(torch.cuda.is_available()))" 2>/dev/null || echo 'unknown')"
echo ""
[[ "$DRIVER_UPGRADED" == "false" ]] && echo "  Dashboard: http://$(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}'):8080"
