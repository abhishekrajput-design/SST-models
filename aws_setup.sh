#!/usr/bin/env bash
# Re-exec with bash if invoked via sh (Ubuntu sh = dash, no pipefail support)
if [ -z "$BASH_VERSION" ]; then exec bash "$0" "$@"; fi
# =============================================================================
#  aws_setup.sh — One-shot AWS GPU server setup for Call Processor
#
#  Targets: Ubuntu 22.04 LTS + NVIDIA GPU (g4dn / g5 / p3 family)
#  Recommended instance: g5.xlarge (A10G 24 GB VRAM) — fits all models
#  Minimum instance:     g4dn.xlarge (T4 16 GB VRAM) — skip VibeVoice
#
#  Usage:
#    # 1. SSH into a fresh Ubuntu 22.04 GPU instance
#    # 2. Upload this repo:
#    #      git clone https://github.com/abhishekrajput-design/SST-models.git
#    # 3. Copy your .env file (see CONFIGURATION section below)
#    # 4. Run:
#    #      chmod +x aws_setup.sh && sudo bash aws_setup.sh
#
#  What this script does:
#    [1] System packages  — ffmpeg, python3.11, git, curl, build-tools
#    [2] CUDA check       — installs CUDA 12.4 toolkit if not already present
#    [3] Python venv      — /opt/callproc/venv
#    [4] PyTorch (CUDA)   — torch + torchaudio cu124 wheels
#    [5] Requirements     — pip install -r call_processor/requirements.txt
#    [6] NeMo toolkit     — nemo_toolkit[asr] (Parakeet)
#    [7] DeepFilterNet3   — neural denoiser
#    [8] Model download   — all 6 models via download_models.py
#    [9] Systemd service  — callproc.service (auto-start on reboot)
#   [10] Firewall         — open port 8080
# =============================================================================

set -euo pipefail
# Uncomment for verbose debug output:
# set -x

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VENV_DIR="/opt/callproc/venv"
APP_DIR="$REPO_DIR/call_processor"
LOG_DIR="/var/log/callproc"
SERVICE_USER="${SUDO_USER:-ubuntu}"
PORT=8080
PYTHON_VERSION="3.11"
CUDA_VERSION="12-4"                  # used for apt package names
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
# Models to skip on small GPUs (T4 16 GB).  Set to "" to download all.
# SKIP_MODELS="vibevoice"            # uncomment for g4dn (T4 16 GB)
SKIP_MODELS=""
# ─────────────────────────────────────────────────────────────────────────────

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${BOLD}${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && die "Run as root: sudo bash aws_setup.sh"
[[ -d "$REPO_DIR" ]] || die "Repo not found at $REPO_DIR. Clone it first."
[[ -f "$APP_DIR/ui.py" ]] || die "ui.py not found — expected repo layout: $APP_DIR/ui.py"

# ─── [1] SYSTEM PACKAGES ─────────────────────────────────────────────────────
step "[1/10] Installing system packages"
export DEBIAN_FRONTEND=noninteractive

echo "  Fixing any broken package state..."
apt-get update -y
apt --fix-broken install -y 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

echo "  Installing core packages..."
# Install in small batches — easier to pinpoint failures
apt-get install -y curl wget git unzip build-essential pkg-config \
    || die "Failed to install core build tools"

apt-get install -y ffmpeg \
    || die "Failed to install ffmpeg"

# python3-venv may need the matching python3.X-venv; try both
apt-get install -y python3 python3-dev python3-pip || true
apt-get install -y python3-venv 2>/dev/null \
    || apt-get install -y python3.12-venv 2>/dev/null \
    || apt-get install -y python3.11-venv 2>/dev/null \
    || warn "python3-venv install failed — will try python -m venv directly"

# libsndfile (audio I/O for speechbrain/soundfile) — dev headers optional
apt-get install -y libsndfile1 2>/dev/null || true
apt-get install -y libsndfile1-dev 2>/dev/null || true   # may have unmet deps on some AMIs

# Optional monitoring / server tools — never fail the script
echo "  Installing optional packages (non-fatal)..."
for pkg in htop iotop nvtop nginx sox portaudio19-dev; do
    apt-get install -y "$pkg" 2>/dev/null || true
done

# Detect which python3.X is available and set PYTHON_VERSION accordingly
if python${PYTHON_VERSION} --version &>/dev/null 2>&1; then
    echo "  python${PYTHON_VERSION} available"
else
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '3\.\d+' | head -1)
    warn "python${PYTHON_VERSION_:-3.11} not found — using python${PYTHON_VERSION}"
fi

echo "  ffmpeg:  $(ffmpeg -version 2>&1 | head -1)"
echo "  python:  $(python3 --version)"

# ─── [2] CUDA CHECK / INSTALL ────────────────────────────────────────────────
step "[2/10] Checking CUDA / NVIDIA driver"

# Try loading the nvidia module if it exists but isn't loaded yet
modprobe nvidia 2>/dev/null || true

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")

if [[ -z "$GPU_INFO" ]]; then
    # No GPU visible — try installing NVIDIA open drivers
    warn "nvidia-smi not available. Attempting to install NVIDIA drivers..."
    apt-get install -y ubuntu-drivers-common 2>/dev/null || true
    ubuntu-drivers install 2>/dev/null || apt-get install -y nvidia-driver-535 2>/dev/null || true
    modprobe nvidia 2>/dev/null || true
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
fi

if [[ -n "$GPU_INFO" ]]; then
    echo "  GPU: $GPU_INFO"
    # Warn if T4 (16 GB) and VibeVoice not skipped
    if echo "$GPU_INFO" | grep -qi "T4" && [[ -z "$SKIP_MODELS" ]]; then
        warn "T4 GPU (16 GB VRAM) — VibeVoice needs ~18 GB. Set SKIP_MODELS=vibevoice to skip."
    fi
    # Install CUDA toolkit if nvcc missing
    if ! command -v nvcc &>/dev/null; then
        echo "  Installing CUDA ${CUDA_VERSION} toolkit..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
            && dpkg -i cuda-keyring_1.1-1_all.deb \
            && apt-get update -y \
            && apt-get install -y cuda-toolkit-${CUDA_VERSION} 2>/dev/null \
            && export PATH=/usr/local/cuda/bin:$PATH \
            && echo "  CUDA toolkit installed" \
            || warn "CUDA toolkit install failed — PyTorch CUDA wheels will still work via driver API"
        rm -f cuda-keyring_1.1-1_all.deb
    else
        echo "  CUDA toolkit: $(nvcc --version | grep release)"
    fi
else
    warn "No GPU found — installing CPU-only PyTorch. Transcription will be slow."
    warn "If this is a GPU instance, reboot after setup: sudo reboot"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# ─── [3] PYTHON VIRTUAL ENVIRONMENT ──────────────────────────────────────────
step "[3/10] Creating Python virtual environment at $VENV_DIR"
mkdir -p /opt/callproc
python${PYTHON_VERSION} -m venv "$VENV_DIR"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
"$VENV_PIP" install --quiet --upgrade pip setuptools wheel

# ─── [4] PYTORCH (CUDA) ──────────────────────────────────────────────────────
step "[4/10] Installing PyTorch with CUDA"
"$VENV_PIP" install --quiet \
    torch torchaudio \
    --index-url "$TORCH_INDEX"
"$VENV_PY" -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}')"

# ─── [5] PROJECT REQUIREMENTS ────────────────────────────────────────────────
step "[5/10] Installing project requirements"

# Pre-install numba>=0.59 before librosa — older numba fails on Python 3.12
echo "  Pre-installing numba (Python 3.12 compat)..."
"$VENV_PIP" install --quiet "numba>=0.59.0"

echo "  Installing requirements.txt..."
"$VENV_PIP" install -r "$APP_DIR/requirements.txt"

# Additional packages not in requirements.txt but needed at runtime
"$VENV_PIP" install --quiet \
    python-dotenv \
    deepgram-sdk>=3.0.0 \
    ctranslate2>=4.0.0

# ─── [6] NeMo TOOLKIT (Parakeet) ─────────────────────────────────────────────
step "[6/10] Installing NVIDIA NeMo (Parakeet TDT)"
# NeMo has heavy deps — install separately for clearer error reporting
"$VENV_PIP" install --quiet \
    "nemo_toolkit[asr]>=2.4.0" \
    "Cython" \
    "packaging" \
    2>/dev/null || warn "NeMo install had warnings — Parakeet may still work"

# ─── [7] DEEPFILTERNET3 (neural denoiser) ────────────────────────────────────
step "[7/10] Installing DeepFilterNet3"
"$VENV_PIP" install --quiet deepfilterlib deepfilternet3 2>/dev/null || \
"$VENV_PIP" install --quiet df 2>/dev/null || \
    warn "DeepFilterNet3 not available — angelina pipeline will skip it"

# ─── [8] DOWNLOAD ALL MODELS ─────────────────────────────────────────────────
step "[8/10] Downloading models (this will take 10-30 min on first run)"
mkdir -p "$LOG_DIR"

# Build skip flags
SKIP_FLAGS=""
[[ -n "$SKIP_MODELS" ]] && SKIP_FLAGS="--skip $SKIP_MODELS"

cd "$APP_DIR"
# Load .env so HF_TOKEN and DEEPGRAM_API_KEY are visible to downloader
set -o allexport; [[ -f "$REPO_DIR/.env" ]] && source "$REPO_DIR/.env"; set +o allexport

"$VENV_PY" download_models.py $SKIP_FLAGS 2>&1 | tee "$LOG_DIR/download_models.log"
echo "  Models saved to: $APP_DIR/models/"

# ─── [9] SYSTEMD SERVICE ─────────────────────────────────────────────────────
step "[9/10] Creating systemd service: callproc.service"
mkdir -p "$LOG_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

cat > /etc/systemd/system/callproc.service <<EOF
[Unit]
Description=AI Call Processor Dashboard
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$REPO_DIR/.env
ExecStart=$VENV_DIR/bin/python ui.py
Restart=on-failure
RestartSec=5
StandardOutput=append:$LOG_DIR/server.log
StandardError=append:$LOG_DIR/server.log
# Give GPU enough time to warm up
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable callproc.service
systemctl restart callproc.service
sleep 3
if systemctl is-active --quiet callproc.service; then
    echo "  callproc.service is RUNNING"
else
    warn "Service failed to start — check logs: journalctl -u callproc -n 50"
fi

# ─── [10] FIREWALL ───────────────────────────────────────────────────────────
step "[10/10] Opening port $PORT"
# ufw (Ubuntu default firewall)
if command -v ufw &>/dev/null; then
    ufw allow $PORT/tcp 2>/dev/null || true
    echo "  ufw: port $PORT open"
fi
# Also remind about AWS Security Group
echo "  IMPORTANT: Also open port $PORT in your AWS Security Group (EC2 console)"

# ─── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Setup complete!${NC}"
echo ""
echo -e "  Dashboard:  ${BOLD}http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}'):$PORT${NC}"
echo -e "  Logs:       $LOG_DIR/server.log"
echo -e "  Models:     $APP_DIR/models/"
echo ""
echo -e "  Manage service:"
echo -e "    sudo systemctl status  callproc"
echo -e "    sudo systemctl restart callproc"
echo -e "    sudo journalctl -u callproc -f"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
