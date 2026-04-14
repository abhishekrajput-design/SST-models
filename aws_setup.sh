#!/usr/bin/env bash
# Re-exec with bash if invoked via sh (Ubuntu sh = dash, no pipefail support)
if [ -z "$BASH_VERSION" ]; then exec bash "$0" "$@"; fi
# =============================================================================
#  aws_setup.sh — One-shot AWS GPU server setup for Call Processor (Conda)
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
#    [1] System packages  — ffmpeg, git, curl, build-tools
#    [2] CUDA check       — installs CUDA 12.4 toolkit if not already present
#    [3] Miniconda        — installs Miniconda3 if not already present
#    [4] Conda env        — creates callproc env with Python 3.11
#    [5] PyTorch (CUDA)   — conda install pytorch + torchaudio cu124
#    [6] Requirements     — pip install -r call_processor/requirements.txt
#    [7] NeMo toolkit     — nemo_toolkit[asr] (Parakeet)
#    [8] DeepFilterNet3   — neural denoiser
#    [9] Model download   — all 6 models via download_models.py
#   [10] Systemd service  — callproc.service (auto-start on reboot)
#   [11] Firewall         — open port 8080
# =============================================================================

set -euo pipefail
# Uncomment for verbose debug output:
# set -x

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"
APP_DIR="$REPO_DIR/call_processor"
LOG_DIR="/var/log/callproc"
SERVICE_USER="${SUDO_USER:-ubuntu}"
PORT=8080
PYTHON_VERSION="3.11"
CONDA_ENV="callproc"
MINICONDA_DIR="/opt/miniconda3"
MINICONDA_INSTALLER="/tmp/miniforge.sh"
CUDA_VERSION="12-4"                  # used for apt package names
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
step "[1/11] Installing system packages"
export DEBIAN_FRONTEND=noninteractive

echo "  Fixing any broken package state..."
apt-get update -y
apt --fix-broken install -y 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

echo "  Installing core packages..."
apt-get install -y curl wget git unzip build-essential pkg-config \
    || die "Failed to install core build tools"

apt-get install -y ffmpeg \
    || die "Failed to install ffmpeg"

# libsndfile (audio I/O for speechbrain/soundfile)
apt-get install -y libsndfile1 2>/dev/null || true
apt-get install -y libsndfile1-dev 2>/dev/null || true

# Optional monitoring tools — never fail the script
echo "  Installing optional packages (non-fatal)..."
for pkg in htop iotop nvtop nginx sox portaudio19-dev; do
    apt-get install -y "$pkg" 2>/dev/null || true
done

echo "  ffmpeg:  $(ffmpeg -version 2>&1 | head -1)"

# ─── [2] CUDA CHECK / INSTALL ────────────────────────────────────────────────
step "[2/11] Checking CUDA / NVIDIA driver"

modprobe nvidia 2>/dev/null || true

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")

if [[ -z "$GPU_INFO" ]]; then
    warn "nvidia-smi not available. Attempting to install NVIDIA drivers..."
    apt-get install -y ubuntu-drivers-common 2>/dev/null || true
    ubuntu-drivers install 2>/dev/null || apt-get install -y nvidia-driver-535 2>/dev/null || true
    modprobe nvidia 2>/dev/null || true
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
fi

PYTORCH_CHANNEL="pytorch-cuda=12.4"
if [[ -n "$GPU_INFO" ]]; then
    echo "  GPU: $GPU_INFO"
    if echo "$GPU_INFO" | grep -qi "T4" && [[ -z "$SKIP_MODELS" ]]; then
        warn "T4 GPU (16 GB VRAM) — VibeVoice needs ~18 GB. Set SKIP_MODELS=vibevoice to skip."
    fi
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
    PYTORCH_CHANNEL="cpuonly"
fi

# ─── [3] MINIFORGE ───────────────────────────────────────────────────────────
step "[3/11] Installing Miniforge3 (conda-forge, no ToS required)"
# Miniforge uses conda-forge by default — no Anaconda Terms of Service issues.
# If Miniconda is already installed at the same path we reuse it but accept ToS.

if [[ -f "$MINICONDA_DIR/bin/conda" ]]; then
    echo "  conda already installed at $MINICONDA_DIR"
else
    echo "  Downloading Miniforge installer..."
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -O "$MINICONDA_INSTALLER" \
        || die "Failed to download Miniforge"

    echo "  Installing Miniforge to $MINICONDA_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR" \
        || die "Miniforge installation failed"
    rm -f "$MINICONDA_INSTALLER"
    echo "  Miniforge installed"
fi

# Make conda available in this shell
export PATH="$MINICONDA_DIR/bin:$PATH"
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Accept Anaconda ToS for any default channels that may be configured
# (needed if Miniconda was already installed instead of Miniforge)
echo "  Accepting conda channel ToS (if required)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# Also make conda available system-wide for all users
if ! grep -q "miniconda3" /etc/environment 2>/dev/null; then
    echo "PATH=\"$MINICONDA_DIR/bin:$(cat /etc/environment | grep -oP '(?<=PATH=\")[^\"]+' || echo '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')\"" \
        > /etc/environment 2>/dev/null || true
fi

# Add to the service user's bashrc
BASHRC="/home/$SERVICE_USER/.bashrc"
if [[ -f "$BASHRC" ]] && ! grep -q "$MINICONDA_DIR" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# Conda (Miniforge)" >> "$BASHRC"
    echo "export PATH=\"$MINICONDA_DIR/bin:\$PATH\"" >> "$BASHRC"
    echo "source \"$MINICONDA_DIR/etc/profile.d/conda.sh\"" >> "$BASHRC"
fi

conda --version
conda update -n base -c conda-forge conda -y --quiet 2>/dev/null || true

# ─── [4] CONDA ENVIRONMENT ───────────────────────────────────────────────────
step "[4/11] Creating conda environment: $CONDA_ENV (Python $PYTHON_VERSION)"

if conda env list | grep -q "^$CONDA_ENV "; then
    echo "  Environment '$CONDA_ENV' already exists — skipping create"
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -c conda-forge -y --quiet \
        || die "Failed to create conda environment"
    echo "  Created conda env: $CONDA_ENV"
fi

# Shorthand helpers — run commands inside the conda env
CONDA_PY="$MINICONDA_DIR/envs/$CONDA_ENV/bin/python"
CONDA_PIP="$MINICONDA_DIR/envs/$CONDA_ENV/bin/pip"

"$CONDA_PIP" install --quiet --upgrade pip setuptools wheel

# Install uv inside the conda env for fast dependency resolution
# uv needs --python to target conda envs (it does not auto-detect them)
echo "  Installing uv (fast resolver)..."
"$CONDA_PIP" install --quiet uv
CONDA_UV="$MINICONDA_DIR/envs/$CONDA_ENV/bin/uv"
# Wrapper: always pass --python so uv installs into the conda env
uv_install() { "$CONDA_UV" pip install --python "$CONDA_PY" "$@"; }

echo "  Python: $($CONDA_PY --version)"

# ─── [5] PYTORCH (CUDA) ──────────────────────────────────────────────────────
step "[5/11] Installing PyTorch with CUDA (pip wheels)"
# Use official pip wheels — more reliable than conda channel for CUDA builds

if [[ "$PYTORCH_CHANNEL" == "cpuonly" ]]; then
    PIP_TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    PIP_TORCH_INDEX="https://download.pytorch.org/whl/cu124"
fi

uv_install torch torchaudio --index-url "$PIP_TORCH_INDEX" \
    || "$CONDA_PIP" install torch torchaudio --index-url "$PIP_TORCH_INDEX" \
    || die "PyTorch installation failed"

"$CONDA_PY" -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}')"

# ─── [6] PROJECT REQUIREMENTS ────────────────────────────────────────────────
step "[6/11] Installing project requirements"

# Pre-install numba>=0.59 before librosa — older numba fails on Python 3.12
echo "  Pre-installing numba (Python 3.12 compat)..."
uv_install "numba>=0.59.0"

# Pin lightning to avoid deep resolver issues from pyannote.audio
echo "  Pre-pinning lightning to avoid resolution-too-deep..."
uv_install "lightning>=2.0,<2.5" "pytorch-lightning>=2.0,<2.5"

echo "  Installing requirements.txt..."
uv_install -r "$APP_DIR/requirements.txt" \
    || "$CONDA_PIP" install --use-deprecated=legacy-resolver -r "$APP_DIR/requirements.txt"

# Additional packages not in requirements.txt but needed at runtime
uv_install \
    "python-dotenv" \
    "deepgram-sdk>=3.0.0" \
    "ctranslate2>=4.0.0"

# ─── [7] NeMo TOOLKIT (Parakeet) ─────────────────────────────────────────────
step "[7/11] Installing NVIDIA NeMo (Parakeet TDT)"
"$CONDA_PIP" install --quiet \
    "nemo_toolkit[asr]>=2.4.0" \
    "Cython" \
    "packaging" \
    2>/dev/null || warn "NeMo install had warnings — Parakeet may still work"

# ─── [8] DEEPFILTERNET3 (neural denoiser) ────────────────────────────────────
step "[8/11] Installing DeepFilterNet3"
"$CONDA_PIP" install --quiet deepfilterlib deepfilternet3 2>/dev/null || \
"$CONDA_PIP" install --quiet df 2>/dev/null || \
    warn "DeepFilterNet3 not available — angelina pipeline will skip it"

# ─── [9] DOWNLOAD ALL MODELS ─────────────────────────────────────────────────
step "[9/11] Downloading models (this will take 10-30 min on first run)"
mkdir -p "$LOG_DIR"

SKIP_FLAGS=""
[[ -n "$SKIP_MODELS" ]] && SKIP_FLAGS="--skip $SKIP_MODELS"

cd "$APP_DIR"
# Load .env so HF_TOKEN and DEEPGRAM_API_KEY are visible to downloader
set -o allexport; [[ -f "$REPO_DIR/.env" ]] && source "$REPO_DIR/.env"; set +o allexport

"$CONDA_PY" download_models.py $SKIP_FLAGS 2>&1 | tee "$LOG_DIR/download_models.log"
echo "  Models saved to: $APP_DIR/models/"

# ─── [10] SYSTEMD SERVICE ────────────────────────────────────────────────────
step "[10/11] Creating systemd service: callproc.service"
mkdir -p "$LOG_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

# Create a template .env if none exists — service will start without API keys
# and user can fill them in later with: nano $REPO_DIR/.env
if [[ ! -f "$REPO_DIR/.env" ]]; then
    warn ".env not found — creating empty template at $REPO_DIR/.env"
    cat > "$REPO_DIR/.env" <<'ENVEOF'
# Call Processor environment variables
# Fill in your API keys, then restart: sudo systemctl restart callproc

# Required for pyannote diarization (get token at https://huggingface.co/settings/tokens)
HF_TOKEN=

# Required for Deepgram Nova-3 transcription (get key at https://console.deepgram.com)
DEEPGRAM_API_KEY=
ENVEOF
    chown "$SERVICE_USER:$SERVICE_USER" "$REPO_DIR/.env"
    chmod 600 "$REPO_DIR/.env"
    echo "  Template .env created — edit it and add your API keys"
fi

cat > /etc/systemd/system/callproc.service <<EOF
[Unit]
Description=AI Call Processor Dashboard
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$APP_DIR
# Leading dash makes EnvironmentFile optional — service starts even without .env
EnvironmentFile=-$REPO_DIR/.env
Environment="PATH=$MINICONDA_DIR/envs/$CONDA_ENV/bin:$MINICONDA_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$MINICONDA_DIR/envs/$CONDA_ENV/bin/python ui.py
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

# ─── [11] FIREWALL ───────────────────────────────────────────────────────────
step "[11/11] Opening port $PORT"
if command -v ufw &>/dev/null; then
    ufw allow $PORT/tcp 2>/dev/null || true
    echo "  ufw: port $PORT open"
fi
echo "  IMPORTANT: Also open port $PORT in your AWS Security Group (EC2 console)"

# ─── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Setup complete!${NC}"
echo ""
echo -e "  Dashboard:  ${BOLD}http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}'):$PORT${NC}"
echo -e "  Logs:       $LOG_DIR/server.log"
echo -e "  Models:     $APP_DIR/models/"
echo -e "  Conda env:  conda activate $CONDA_ENV"
echo ""
echo -e "  Manage service:"
echo -e "    sudo systemctl status  callproc"
echo -e "    sudo systemctl restart callproc"
echo -e "    sudo journalctl -u callproc -f"
echo -e ""
echo -e "  Manual run (foreground):"
echo -e "    conda activate $CONDA_ENV && cd $APP_DIR && python ui.py"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
