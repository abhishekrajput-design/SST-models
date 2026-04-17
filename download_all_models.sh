#!/usr/bin/env bash
# download_all_models.sh — Pre-download all local GPU models to avoid
# first-run delays when a user selects a model in the UI.
#
# Usage: bash download_all_models.sh
# Skips models that are already cached.

set -e

REPO_DIR="/home/ubuntu/projects/SST-models"
APP_DIR="$REPO_DIR/call_processor"
PYTHON="/opt/miniconda3/envs/callproc/bin/python"

echo "=================================================="
echo " Model Pre-Download Script"
echo "=================================================="

cd "$APP_DIR"

# Load .env so HF_TOKEN is available
if [ -f "$REPO_DIR/.env" ]; then
    set -o allexport
    source "$REPO_DIR/.env"
    set +o allexport
fi

download_model() {
    local name="$1"
    echo ""
    echo "--- Downloading: $name ---"
    "$PYTHON" -c "
import sys
sys.path.insert(0, '.')
try:
    from src.transcribers import get_transcriber
    t = get_transcriber('$name')
    t.load()
    t.unload()
    print('  OK: $name downloaded and cached')
except Exception as e:
    print(f'  SKIP: $name — {e}')
" 2>&1 | grep -v "^\[NeMo\|^W0\|^OneLogger\|^No exporters\|triton"
}

# Whisper models (faster-whisper — downloads to models/faster-whisper/)
download_model "whisper-large-v3"
download_model "whisper-large-v3-turbo"
download_model "distil-whisper-large-v3.5"

# Parakeet (NeMo — downloads to models/nemo/)
download_model "parakeet-tdt-0.6b-v3"

# Cohere (HuggingFace — downloads to models/hf/)
download_model "cohere-transcribe-03-2026"

# Canary-Qwen (NeMo SALM — largest model ~5 GB)
download_model "canary-qwen-2.5b"

echo ""
echo "=================================================="
echo " All downloads complete!"
echo " Cloud models (Deepgram, AssemblyAI) need no download."
echo "=================================================="
