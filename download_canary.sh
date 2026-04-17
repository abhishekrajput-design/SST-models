#!/usr/bin/env bash
# download_canary.sh — Download and cache Canary-Qwen 2.5B model (~5 GB)
set -e

REPO_DIR="/home/ubuntu/projects/SST-models"
PYTHON="/opt/miniconda3/envs/callproc/bin/python"

echo "==> Downloading Canary-Qwen 2.5B (~5 GB, may take 10-20 min)..."

cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -o allexport
    source ".env"
    set +o allexport
fi

"$PYTHON" - <<'EOF'
import sys
sys.path.insert(0, 'call_processor')
from src.transcribers import get_transcriber
t = get_transcriber('canary-qwen-2.5b')
print("Loading model (downloads if not cached)...")
t.load()
print("Unloading...")
t.unload()
print("Canary-Qwen 2.5B: OK — cached and ready")
EOF
