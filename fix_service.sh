#!/usr/bin/env bash
# fix_service.sh — Update callproc.service to use sst-models conda env
set -e

# Find sst-models Python
PYTHON=""
for p in \
    "/home/ubuntu/anaconda3/envs/sst-models/bin/python" \
    "/opt/miniconda3/envs/sst-models/bin/python" \
    "$(conda info --base 2>/dev/null)/envs/sst-models/bin/python"; do
    if [ -x "$p" ]; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: sst-models Python not found"
    exit 1
fi

echo "Using Python: $PYTHON"
"$PYTHON" --version

SERVICE_FILE="/etc/systemd/system/callproc.service"

sudo tee "$SERVICE_FILE" > /dev/null << SVCEOF
[Unit]
Description=AI Call Processor Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/projects/SST-models/call_processor
EnvironmentFile=-/home/ubuntu/projects/SST-models/.env
ExecStart=$PYTHON ui.py
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/callproc/server.log
StandardError=append:/var/log/callproc/server.log
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
SVCEOF

echo "Service file written:"
grep ExecStart "$SERVICE_FILE"

sudo systemctl daemon-reload
sudo systemctl stop callproc 2>/dev/null || true
sudo systemctl start callproc
sleep 3
sudo systemctl status callproc --no-pager
echo ""
curl -s http://localhost:8080/api/status
