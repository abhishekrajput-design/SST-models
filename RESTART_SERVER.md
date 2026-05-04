# Server Restart Guide - Production

**Date**: 2026-05-04  
**Status**: Production Ready

---

## 🔄 Main Restart Script

### Location
```
/path/to/SST-models/restart.sh
```

### What It Does
- Pulls latest code from GitHub (`git pull origin main`)
- Stops existing server on port 8080
- Clears old logs
- Starts server with unbuffered output
- Verifies server is running
- Shows server status and access URL

---

## 📋 Restart Commands

### Quick Restart (Production Server)

```bash
# From any directory
bash /home/ubuntu/projects/SST-models/restart.sh

# Or if in repo directory
cd /path/to/SST-models
bash restart.sh
```

### Expected Output
```
==> Pulling latest code...
Already up to date.

==> Killing existing server on port 8080...

==> Truncating old log to clear stale buffered traces...

==> Starting server (unbuffered stdout)...
    PID=12345  Log=/var/log/callproc/server.log

==> Status: {"status": "ok", "version": "1.0"}

==> Done. Open http://192.168.1.100:8080
```

---

## 🚀 Alternative Restart Methods

### Method 1: Using start_server.sh (Linux/Mac)

```bash
# Stop
bash start_server.sh --stop

# Wait a moment
sleep 2

# Start
bash start_server.sh
```

### Method 2: Manual Restart

```bash
# 1. Stop the server
sudo fuser -k 8080/tcp

# 2. Pull latest code
git pull origin main

# 3. Start server
cd call_processor
nohup python ui.py >> ../server.log 2>&1 &
```

### Method 3: Using Systemd (Recommended for Production)

```bash
# Restart service
sudo systemctl restart callproc

# Check status
sudo systemctl status callproc

# View logs
sudo journalctl -u callproc -f
```

### Method 4: Windows (Local/Dev)

```batch
# Stop (Ctrl+C in running terminal)

# Pull latest
git pull origin main

# Start
cd call_processor
python ui.py
```

---

## 🔍 Restart Script Details

### The Script (`restart.sh`)

```bash
#!/usr/bin/env bash
# restart.sh — Pull latest code and restart Call Processor server

set -e  # Exit on error

# Configuration (adjust for your environment)
REPO_DIR="/home/ubuntu/projects/SST-models"
APP_DIR="$REPO_DIR/call_processor"
PYTHON="/opt/miniconda3/envs/callproc/bin/python"
LOG="/var/log/callproc/server.log"

# 1. Pull latest code from GitHub
echo "==> Pulling latest code..."
cd "$REPO_DIR"
git pull origin main

# 2. Kill existing server on port 8080
echo "==> Killing existing server on port 8080..."
sudo fuser -k 8080/tcp 2>/dev/null || true
sleep 2

# 3. Clear old logs
echo "==> Truncating old log..."
: > "$LOG"

# 4. Start server with unbuffered output
echo "==> Starting server..."
cd "$APP_DIR"
PYTHONUNBUFFERED=1 nohup "$PYTHON" -u ui.py >> "$LOG" 2>&1 &
PID=$!
echo "    PID=$PID  Log=$LOG"

# 5. Wait for server to start
sleep 3

# 6. Check status
STATUS=$(curl -s --max-time 5 http://localhost:8080/api/status 2>/dev/null || echo "not responding")
echo "==> Status: $STATUS"

# 7. Show access URL
echo "==> Done. Open http://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}'):8080"
```

### Key Parameters to Customize

| Parameter | Current | Change to | Purpose |
|-----------|---------|-----------|---------|
| `REPO_DIR` | `/home/ubuntu/projects/SST-models` | Your repo path | Repository location |
| `APP_DIR` | `$REPO_DIR/call_processor` | Your app path | Application directory |
| `PYTHON` | `/opt/miniconda3/envs/callproc/bin/python` | Your Python path | Python executable |
| `LOG` | `/var/log/callproc/server.log` | Your log path | Log file location |
| `8080` | Port 8080 | Your port | Server port |

---

## ⚙️ Customize for Your Environment

### Step 1: Find Your Paths

```bash
# Find Python executable
which python3
# or if using conda
which conda
conda info --envs

# Find repo directory
pwd
# when in your repo

# Find app directory  
pwd
# when in call_processor folder
```

### Step 2: Edit restart.sh

```bash
# Copy and edit
cp restart.sh restart.sh.bak
nano restart.sh

# Replace these lines with your values:
# REPO_DIR="/your/repo/path"
# APP_DIR="/your/app/path"
# PYTHON="/your/python/path"
# LOG="/your/log/path"
```

### Step 3: Make Executable

```bash
chmod +x restart.sh
```

### Step 4: Test It

```bash
bash restart.sh
```

---

## 📊 Restart Workflow

### Complete Restart with Verification

```bash
# 1. Restart server
bash restart.sh

# 2. Wait for startup
sleep 5

# 3. Check status
curl http://localhost:8080/api/status

# 4. Test API
curl http://localhost:8080/api/calls | head -20

# 5. View logs
tail -50 /var/log/callproc/server.log
```

---

## 🚨 Troubleshooting Restart Issues

### Issue: Port 8080 Still in Use

```bash
# Find process on port 8080
lsof -i :8080

# Force kill it
sudo fuser -k 8080/tcp

# Or kill specific PID
sudo kill -9 12345
```

### Issue: Permission Denied

```bash
# Make script executable
chmod +x restart.sh

# Run with sudo if needed
sudo bash restart.sh
```

### Issue: Python Not Found

```bash
# Update PYTHON path in script
which python3

# Or use full path
/usr/bin/python3
/opt/miniconda3/envs/callproc/bin/python
```

### Issue: Log Directory Missing

```bash
# Create log directory
sudo mkdir -p /var/log/callproc
sudo chown ubuntu:ubuntu /var/log/callproc

# Or change LOG path in script
LOG="$REPO_DIR/server.log"
```

### Issue: Git Pull Fails

```bash
# Check git status
git status

# Stash changes if needed
git stash

# Retry pull
git pull origin main
```

### Issue: Server Won't Start

```bash
# Check Python version
python --version  # Should be 3.11+

# Check GPU
nvidia-smi

# Run in foreground to see errors
cd call_processor
python ui.py
```

---

## 📈 Monitoring After Restart

### Check Server Health

```bash
# API status
curl http://localhost:8080/api/status

# Recent calls
curl http://localhost:8080/api/calls?limit=5

# Full health check
bash start_server.sh --status
```

### Monitor Logs

```bash
# Tail logs (follow in real-time)
tail -f /var/log/callproc/server.log

# View last 100 lines
tail -100 /var/log/callproc/server.log

# Search for errors
grep -i error /var/log/callproc/server.log | tail -20
```

### Verify Latest Code

```bash
# Check commit
git log --oneline -1

# Expected: 477cea5 Add comprehensive test results...
```

---

## ⏰ Scheduled Restart

### Setup Cron Job for Daily Restart

```bash
# Edit crontab
crontab -e

# Add this line (restart daily at 2 AM)
0 2 * * * bash /home/ubuntu/projects/SST-models/restart.sh >> /var/log/callproc/restart.log 2>&1
```

### Setup Cron for Weekly Restart

```bash
# Edit crontab
crontab -e

# Add this line (restart every Sunday at 3 AM)
0 3 * * 0 bash /home/ubuntu/projects/SST-models/restart.sh >> /var/log/callproc/restart.log 2>&1
```

### Verify Cron Job

```bash
# List cron jobs
crontab -l

# Check cron logs
sudo tail -f /var/log/syslog | grep CRON
```

---

## 🔐 Production Restart Checklist

Before restarting in production:

- [ ] Backup current state: `cp -r data/processed data/processed.backup`
- [ ] Verify latest code: `git log --oneline -1`
- [ ] Check disk space: `df -h`
- [ ] Check GPU memory: `nvidia-smi`
- [ ] Notify users (if needed)
- [ ] Run restart: `bash restart.sh`
- [ ] Verify status: `curl http://localhost:8080/api/status`
- [ ] Test upload: Use UI or API
- [ ] Monitor logs: `tail -f /var/log/callproc/server.log`

---

## 📋 Restart Commands Summary

| Task | Command |
|------|---------|
| Quick restart | `bash restart.sh` |
| Stop only | `bash start_server.sh --stop` |
| Check status | `bash start_server.sh --status` |
| View logs | `tail -f /var/log/callproc/server.log` |
| Pull code | `git pull origin main` |
| Kill port 8080 | `sudo fuser -k 8080/tcp` |
| Restart systemd | `sudo systemctl restart callproc` |

---

## ✅ Restart Verification

After restarting, verify with this checklist:

```bash
# 1. Check process
ps aux | grep python | grep ui.py

# 2. Check port
lsof -i :8080

# 3. Test API
curl http://localhost:8080/api/status

# 4. Check logs for errors
grep -i error /var/log/callproc/server.log | tail -5

# 5. Test upload
curl -X POST -F "file=@test.mp3" http://localhost:8080/api/upload
```

---

## 🎯 Summary

**Main Restart Script**: `restart.sh`

**Quick Command**:
```bash
bash /path/to/SST-models/restart.sh
```

**What It Does**:
1. Pulls latest code
2. Stops old server
3. Starts new server
4. Verifies it's running

**Status**: ✅ Production Ready

---

For detailed deployment guide, see: **DEPLOYMENT_GUIDE.md**
