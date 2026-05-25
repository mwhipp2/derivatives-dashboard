#!/bin/bash
# Derivatives Dashboard — Restart Script
# Double-click this file in Finder to stop and restart the dashboard.

# Find the folder this script lives in (same folder as app.py)
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==================================="
echo "  Derivatives Dashboard Restart"
echo "==================================="

# Kill any running instance of app.py
PIDS=$(pgrep -f "python.*app.py")
if [ -n "$PIDS" ]; then
    echo "Stopping existing dashboard (PID $PIDS)..."
    kill $PIDS
    sleep 1
fi

# Start fresh
echo "Starting dashboard..."
echo "Open your browser to: http://127.0.0.1:8050"
echo ""
cd "$DIR"
python3 app.py
