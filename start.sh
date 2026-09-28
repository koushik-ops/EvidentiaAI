#!/usr/bin/env bash
set -e

echo "========================================================"
echo " Starting DriveOps / EvidentiaAI Cloud Services...      "
echo "========================================================"

# Launch the background Cloud Evidence Watcher
python -u watch_cloud_evidence.py &

# Start Web Dashboard Server with Gunicorn
PORT="${PORT:-5000}"
echo "Launching Web Dashboard with Gunicorn on port $PORT..."
exec gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 4 --timeout 180 web_server:app
