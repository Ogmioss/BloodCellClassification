#!/bin/bash
# Stop Streamlit app using PID file

set -e

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Configuration paths
PID_FILE="$PROJECT_ROOT/streamlit.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No PID file found. Streamlit is not running."
    exit 1
fi

# Read PID
PID=$(cat "$PID_FILE")

# Validate PID is a number
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo "❌ Invalid PID in file: $PID"
    rm -f "$PID_FILE"
    exit 1
fi

# Check if process is running
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  Process $PID is not running (stale PID file)"
    rm -f "$PID_FILE"
    exit 1
fi

# Verify it's actually a Streamlit process
PROCESS_CMD=$(ps -p "$PID" -o comm= 2>/dev/null || echo "")
if [[ ! "$PROCESS_CMD" =~ (python|streamlit|uv) ]]; then
    echo "⚠️  Warning: PID $PID doesn't appear to be a Streamlit process"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Stop the process
echo "🛑 Stopping Streamlit (PID: $PID)..."

# Try graceful shutdown first (SIGTERM)
kill -TERM "$PID" 2>/dev/null || true

# Wait up to 10 seconds for graceful shutdown
WAIT_COUNT=0
while ps -p "$PID" > /dev/null 2>&1 && [ $WAIT_COUNT -lt 10 ]; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

# Force kill if still running (SIGKILL)
if ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  Process didn't stop gracefully, forcing shutdown..."
    kill -KILL "$PID" 2>/dev/null || true
    sleep 1
fi

# Verify process is stopped
if ps -p "$PID" > /dev/null 2>&1; then
    echo "❌ Failed to stop process $PID"
    exit 1
else
    echo "✅ Streamlit stopped successfully"
    rm -f "$PID_FILE"
fi
