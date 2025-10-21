#!/bin/bash
# Launch Streamlit app with PID tracking

set -e

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Configuration paths
PID_FILE="$PROJECT_ROOT/streamlit.pid"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/streamlit.log"
APP_PATH="$PROJECT_ROOT/src/app.py"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if Streamlit is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "❌ Streamlit is already running (PID: $PID)"
        echo "   Use ./stop.sh to stop it first"
        exit 1
    else
        echo "⚠️  Stale PID file found, removing..."
        rm -f "$PID_FILE"
    fi
fi

# Verify app file exists
if [ ! -f "$APP_PATH" ]; then
    echo "❌ Application file not found: $APP_PATH"
    exit 1
fi

# Launch Streamlit in background
echo "🚀 Starting Streamlit application..."
echo "   App: $APP_PATH"
echo "   Logs: $LOG_FILE"

nohup uv run streamlit run "$APP_PATH" > "$LOG_FILE" 2>&1 &
STREAMLIT_PID=$!

# Save PID
echo "$STREAMLIT_PID" > "$PID_FILE"

# Wait a moment to check if process started successfully
sleep 2

if ps -p "$STREAMLIT_PID" > /dev/null 2>&1; then
    echo "✅ Streamlit started successfully (PID: $STREAMLIT_PID)"
    echo "   Access the app at: http://localhost:8501"
    echo "   View logs: tail -f $LOG_FILE"
else
    echo "❌ Failed to start Streamlit"
    echo "   Check logs: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
