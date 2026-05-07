#!/bin/bash

# Malaria Detection App - Robust Development Launcher
# This script ensures a clean start by killing old processes and launching both services.

echo "🚀 Starting Malaria Detection App..."

# 1. Cleanup old processes to prevent port conflicts
echo "🧹 Cleaning up old processes on ports 8000 and 5173..."
lsof -ti :8000,5173 | xargs kill -9 2>/dev/null

# Function to clean up background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# 2. Start the Backend (FastAPI)
echo "📂 Starting AI Backend (FastAPI)..."
if [ -d "venv" ]; then
    ./venv/bin/python app.py > backend.log 2>&1 &
else
    python3 app.py > backend.log 2>&1 &
fi
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# Check if backend started successfully
if ps -p $BACKEND_PID > /dev/null; then
    echo "✅ Backend started (PID: $BACKEND_PID). Logs are in backend.log"
else
    echo "❌ Failed to start backend. Check backend.log for details:"
    tail -n 10 backend.log
    exit 1
fi

# 3. Start the Frontend (Vite)
echo "🌐 Starting Frontend (Vite)..."
cd frontend
# Run in background so we can trap exit
npm run dev &
FRONTEND_PID=$!

echo "✨ Both services are running!"
echo "   - Backend: http://localhost:8000"
echo "   - Frontend: http://localhost:5173"
echo "   - Press Ctrl+C to stop both."

# Keep the script running
wait $BACKEND_PID $FRONTEND_PID
