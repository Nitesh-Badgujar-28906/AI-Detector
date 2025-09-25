#!/bin/bash

# AI Detector - Startup Script
# This script starts both the Flask backend and React frontend

echo "🚀 Starting AI Detector Application..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Check if Node.js dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

# Start Flask backend in background
echo -e "${GREEN}Starting Flask API backend on port 5000...${NC}"
python flask_app.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

# Start React frontend
echo -e "${GREEN}Starting React frontend on port 3000...${NC}"
cd frontend
npm start &
REACT_PID=$!

# Function to cleanup on script exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $FLASK_PID 2>/dev/null
    kill $REACT_PID 2>/dev/null
    exit 0
}

# Trap cleanup on script exit
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}✅ AI Detector is now running!${NC}"
echo -e "${GREEN}📱 Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}🔧 API Backend: http://localhost:5000${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Keep script running
wait