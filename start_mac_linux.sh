#!/bin/bash
set -e

echo ""
echo " =========================================="
echo "  Codeyy - AI Code Analysis (Gemini)"
echo " =========================================="
echo ""

cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null; then
    echo " ERROR: Python 3 not found. Install from https://python.org"
    exit 1
fi

cd backend

if [ ! -d "venv" ]; then
    echo " Setting up virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo " Installing dependencies..."
pip install -r requirements.txt -q

if [ ! -f ".env" ]; then
    echo ""
    echo " WARNING: backend/.env file does not exist."
    echo " Creating template file. You can enter your API key in the browser interface."
    echo ""
    echo "GEMINI_API_KEY=YOUR_KEY_HERE" > .env
fi

if grep -q "YOUR_KEY_HERE" .env; then
    echo ""
    echo " WARNING: API key in backend/.env is still a placeholder."
    echo " Please enter your API key directly in the browser interface once it opens."
    echo ""
fi

echo ""
echo " Starting Codeyy at http://localhost:8000"
echo " Press Ctrl+C to stop"
echo ""

sleep 1 && (open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null) &

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
