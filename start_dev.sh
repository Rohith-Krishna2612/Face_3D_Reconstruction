#!/bin/bash

# Start both backend and frontend development servers

echo "🚀 Starting Face 3D Reconstruction Development Servers"
echo "========================================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "Please activate your virtual environment first:"
    echo "   source venv/bin/activate  # Linux/Mac"
    echo "   venv\\Scripts\\activate     # Windows"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start backend server in background
echo "🔧 Starting FastAPI backend server..."
python backend/main.py &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend development server
echo "⚛️  Starting React frontend server..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Development servers started!"
echo ""
echo "🔗 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap 'echo "🛑 Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait