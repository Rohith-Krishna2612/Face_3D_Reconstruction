@echo off
echo 🚀 Starting Face 3D Reconstruction Development Servers
echo ========================================================

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  Warning: No virtual environment detected.
    echo Please activate your virtual environment first:
    echo    venv\Scripts\activate
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Start backend server
echo 🔧 Starting FastAPI backend server...
start /b python backend/main.py

REM Give backend time to start
timeout /t 3 /nobreak >nul

REM Start frontend development server
echo ⚛️  Starting React frontend server...
cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing frontend dependencies...
    call npm install
)

REM Start frontend
start /b npm start

echo.
echo ✅ Development servers started!
echo.
echo 🔗 Access your application:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop servers, or close this window

pause