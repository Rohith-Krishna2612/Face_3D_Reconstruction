@echo off
echo ========================================
echo Face Restoration Web Application
echo ========================================
echo.

echo Starting Backend (FastAPI)...
start "Backend" cmd /k "cd backend && python main.py"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend (React)...
start "Frontend" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo Application Started!
echo ========================================
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause > nul
