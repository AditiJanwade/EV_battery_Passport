@echo off
title EV Battery Dashboard Launcher
echo ===================================================
echo   Starting EV Battery Health Intelligence System
echo ===================================================
echo.

echo [1/2] Launching Backend Server (Port 8000)...
start "EV Backend" cmd /k "cd backend && call venv\Scripts\activate 2>nul & uvicorn main:app --reload --port 8000"

echo [2/2] Launching Frontend Interface...
start "EV Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo   System Started!
echo   Dashboard will be available at: http://localhost:5173
echo   (Please wait a few seconds for servers to boot)
echo ===================================================
pause
