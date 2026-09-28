@echo off
title Evidentia Forensic AI — Launch All Services
cd /d "%~dp0"
echo ========================================================
echo  Launching All Evidentia Forensic AI Services...
echo ========================================================

echo [1/2] Starting Cloud Evidence Watcher...
start "Cloud Watcher" cmd /c launch_cloud_watcher.bat

echo [2/2] Starting Web Dashboard Server...
start "Web Dashboard" cmd /c python -B web_server.py

echo Waiting for Web Server to start...
timeout /t 5 /nobreak >nul

echo Opening browser...
start "" "http://localhost:5000"

echo ========================================================
echo  All services launched in separate windows!
echo ========================================================
pause
