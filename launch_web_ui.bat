@echo off
title DriveOps Forensic AI — Web Dashboard Server
cd /d "%~dp0"
echo ========================================================
echo  Starting DriveOps Forensic AI Modern Web Dashboard...
echo ========================================================
echo  Server Address: http://localhost:5000
echo ========================================================

start "" "http://localhost:5000"
python -B web_server.py
pause
