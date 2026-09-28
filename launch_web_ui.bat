@echo off
title Evidentia Forensic AI — Web Dashboard Server
cd /d "%~dp0"
echo ========================================================
echo  Starting Evidentia Forensic AI Modern Web Dashboard...
echo ========================================================
echo  Server Address: http://localhost:5000
echo ========================================================

start "" "http://localhost:5000"
python -B web_server.py
pause
