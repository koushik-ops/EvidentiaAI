@echo off
title Evidentia AI — Supabase Cloud Evidence Live Watcher
cd /d "%~dp0"
echo ========================================================
echo  Starting Evidentia Realtime Cloud Watcher Daemon...
echo ========================================================
echo  Monitoring Supabase bucket: evidentia-evidence/accidents/
echo ========================================================

python -B watch_cloud_evidence.py --interval 10
pause
