@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [garbage] training gar.py ...
python -m pip install -q -r requirements.txt
python gar.py
pause
