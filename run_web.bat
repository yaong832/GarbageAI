@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [garbage] pip install...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo pip failed. Check Python and PATH.
  pause
  exit /b 1
)

echo.
echo [garbage] Flask server 0.0.0.0:5000
echo   Browser: http://127.0.0.1:5000  ^(opened by app.py^)
echo   Port 5000: do not run farmui or etchflask at the same time.
echo.

python app.py
pause
