@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python program\app.py
echo.
echo Server has exited. Press any key to close this window.
pause >nul
