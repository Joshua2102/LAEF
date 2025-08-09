@echo off
cd /d "C:\Users\17866\OneDrive\AlpacaBot\CODE"
call "Laefenv\Scripts\activate.bat"
echo.
echo ====================================
echo    LAEF Trading System Activated
echo ====================================
echo Virtual Environment: Laefenv
echo Working Directory: %CD%
echo.
echo Available Commands:
echo   python main.py          - Start LAEF main menu
echo   python -m pytest tests  - Run tests
echo   deactivate              - Exit virtual environment
echo.
cmd /k