@echo off
title Equation Slate - Local Server
echo ===================================================
echo   Starting Equation Slate Local Development Server
echo ===================================================
echo.
echo URL: http://localhost:7860
echo.

set FLASK_ENV=development
python app.py

pause
