@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting AI Voice Detection server...
echo Open http://localhost:8000 in your browser
echo.
uvicorn api:app --reload --host 0.0.0.0
