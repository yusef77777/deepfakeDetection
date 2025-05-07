@echo off
REM Activate Miniconda
call "C:\Users\creat\miniconda3\Scripts\activate.bat"

REM Activate virtual environment
call conda activate my_env

REM Navigate to project directory
cd "C:\Users\creat\Desktop\semesters\7th semester\deepfake_fyp1"

REM Run Django server
start /min cmd /k "python manage.py runserver"

REM Wait a few seconds to ensure the server starts
timeout /t 5 /nobreak >nul

REM Open Django server in Microsoft Edge
start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "http://127.0.0.1:8000/"

pause
