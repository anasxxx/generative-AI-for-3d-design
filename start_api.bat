@echo off
echo Starting Fashion 3D API Server...
cd /d "%~dp0"
if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
    conda activate fashion3d
    python api/server.py
) else (
    python api/server.py
)
pause
