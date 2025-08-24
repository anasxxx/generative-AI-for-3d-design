@echo off
echo Starting Fashion 3D Demo...
cd /d "%~dp0"
if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
    conda activate fashion3d
    python deploy.py --demo
) else (
    python deploy.py --demo
)
pause
