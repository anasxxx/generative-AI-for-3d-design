@echo off
echo Starting Fashion 3D Training...
cd /d "%~dp0"
if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
    conda activate fashion3d
    python scripts/train.py --hours 8
) else (
    python scripts/train.py --hours 8
)
pause
