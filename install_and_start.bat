@echo off
echo ====================================================
echo FASHION 2D-to-3D GAN - AUTOMATED SETUP
echo ====================================================
echo.
echo Installing required packages...
echo This may take a few minutes...
echo.

REM Install essential packages
pip install tensorflow
pip install fastapi uvicorn
pip install pyyaml opencv-python pillow numpy
pip install requests tqdm

echo.
echo ====================================================
echo Installation completed!
echo Starting API server...
echo ====================================================
echo.

REM Navigate to project directory
cd /d "C:\Users\mahmo\OneDrive\Desktop\fashion-2d-3d-gan"

REM Start the API
python deploy.py --api

pause
