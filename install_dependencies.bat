@echo off
echo ============================================================
echo FASHION 2D-to-3D GAN - DEPENDENCY INSTALLER
echo ============================================================

echo.
echo [INFO] Installing core dependencies...
echo.

REM Install TensorFlow
echo [STEP] Installing TensorFlow...
conda install -c conda-forge tensorflow=2.13.0 -y
if %errorlevel% neq 0 (
    echo [WARNING] TensorFlow installation had issues, trying alternative...
    pip install tensorflow==2.13.0
)

REM Install essential packages
echo [STEP] Installing essential packages...
conda install -c conda-forge tqdm numpy scipy matplotlib pillow -y

echo [STEP] Installing image processing...
conda install -c conda-forge opencv scikit-image -y

echo [STEP] Installing utilities...
conda install -c conda-forge h5py pyyaml -y

echo [STEP] Installing API dependencies...
conda install -c conda-forge fastapi uvicorn python-multipart -y

echo [STEP] Installing additional pip packages...
pip install rich typer plotly

echo.
echo [INFO] Testing installation...
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import tqdm; print('tqdm: OK')"
python -c "import numpy as np; print(f'numpy: {np.__version__}')"

echo.
echo ============================================================
echo INSTALLATION COMPLETE
echo ============================================================
echo.
echo Next steps:
echo 1. Run: python deploy.py --test
echo 2. Run: python deploy.py --analyze
echo 3. Run: python deploy.py --demo
echo.
pause
