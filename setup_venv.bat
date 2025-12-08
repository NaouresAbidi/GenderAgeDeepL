@echo off
echo 🐍 Setting up Virtual Environment for Age & Gender Prediction
echo ============================================================

REM Check if .venv already exists
if exist .venv (
    echo ✅ Virtual environment already exists
    echo 🔄 Activating existing environment...
    call .venv\Scripts\activate.bat
    goto :install_packages
)

echo 📁 Creating virtual environment...
python -m venv .venv

echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

:install_packages
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

echo 📦 Installing TensorFlow (compatible version)...
pip install tensorflow>=2.13.0

echo 📦 Installing other requirements...
pip install numpy pandas seaborn matplotlib scikit-learn opencv-python flask pillow tqdm

echo 🔍 Checking installation...
python check_prerequisites.py

echo.
echo ✅ Setup complete!
echo =================
echo.
echo 🎯 To use this environment:
echo    1. Run: .venv\Scripts\activate.bat
echo    2. Then run your project commands
echo.
echo 🚀 Quick start:
echo    python src\api\api.py