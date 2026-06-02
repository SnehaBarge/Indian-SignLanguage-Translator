@echo off
echo Installing ISL-ML-Project dependencies...
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Done! Run: python 1_collect_data.py --sequences 30
pause
