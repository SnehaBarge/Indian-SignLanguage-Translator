#!/usr/bin/env bash
set -e
echo "Installing ISL-ML-Project dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Done! Run: python 1_collect_data.py --sequences 30"
