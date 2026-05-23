#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 was not found."
  echo "Install Python 3.10 first, then run this script again."
  exit 1
fi

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "Mac setup complete."
echo "Run: source .venv/bin/activate"
echo "Then: python app.py --config configs/mac_mini.yaml"

