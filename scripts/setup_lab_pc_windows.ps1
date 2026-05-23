$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

py -3.10 --version
py -3.10 -m venv .venv

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

Write-Host "Lab PC setup complete."
Write-Host "Run: .\.venv\Scripts\Activate.ps1"
Write-Host "Then: python app.py --config configs/lab_pc.yaml"

