# Ensure pip is installed and upgraded
python -m ensurepip --upgrade

python -m pip install venv

# create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
# Install the 'sam' package
python -m pip install sam
python -m sam