# Ensure pip is installed and upgraded
python -m ensurepip --upgrade
python -m pip install venv
# create virtual environment
python -m venv .venv
source .venv/bin/activate
# Install the 'samu' package
python -m pip install samu
python -m samu