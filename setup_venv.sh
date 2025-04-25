#!/bin/bash

VENV_DIR="venv"

# Check if venv directory exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
  fi
else
  echo "Virtual environment '$VENV_DIR' already exists."
fi

# Activate the virtual environment (for this script's context)
source "$VENV_DIR/bin/activate"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
  echo "Error: requirements.txt not found."
  deactivate > /dev/null 2>&1 || true # Attempt to deactivate before exit
  exit 1
fi

# Install/update requirements
echo "Installing/updating packages from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
  echo "Error: Failed to install requirements."
  deactivate > /dev/null 2>&1 || true # Attempt to deactivate before exit
  exit 1
fi

# Deactivate the temporary activation
deactivate > /dev/null 2>&1 || true

echo ""
echo "Setup complete. Virtual environment '$VENV_DIR' is ready."
echo "To activate it in your current shell, run:"
echo "source $VENV_DIR/bin/activate"

exit 0 