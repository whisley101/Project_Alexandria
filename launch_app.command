#!/bin/bash

set -euo pipefail

PROJECT_DIR="/Users/nathanrandall/Documents/Project_Alexandria"
APP_URL="http://localhost:8501"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found at $PROJECT_DIR/.venv"
  echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source "$PROJECT_DIR/.venv/bin/activate"

open "$APP_URL" || true
exec streamlit run app.py
