#!/bin/bash
cd "$(dirname "$0")" || exit 1
if ! command -v python3.11 >/dev/null 2>&1; then
  echo 'Python 3.11 is required. Install it from https://www.python.org/downloads/release/python-3119/ or use Homebrew: brew install python@3.11'
  read -p 'Press Enter to close...'
  exit 1
fi
if [ ! -d .venv ]; then
  python3.11 -m venv .venv || exit 1
fi
source .venv/bin/activate
python -m pip install -r requirements.txt || { echo 'Dependency installation failed'; read -p 'Press Enter to close...'; exit 1; }
MODEL=models/face_landmarker.task
if [ ! -f "$MODEL" ]; then
  echo 'Downloading face model (about 4 MB)...'
  mkdir -p models
  curl -fL -o "$MODEL" https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
    || { rm -f "$MODEL"; echo 'Face model download failed'; read -p 'Press Enter to close...'; exit 1; }
fi
python app.py
read -p 'Press Enter to close...'
