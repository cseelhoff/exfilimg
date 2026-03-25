#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install --no-index --find-links "$SCRIPT_DIR/package" opencv-python numpy
echo "Done. You can now run: python sender.py"
