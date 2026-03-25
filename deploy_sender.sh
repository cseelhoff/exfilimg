#!/bin/bash
# Generates a single-line command that decodes sender_b64.txt into the binary.
# Usage: ./deploy_sender.sh
#   Prints a one-liner you can copy-paste onto any Linux box.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
B64="$SCRIPT_DIR/sender_b64.txt"

if [ ! -f "$B64" ]; then
    echo "ERROR: $B64 not found. Run build_sender.sh first." >&2
    exit 1
fi

echo "echo '$(cat "$B64")' | base64 -d | gunzip > sender && chmod +x sender"
