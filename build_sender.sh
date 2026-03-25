#!/bin/sh
# Compiles sender.c and generates a base64-encoded text file of the binary.
# Usage: ./build_sender.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/sender.c"
BIN="$SCRIPT_DIR/sender"
B64="$SCRIPT_DIR/sender_b64.txt"

echo "=== Compiling sender.c ==="
gcc -Os -s -ffunction-sections -fdata-sections -Wl,--gc-sections -o "$BIN" "$SRC" -lX11
echo "Binary: $(ls -lh "$BIN" | awk '{print $5}') -> $BIN"

echo ""
echo "=== Compressing + encoding to base64 ==="
gzip -c -9 "$BIN" | base64 -w150 > "$B64"
echo "Base64 (gzipped): $(wc -c < "$B64") bytes -> $B64"

echo ""
echo "Run this command to deploy the sender on any Linux box:"
echo "  base64 -d sender_b64.txt | gunzip > sender && chmod +x sender"

echo ""
echo "=== Done ==="
