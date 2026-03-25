#!/bin/bash
# Downloads Python wheels for an air-gapped Linux target and packages everything into a tar.gz
# Usage: ./package_deps.sh 3.11

PYTHON_VERSION="${1:-3.11}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/package"
BUNDLE_NAME="exfilimg_bundle"

echo "=== Downloading wheels for Python ${PYTHON_VERSION} (linux x86_64) ==="

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

pip download \
    opencv-python numpy \
    --platform manylinux2014_x86_64 \
    --python-version "$PYTHON_VERSION" \
    --only-binary=:all: \
    -d "$PACKAGE_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: pip download failed"
    exit 1
fi

echo ""
echo "=== Downloaded wheels ==="
ls -lh "$PACKAGE_DIR"

echo ""
echo "=== Packaging bundle ==="
rm -f "$SCRIPT_DIR/${BUNDLE_NAME}.tar.gz"
tar -czf "$SCRIPT_DIR/${BUNDLE_NAME}.tar.gz" \
    -C "$SCRIPT_DIR" \
    package/ \
    install.sh \
    sender.py
    # receiver.py - not needed on the target, but we can include it for reference

echo ""
echo "=== Done ==="
echo "Bundle: ${BUNDLE_NAME}.tar.gz ($(du -h "$SCRIPT_DIR/${BUNDLE_NAME}.tar.gz" | cut -f1))"
echo ""
echo "On the Kali VM:"
echo "  tar xzf ${BUNDLE_NAME}.tar.gz"
echo "  cd package && bash install.sh"
