#!/usr/bin/env python3
"""Python sender — watches a drop directory, compresses files with tar+gzip,
splits into chunks, and displays each as a PPM frame via feh. Operator presses
any key in the terminal to advance to the next frame.

Usage: python3 sender.py [watch_dir]   # default: ./drop

Requires: numpy, feh (or display/eog)
"""
import sys, os, time, glob, shutil, subprocess, tempfile, zlib, numpy

BORDER = 5
HEADER_SIZE = 17  # 4 sync + 2 ppr + 2 total + 2 idx + 3 len + 4 crc32
SYNC = numpy.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=numpy.uint8)

def get_screen_width():
    """Auto-detect screen width via xdpyinfo or xrandr, fallback to 1900."""
    for cmd, parse in [
        (["xdpyinfo"], lambda o: int([l for l in o.split('\n') if 'dimensions' in l][0].split()[1].split('x')[0])),
        (["xrandr"], lambda o: max(int(l.split()[0].split('x')[0]) for l in o.split('\n') if '*' in l)),
    ]:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            return parse(out) - 2 * BORDER - 40
        except Exception:
            continue
    return 1900

def write_ppm(path, rgb):
    h, w = rgb.shape[:2]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())
    os.replace(tmp, path)

def encode_frame(payload, idx, total, ppr):
    data = numpy.frombuffer(payload, dtype=numpy.uint8)
    n = len(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    hdr = numpy.concatenate((SYNC, numpy.array([
        ppr & 0xFF, (ppr >> 8) & 0xFF,
        total & 0xFF, (total >> 8) & 0xFF,
        idx & 0xFF, (idx >> 8) & 0xFF,
        n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF,
        crc & 0xFF, (crc >> 8) & 0xFF, (crc >> 16) & 0xFF, (crc >> 24) & 0xFF,
    ], dtype=numpy.uint8)))
    bits = numpy.unpackbits(numpy.concatenate((hdr, data))) * 255
    bpr = ppr * 3
    pad = (bpr - len(bits) % bpr) % bpr
    bits = numpy.concatenate((bits, numpy.zeros(pad, dtype=numpy.uint8)))
    rgb = bits.reshape(-1, 3)[:, ::-1].copy().reshape(-1, ppr, 3)
    return numpy.pad(rgb, ((BORDER, BORDER), (BORDER, BORDER), (0, 0)), constant_values=0)

def find_viewer():
    for cmd in ["feh", "display", "eog"]:
        if shutil.which(cmd): return cmd
    return None

def compress_and_split(filepath, max_payload, tmpdir):
    """tar+gzip a file and split into chunks. Returns sorted list of chunk paths."""
    dirname, basename = os.path.dirname(filepath) or ".", os.path.basename(filepath)
    cmd = f"tar czf - -C '{dirname}' '{basename}' | split -b {max_payload} - '{tmpdir}/chunk.'"
    if os.system(cmd) != 0: return []
    return sorted(glob.glob(os.path.join(tmpdir, "chunk.*")))

def main():
    watchdir = sys.argv[1] if len(sys.argv) > 1 else "./drop"
    os.makedirs(watchdir, exist_ok=True)

    viewer = find_viewer()
    if not viewer:
        print("No image viewer found. Install: sudo apt install feh"); sys.exit(1)

    ppr = get_screen_width()
    max_rows = 1050
    max_payload = (ppr * 3 * max_rows) // 8 - HEADER_SIZE
    print(f"ppr={ppr}, max_payload={max_payload} bytes/frame")
    print(f"Watching '{watchdir}'. Press Enter after each frame. Ctrl+C to exit.\n")

    ppm_path = os.path.join(tempfile.gettempdir(), "sender.ppm")
    viewer_proc = None

    try:
        while True:
            # Find next file
            files = [os.path.join(watchdir, f) for f in os.listdir(watchdir)
                     if os.path.isfile(os.path.join(watchdir, f)) and not f.startswith('.')]
            if not files:
                time.sleep(0.5); continue

            filepath = files[0]
            print(f"Found: {filepath}")

            tmpdir = tempfile.mkdtemp(prefix="s3b_")
            chunks = compress_and_split(filepath, max_payload, tmpdir)
            if not chunks:
                print("  Compress/split failed, skipping."); shutil.rmtree(tmpdir); continue

            total = len(chunks)
            print(f"  {total} chunk(s)")

            for i, cp in enumerate(chunks):
                with open(cp, "rb") as f: payload = f.read()
                frame = encode_frame(payload, i, total, ppr)
                write_ppm(ppm_path, frame)

                if viewer_proc is None:
                    if viewer == "feh":
                        viewer_proc = subprocess.Popen(
                            ["feh", "-R", "0.1", "--force-aliasing", ppm_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        viewer_proc = subprocess.Popen(
                            [viewer, ppm_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                print(f"  Frame {i+1}/{total} — press Enter to advance")
                input()

            shutil.rmtree(tmpdir)
            os.unlink(filepath)
            print(f"  Done, deleted '{filepath}'.\n")

            # Show black (idle)
            idle = numpy.zeros((max_rows + 2*BORDER, ppr + 2*BORDER, 3), dtype=numpy.uint8)
            write_ppm(ppm_path, idle)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        if viewer_proc: viewer_proc.terminate()
        for p in [ppm_path, ppm_path + ".tmp"]:
            if os.path.exists(p): os.unlink(p)

if __name__ == "__main__":
    main()
