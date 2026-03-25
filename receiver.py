#!/usr/bin/env python3
"""Receiver for visual data exfiltration. Captures frames from the sender's
X11 window via mss, saves .tgz chunks, reassembles, and extracts.

The sender displays frames one at a time. After capturing each frame, the
operator presses any key in the sender's VM window to advance to the next.

Usage: python3 receiver.py [--monitor N] [--outdir DIR]
"""
import os, sys, time, shutil, argparse, zlib, numpy, mss

# Header: 4 sync + 2 ppr + 2 total + 2 idx + 3 len + 4 crc32 = 17 bytes
SYNC = numpy.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=numpy.uint8)
HDR = 17; BDR = 5

def _cleanup_chunks(chunk_dir):
    """Remove temporary chunk directory."""
    if chunk_dir and os.path.isdir(chunk_dir):
        shutil.rmtree(chunk_dir, ignore_errors=True)

def grab_bgr(sct, mon):
    return numpy.delete(numpy.array(sct.grab(mon)), numpy.s_[3], axis=2)

def decode(img):
    """Try to decode a frame. Returns (ppr, chunk_idx, total, payload) or None."""
    # Strip the border so rows are exactly ppr-wide and data is contiguous
    if img.shape[0] <= 2 * BDR or img.shape[1] <= 2 * BDR:
        return None
    content = img[BDR:-BDR, BDR:-BDR]
    packed = numpy.packbits(content.flatten() // 128)
    if packed.size < HDR:
        return None
    if not numpy.array_equal(packed[0:4], SYNC):
        return None
    h = packed[:HDR]
    ppr   = int(h[4]) + int(h[5]) * 256
    total = int(h[6]) + int(h[7]) * 256
    idx   = int(h[8]) + int(h[9]) * 256
    plen  = int(h[10]) + int(h[11]) * 256 + int(h[12]) * 65536
    expected_crc = int(h[13]) | (int(h[14]) << 8) | (int(h[15]) << 16) | (int(h[16]) << 24)
    if ppr < 100 or total < 1:
        return None
    if HDR + plen > packed.size:
        return None
    payload = bytes(packed[HDR:HDR + plen])
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        return None  # CRC mismatch — corrupted frame
    return ppr, idx, total, payload

def find_sender(sct, midx):
    """Full-screen scan to locate the sender and return a tight capture region."""
    mon = sct.monitors[midx]
    print(f"Scanning {mon['width']}x{mon['height']}...")
    while True:
        img = grab_bgr(sct, mon)
        h, w = img.shape[:2]
        # Scan each row independently — the SYNC pattern fits within one row
        for y in range(h):
            row = img[y].flatten()  # w*3 BGR channel values
            for off in range(8):
                packed = numpy.packbits(row[off:] // 128)
                if packed.size < HDR:
                    continue
                # Fast filter: only check positions where first SYNC byte appears
                candidates = numpy.where(packed[:packed.size - 3] == SYNC[0])[0]
                for i in candidates:
                    if i + HDR > packed.size:
                        break
                    if not numpy.array_equal(packed[i:i+4], SYNC):
                        continue
                    hdr = packed[i:i+HDR]
                    ppr = int(hdr[4]) + int(hdr[5]) * 256
                    if ppr < 100:
                        continue
                    x = (off + i * 8) // 3
                    region = {"top": mon["top"] + y - BDR,
                              "left": mon["left"] + x - BDR,
                              "width": ppr + 2 * BDR,
                              "height": mon["height"] - y}
                    print(f"  Sender found: ppr={ppr} at ({region['left']},{region['top']})")
                    return region
        time.sleep(0.2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    with mss.mss() as sct:
        region = find_sender(sct, a.monitor)
        last_idx = -1
        received = set()  # chunk indices saved to disk
        total = None
        file_num = 0
        completed = False  # True while waiting for sender to advance after completion
        chunk_dir = None   # temp directory for current transfer's chunks

        print("Capturing frames. Press any key in sender VM to advance.\n")

        while True:
            img = grab_bgr(sct, region)
            result = decode(img)

            if result is None:
                # Idle screen (black) = sender has no more files
                if received:
                    file_num += 1
                    tgz = os.path.join(a.outdir, f"received_{file_num:03d}.tgz")
                    with open(tgz, "wb") as out:
                        for i in range(total):
                            cp = os.path.join(chunk_dir, f"chunk_{i:04d}.bin")
                            if os.path.exists(cp):
                                with open(cp, "rb") as cf:
                                    out.write(cf.read())
                    _cleanup_chunks(chunk_dir)
                    print(f"\nSaved {tgz} ({os.path.getsize(tgz)} bytes)")
                    print(f"Extract with: tar xzf {tgz}")
                    received = set(); total = None; last_idx = -1; chunk_dir = None
                completed = False
                time.sleep(0.5)
                continue

            ppr, idx, tot, payload = result

            # Still showing the same completed frame — wait silently
            if completed:
                time.sleep(0.5)
                continue

            if total is None:
                total = tot
                chunk_dir = os.path.join(a.outdir, f".transfer_{file_num + 1:03d}")
                os.makedirs(chunk_dir, exist_ok=True)

            if idx not in received:
                # Write chunk to disk immediately
                cp = os.path.join(chunk_dir, f"chunk_{idx:04d}.bin")
                with open(cp, "wb") as cf:
                    cf.write(payload)
                received.add(idx)
                last_idx = idx
                print(f"  [{idx+1}/{tot}] {len(payload)} bytes  CRC OK")

                if len(received) == total:
                    file_num += 1
                    tgz = os.path.join(a.outdir, f"received_{file_num:03d}.tgz")
                    with open(tgz, "wb") as out:
                        for i in range(total):
                            cp = os.path.join(chunk_dir, f"chunk_{i:04d}.bin")
                            with open(cp, "rb") as cf:
                                out.write(cf.read())
                    _cleanup_chunks(chunk_dir)
                    print(f"\nComplete! Saved {tgz} ({os.path.getsize(tgz)} bytes)")
                    print(f"Extract with: tar xzf {tgz}")
                    print("ACK the last frame in sender to finish.\n")
                    received = set(); total = None; last_idx = -1; chunk_dir = None
                    completed = True

            time.sleep(0.1)

if __name__ == "__main__":
    main()
