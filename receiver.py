#!/usr/bin/env python3
"""Receiver for visual data exfiltration. Captures frames from the sender's
X11 window via mss, saves .tgz chunks, reassembles, and extracts.

The sender displays frames one at a time. After capturing each frame, the
operator presses any key in the sender's VM window to advance to the next.

Usage: python3 receiver.py [--monitor N] [--outdir DIR]
"""
import os, sys, time, shutil, argparse, zlib, numpy, mss

# Header: 4 sync + 2 ppr + 1 bpp + 2 total + 2 idx + 3 len + 4 crc32 = 18 bytes
SYNC = numpy.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=numpy.uint8)
HDR = 18; BDR = 5
HDR_PIXELS = HDR * 8 // 3   # 48 pixels: header always encoded at 3 bpp
BPP_VALUES = [3, 6, 9, 12, 15, 18, 21, 24]
_debug = False

def _cleanup_chunks(chunk_dir):
    """Remove temporary chunk directory."""
    if chunk_dir and os.path.isdir(chunk_dir):
        shutil.rmtree(chunk_dir, ignore_errors=True)

def grab_bgr(sct, mon):
    return numpy.delete(numpy.array(sct.grab(mon)), numpy.s_[3], axis=2)

def pixels_to_bytes(img, bpp):
    """Convert pixel data to a packed byte stream at the given bits-per-pixel."""
    bpc = bpp // 3
    levels = 1 << bpc
    q = numpy.round(img.astype(numpy.float32) * (levels - 1) / 255.0).astype(numpy.uint8)
    flat = q.reshape(-1, 3)
    bits = numpy.zeros((flat.shape[0], 3, bpc), dtype=numpy.uint8)
    for b in range(bpc):
        bits[:, :, b] = (flat >> (bpc - 1 - b)) & 1
    return numpy.packbits(bits.reshape(-1))

def decode(img):
    """Try to decode a frame. Returns (ppr, bpp, chunk_idx, total, payload) or None."""
    if img.shape[0] <= 2 * BDR or img.shape[1] <= 2 * BDR:
        if _debug: print(f"  [dbg] image too small: {img.shape}")
        return None
    content = img[BDR:-BDR, BDR:-BDR]

    flat = content.reshape(-1, 3)
    if flat.shape[0] < HDR_PIXELS:
        if _debug: print(f"  [dbg] not enough pixels: {flat.shape[0]} < {HDR_PIXELS}")
        return None

    # Decode header at fixed 3 bpp
    hdr_packed = pixels_to_bytes(flat[:HDR_PIXELS].reshape(1, -1, 3), 3)
    if hdr_packed.size < HDR:
        if _debug: print(f"  [dbg] hdr_packed too small: {hdr_packed.size}")
        return None
    if not numpy.array_equal(hdr_packed[0:4], SYNC):
        if _debug: print(f"  [dbg] SYNC mismatch: got {hdr_packed[:4].tolist()} expect {SYNC.tolist()}")
        return None

    h = hdr_packed[:HDR]
    ppr   = int(h[4]) + int(h[5]) * 256
    hdr_bpp = int(h[6])
    total = int(h[7]) + int(h[8]) * 256
    idx   = int(h[9]) + int(h[10]) * 256
    plen  = int(h[11]) + int(h[12]) * 256 + int(h[13]) * 65536
    expected_crc = int(h[14]) | (int(h[15]) << 8) | (int(h[16]) << 16) | (int(h[17]) << 24)
    if _debug: print(f"  [dbg] hdr: ppr={ppr} bpp={hdr_bpp} total={total} idx={idx} plen={plen} crc={expected_crc:#010x}")
    if ppr < 100 or total < 1 or hdr_bpp not in BPP_VALUES:
        if _debug: print(f"  [dbg] bad header values")
        return None

    # Decode payload at the bpp indicated in header
    payload_pixels = flat[HDR_PIXELS:]
    if payload_pixels.shape[0] == 0:
        if _debug: print(f"  [dbg] no payload pixels")
        return None
    payload_packed = pixels_to_bytes(payload_pixels.reshape(1, -1, 3), hdr_bpp)
    if _debug: print(f"  [dbg] payload_packed.size={payload_packed.size} need={plen}")
    if plen > payload_packed.size:
        if _debug: print(f"  [dbg] payload too short")
        return None
    payload = bytes(payload_packed[:plen])
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        if _debug:
            print(f"  [dbg] CRC mismatch: got {actual_crc:#010x} expect {expected_crc:#010x}")
            # Pixel quality analysis: check how captured values deviate from ideal levels
            bpc = hdr_bpp // 3
            levels = 1 << bpc
            ideal = numpy.array([v * 255 // (levels - 1) for v in range(levels)])
            sample = payload_pixels[:min(1000, payload_pixels.shape[0])].ravel()
            diffs = numpy.abs(sample.astype(int)[:, None] - ideal[None, :]).min(axis=1)
            print(f"  [dbg] pixel quality (first 1000 px): "
                  f"mean_err={diffs.mean():.1f} max_err={diffs.max()} "
                  f">0:{(diffs>0).sum()}/{diffs.size}")
        return None
    return ppr, hdr_bpp, idx, total, payload

def find_sender(sct, midx, scan_limit=0):
    """Full-screen scan to locate the sender and return a tight capture region."""
    mon = sct.monitors[midx]
    print(f"Scanning {mon['width']}x{mon['height']}...")
    sync_bits = numpy.unpackbits(SYNC) * 2 - 1  # ±1 for correlation
    while True:
        img = grab_bgr(sct, mon)
        rows, w = img.shape[:2]
        for y in range(rows):
            row = img[y]  # (W, 3)
            if row.max() == 0:
                continue
            # At 3 bpp each channel is 1 bit; threshold at 128
            bitstream = (row >= 128).astype(numpy.uint8).reshape(-1)
            if scan_limit > 0:
                bitstream = bitstream[:scan_limit * 8]
            if bitstream.size < 32:
                continue
            # Vectorised SYNC search across the entire row at once
            corr = numpy.correlate(bitstream * 2 - 1, sync_bits, mode="valid")
            matches = numpy.where(corr == 32)[0]
            for bit_pos in matches:
                if bit_pos % 3 != 0:
                    continue
                if bit_pos + HDR * 8 > bitstream.size:
                    continue
                ha = numpy.packbits(bitstream[bit_pos:bit_pos + HDR * 8])
                ppr = int(ha[4]) + int(ha[5]) * 256
                hdr_bpp = int(ha[6])
                if hdr_bpp not in BPP_VALUES or ppr < 100:
                    continue
                x = bit_pos // 3
                region = {"top": mon["top"] + y - BDR,
                          "left": mon["left"] + x - BDR,
                          "width": ppr + 2 * BDR,
                          "height": mon["height"] - y}
                print(f"  Sender found: ppr={ppr}, bpp={hdr_bpp} at ({region['left']},{region['top']})")
                return region
        time.sleep(0.2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--outdir", default=".")
    p.add_argument("--scan-limit", type=int, default=0,
                   help="Max bytes to search per row during scan (0 = unlimited)")
    p.add_argument("--debug", action="store_true",
                   help="Print decode diagnostics")
    a = p.parse_args()
    global _debug
    _debug = a.debug
    os.makedirs(a.outdir, exist_ok=True)

    with mss.mss() as sct:
        region = find_sender(sct, a.monitor, a.scan_limit)
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

            ppr, bpp_v, idx, tot, payload = result

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
                print(f"  [{idx+1}/{tot}] {len(payload)} bytes  {bpp_v}bpp  CRC OK")

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
