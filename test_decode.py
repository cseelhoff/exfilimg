#!/usr/bin/env python3
"""Test harness: validate receiver decode logic against screenshot.bmp.
Simulates a full-screen capture with the sender window embedded in it."""
import numpy as np
from PIL import Image

SYNC = np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8)
HDR = 13
BDR = 5

img_pil = Image.open("screenshot.bmp")
img_rgb = np.array(img_pil)
img_bgr = img_rgb[:, :, ::-1].copy()

print(f"Screenshot shape (BGR): {img_bgr.shape}")

# --- Test 1: find_sender row-by-row scan on raw screenshot ---
print("\n=== Test 1: find_sender row-by-row scan ===")
h, w = img_bgr.shape[:2]
found = False
for y in range(h):
    row = img_bgr[y].flatten()
    for off in range(8):
        packed = np.packbits(row[off:] // 128)
        if packed.size < HDR:
            continue
        candidates = np.where(packed[:packed.size - 3] == SYNC[0])[0]
        for i in candidates:
            if i + HDR > packed.size:
                break
            if not np.array_equal(packed[i:i+4], SYNC):
                continue
            hdr = packed[i:i+HDR]
            ppr = int(hdr[4]) + int(hdr[5]) * 256
            if ppr < 100:
                continue
            x = (off + i * 8) // 3
            print(f"  FOUND at row={y}, col={x}, off={off}, packed_idx={i}, ppr={ppr}")
            found = True
            break
        if found:
            break
    if found:
        break
if not found:
    print("  NOT FOUND")
    exit(1)

# --- Test 2: Simulate full screen with sender embedded ---
print("\n=== Test 2: decode on simulated screen capture ===")
screen_h, screen_w = 2160, 3840
screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

win_y, win_x = 200, 300
# Extract data content from screenshot and build a properly bordered sender window
data_row_start = y
data_col_start = x
data_rows = img_bgr[data_row_start:, data_col_start:data_col_start + ppr]
print(f"  Data content from screenshot: {data_rows.shape}")

padded = np.zeros((data_rows.shape[0] + 2 * BDR, ppr + 2 * BDR, 3), dtype=np.uint8)
dr = min(data_rows.shape[0], padded.shape[0] - BDR)
dc = min(data_rows.shape[1], padded.shape[1] - BDR)
padded[BDR:BDR + dr, BDR:BDR + dc] = data_rows[:dr, :dc]
print(f"  Padded sender window: {padded.shape}")

sh = min(padded.shape[0], screen_h - win_y)
sw = min(padded.shape[1], screen_w - win_x)
screen[win_y:win_y + sh, win_x:win_x + sw] = padded[:sh, :sw]

print(f"  Screen: {screen.shape}")
found2 = False
for sy in range(screen_h):
    row = screen[sy].flatten()
    for off in range(8):
        packed = np.packbits(row[off:] // 128)
        if packed.size < HDR:
            continue
        candidates = np.where(packed[:packed.size - 3] == SYNC[0])[0]
        for i in candidates:
            if i + HDR > packed.size:
                break
            if not np.array_equal(packed[i:i+4], SYNC):
                continue
            hdr = packed[i:i+HDR]
            s_ppr = int(hdr[4]) + int(hdr[5]) * 256
            if s_ppr < 100:
                continue
            s_x = (off + i * 8) // 3
            region = {"top": sy - BDR, "left": s_x - BDR,
                      "width": s_ppr + 2 * BDR, "height": screen_h - sy}
            print(f"  find_sender: ppr={s_ppr} at ({region['left']},{region['top']})")
            found2 = True
            break
        if found2:
            break
    if found2:
        break

if not found2:
    print("  find_sender FAILED on simulated screen")
    exit(1)

region_img = screen[region["top"]:region["top"] + region["height"],
                    region["left"]:region["left"] + region["width"]]
print(f"  Region captured: {region_img.shape}")

content = region_img[BDR:-BDR, BDR:-BDR]
print(f"  After border crop: {content.shape}")
packed = np.packbits(content.flatten() // 128)
if packed.size >= HDR and np.array_equal(packed[0:4], SYNC):
    hh = packed[:HDR]
    d_ppr   = int(hh[4]) + int(hh[5]) * 256
    d_total = int(hh[6]) + int(hh[7]) * 256
    d_idx   = int(hh[8]) + int(hh[9]) * 256
    d_plen  = int(hh[10]) + int(hh[11]) * 256 + int(hh[12]) * 65536
    print(f"  DECODED: ppr={d_ppr}, total={d_total}, idx={d_idx}, plen={d_plen}")
    if HDR + d_plen <= packed.size:
        payload = bytes(packed[HDR:HDR + d_plen])
        print(f"  Payload: {len(payload)} bytes, first 20: {payload[:20].hex()}")
        print(f"  SUCCESS!")
    else:
        print(f"  FAIL: payload truncated ({HDR + d_plen} > {packed.size})")
else:
    print(f"  FAIL: SYNC not found at packed[0:4]")

print("\nAll tests passed." if found and found2 else "\nSome tests failed.")
