# exfilimg

Optical data exfiltration over screen pixels. Encodes arbitrary files as images displayed in an X11 window, captured by a receiver via screen capture on the host. Designed for air-gapped environments (e.g. SimSpace VMs in a browser) where no network, clipboard, or file-sharing channel is available — the screen is the only data path out.

## How It Works

```
┌─────────────────────────────┐          screen capture          ┌──────────────────────┐
│  Sender (inside air-gapped  │  ──── pixel-encoded frames ────▶ │  Receiver (host)     │
│  VM, X11 window)            │                                  │  mss screen capture  │
│                             │  ◀─── keypress ACK ───────────── │  operator keyboard   │
│  ./drop/ → tar+gz → split   │                                  │  → reassemble .tgz   │
│  → encode → display frame   │                                  │  → tar xzf           │
└─────────────────────────────┘                                  └──────────────────────┘
```

1. Drop a file into the sender's `./drop/` directory
2. Sender compresses it (`tar czf`), splits into frame-sized chunks, and displays the first chunk as a pixel-encoded image
3. Receiver captures the screen, decodes the bitstream, and stores the chunk
4. Operator presses any key in the sender VM to advance to the next frame
5. Repeat until all chunks are transferred
6. Receiver reassembles into a `.tgz` archive; sender deletes the source file

Each pixel encodes 3 bits (one per color channel). A 1900-px-wide frame carries ~700KB per image. The sender auto-detects screen resolution so no hardcoded dimensions are needed.

## Frame Protocol

Frames use a 13-byte header followed by raw payload bytes:

| Offset | Size | Field |
|--------|------|-------|
| 0–3 | 4 | CCSDS sync marker (`0x1ACFFC1D`) |
| 4–5 | 2 | pixels per row (LE, auto-detected) |
| 6–7 | 2 | total chunks (LE) |
| 8–9 | 2 | chunk index (LE) |
| 10–12 | 3 | payload length (LE) |

The [CCSDS Attached Sync Marker](https://public.ccsds.org/Pubs/131x0b5.pdf) is a standard frame synchronization pattern from space/satellite communications, chosen for its optimal autocorrelation properties — it minimizes false-positive sync detection when scanning a noisy bitstream.

Integrity is provided by gzip's built-in CRC-32, so no application-level checksum is needed.

## Files

| File | Role | Dependencies |
|------|------|-------------|
| `sender.c` | **Sender (C)** — recommended for air-gapped targets. Watches a drop directory, compresses/splits files, displays frames via X11, waits for keystroke ACK. | `libX11`, `tar`, `gzip`, `split` (all preinstalled on Linux) |
| `sender.py` | **Sender (Python)** — alternative if C compilation isn't possible. Same protocol, uses feh/display for rendering. | `numpy`, `feh` or `display` |
| `receiver.py` | **Receiver (Python)** — runs on the host. Scans screen for frames, decodes, reassembles. | `numpy`, `mss` |
| `package_deps.sh` | Downloads Python wheels for offline install on air-gapped targets | `pip` |
| `install.sh` | Installs pre-downloaded wheels on the target | `pip` |

## Quick Start

### C Sender (recommended)

On the air-gapped VM (or cross-compile and transfer the binary):

```bash
gcc -O2 -o sender sender.c -lX11
./sender              # watches ./drop/ by default
./sender /path/to/dir # custom watch directory
```

If `gcc`/`libx11-dev` aren't installed:
```bash
sudo apt install gcc libx11-dev
```

### Python Sender (alternative)

```bash
pip install numpy
sudo apt install feh
python3 sender.py             # watches ./drop/
python3 sender.py /some/dir   # custom watch directory
```

### Receiver (on host machine)

```bash
pip install numpy mss
python3 receiver.py                    # default: monitor 1, output to .
python3 receiver.py --outdir received  # save to a subdirectory
python3 receiver.py --monitor 2        # use a different monitor
```

After transfer completes:
```bash
tar xzf received.tgz
```

## Transferring to an Air-Gapped Environment

### Option A: Compile on target
Copy `sender.c` to the VM, compile with `gcc -O2 -o sender sender.c -lX11`, run.

### Option B: Python with pre-packaged wheels
On an internet-connected machine:
```bash
bash package_deps.sh 3.11    # download wheels for target Python version
```

Transfer `exfilimg_bundle.tar.gz` to the VM, then:
```bash
tar xzf exfilimg_bundle.tar.gz
bash install.sh
python3 sender.py
```

## Dev Container

A VS Code dev container definition is included in `.devcontainer/` for building and testing the C sender in a Linux environment with `gcc` and `libx11-dev`.

## License

MIT — see [LICENSE](LICENSE).