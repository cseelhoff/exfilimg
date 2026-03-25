# exfilimg

Get files off an air-gapped VM when the screen is the only way out.

## The Problem

You're working in a locked-down environment — a SimSpace VM in a browser, a restricted lab, a sandboxed desktop — and you need to extract a file. There's no network path, no clipboard sharing, no USB, no file transfer. But you can *see* the screen.

## How It Works

**exfilimg** encodes files as pixel data in an X11 window. A receiver on your host machine watches the screen via screen capture, decodes the pixels back into bytes, and reassembles the original file.

1. Drop a file into the sender's watch directory
2. The sender compresses it and displays each chunk as a frame of colored pixels
3. The receiver captures and decodes each frame automatically
4. Press any key in the sender window to advance to the next frame
5. The receiver reassembles the chunks into a `.tgz` archive

Each pixel encodes multiple bits across its RGB channels (configurable 3–24 bits per pixel). A maximized window at 1920px wide with the default 3 bpp carries ~700 KB per frame.

## Files

| File | Purpose |
|------|---------|
| `sender.c` | Sender — runs inside the air-gapped VM. Watches a drop directory, compresses/splits files, displays frames via X11. |
| `receiver.py` | Receiver — runs on your host. Scans screen for frames, decodes, and reassembles into `.tgz` files. |
| `build_sender.sh` | Compiles the sender and produces a base64-encoded text file (`sender_b64.txt`) for easy transfer. |
| `fileserver.py` | Minimal HTTP file server (no dependencies) for serving/uploading files to make getting files onto the target easier. |

## Usage

### 1. Build the sender

On any Linux machine with `gcc` and `libx11-dev`:

```bash
./build_sender.sh
```

This produces the `sender` binary and `sender_b64.txt` (a base64-encoded, gzip-compressed copy you can paste into a terminal).

### 2. Deploy to the air-gapped target

If you can copy files in, transfer the `sender` binary directly. Otherwise, paste the contents of `sender_b64.txt` and decode it:

```bash
base64 -d sender_b64.txt | gunzip > sender && chmod +x sender
```

### 3. Run the sender (on the target)

```bash
mkdir -p drop
./sender              # watches ./drop/ by default
./sender /path/to/dir # or specify a different directory
./sender -b 6         # 6 bits per pixel for higher throughput
```

Maximize the window for maximum payload per frame.

### 4. Run the receiver (on your host)

```bash
pip install numpy mss
python3 receiver.py
python3 receiver.py --monitor 2 --outdir received  # options
```

### 5. Transfer a file

```bash
# On the target, drop a file into the watch directory:
cp secret_plans.pdf drop/

# The sender displays the first frame. The receiver decodes it automatically.
# Press any key in the sender window to advance to the next frame.
# Repeat until the receiver reports completion.

# On your host, extract the result:
tar xzf received_001.tgz
```

## License

MIT — see [LICENSE](LICENSE).