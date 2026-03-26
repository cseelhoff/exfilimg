/*
 * sender.c — Visual data exfiltration sender.
 *
 * Watches a drop directory for files, compresses/splits them with tar+gzip+split,
 * and displays each chunk as a pixel-encoded frame via X11. The receiver sends a
 * single keystroke ACK (any key) to advance to the next frame. After all chunks of
 * a file are confirmed, the source file is deleted and the sender watches for more.
 *
 * Window is freely resizable/maximizable. When a new file appears in the
 * drop directory, the current window size is queried and used to compute
 * the frame capacity and split size. Frame header includes width so the
 * receiver can decode without hardcoded constants.
 *
 * Build:
 *   gcc -O2 -o sender sender.c -lX11
 *
 * Usage:
 *   ./sender [-b bpp] [watch_dir] # default: 3 bpp, ./drop
 *
 * Frame header (18 bytes):
 *   [0..3]   CCSDS sync: 0x1A 0xCF 0xFC 0x1D
 *   [4..5]   pixels_per_row (LE) — auto-detected
 *   [6]      bits_per_pixel (3,6,9,...,24)
 *   [7..8]   total_chunks (LE)
 *   [9..10]  chunk_index (LE)
 *   [11..13] payload_length (LE, 3 bytes)
 *   [14..17] CRC32 of payload (LE)
 *
 * Back channel: any keypress in X11 window = ACK (advance to next frame).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#define BORDER            5     /* pixel border around encoded content */
#define HEADER_SIZE       18    /* 4 sync + 2 ppr + 1 bpp + 2 total + 2 idx + 3 len + 4 crc */
#define HDR_PIXEL_COUNT   (HEADER_SIZE * 8 / 3) /* 48 pixels: header always encoded at 3 bpp */
#define INIT_WIDTH        800   /* initial window width; user can resize/maximize */
#define INIT_HEIGHT       600   /* initial window height */

/* CCSDS sync marker — a standard frame synchronization pattern from satellite
   communications, chosen for its optimal autocorrelation properties (minimizes
   false-positive sync detection when scanning a noisy pixel stream). */
static const uint8_t SYNC_MARKER[4] = { 0x1A, 0xCF, 0xFC, 0x1D };

/* Bits per pixel: each pixel encodes this many bits across its 3 RGB channels.
   Must be a multiple of 3 (valid: 3, 6, 9, ... 24). Higher = more data per
   frame but requires more accurate color reproduction by the display. */
static int bits_per_pixel = 3;

/* X11 display resources */
static Display *display;
static Window   window;
static Visual  *visual;
static GC       graphics_ctx;
static int      screen_num, color_depth;
static Atom     wm_delete_atom;       /* WM_DELETE_WINDOW protocol atom */

/* Frame geometry (recomputed on window resize) */
static int      pixels_per_row;       /* usable pixel columns (width minus borders) */
static int      max_rows;             /* usable pixel rows (height minus borders) */
static int      win_width, win_height;
static size_t   max_payload;          /* max payload bytes that fit in one frame */

static volatile sig_atomic_t quit = 0;
static void on_signal(int sig) { (void)sig; quit = 1; }

/* ── CRC32 (same polynomial as zlib) ─────────────────────────────────── */

static uint32_t crc32_compute(const uint8_t *data, size_t length)
{
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}

/* ── helpers ─────────────────────────────────────────────────────────── */

static void msleep(int milliseconds)
{
    struct timespec ts = { milliseconds / 1000, (milliseconds % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/* Read an entire file into a malloc'd buffer. Caller must free(). */
static uint8_t *load_file(const char *path, size_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    if (file_size <= 0) { fclose(fp); return NULL; }
    rewind(fp);
    uint8_t *buf = malloc((size_t)file_size);
    if (!buf || (long)fread(buf, 1, (size_t)file_size, fp) != file_size) {
        free(buf); fclose(fp); return NULL;
    }
    fclose(fp);
    *out_len = (size_t)file_size;
    return buf;
}

/* Little-endian serialization helpers */
static void put_le16(uint8_t *dst, uint16_t v) { dst[0] = v; dst[1] = v >> 8; }
static void put_le24(uint8_t *dst, uint32_t v) { dst[0] = v; dst[1] = v >> 8; dst[2] = v >> 16; }
static void put_le32(uint8_t *dst, uint32_t v) { dst[0] = v; dst[1] = v >> 8; dst[2] = v >> 16; dst[3] = v >> 24; }

/* ── X11 setup ───────────────────────────────────────────────────────── */

/* Recompute frame geometry from current window dimensions.
   The usable pixel area is the window minus BORDER pixels on each side.
   max_payload determines how many data bytes fit in one frame at current bpp. */
static void update_geometry(void)
{
    XWindowAttributes win_attrs;
    XGetWindowAttributes(display, window, &win_attrs);
    win_width  = win_attrs.width;
    win_height = win_attrs.height;

    /* Usable pixel grid = window area minus borders on each side */
    pixels_per_row = win_width  - 2 * BORDER;
    max_rows       = win_height - 2 * BORDER;
    if (pixels_per_row < 1) pixels_per_row = 1;
    if (max_rows < 1) max_rows = 1;

    /* Total usable pixels, minus fixed-size header pixels, converted to bytes */
    max_payload = ((size_t)pixels_per_row * (size_t)max_rows - HDR_PIXEL_COUNT)
                  * (size_t)bits_per_pixel / 8;
    printf("Window: %dx%d, usable: %d\xc3\x97%d px, %d bpp, payload: %zu bytes/frame\n",
           win_width, win_height, pixels_per_row, max_rows, bits_per_pixel, max_payload);
}

static void x11_init(void)
{
    display = XOpenDisplay(NULL);
    if (!display) { fprintf(stderr, "Cannot open X display\n"); exit(1); }

    screen_num   = DefaultScreen(display);
    visual       = DefaultVisual(display, screen_num);
    color_depth  = DefaultDepth(display, screen_num);
    graphics_ctx = DefaultGC(display, screen_num);

    window = XCreateSimpleWindow(display, RootWindow(display, screen_num),
                                 0, 0, INIT_WIDTH, INIT_HEIGHT, 0, 0, 0);
    XStoreName(display, window, "sender");

    /* Register for WM_DELETE_WINDOW so we handle window close gracefully */
    wm_delete_atom = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wm_delete_atom, 1);
    XSelectInput(display, window, ExposureMask | KeyPressMask | StructureNotifyMask);
    XMapWindow(display, window);

    /* Block until the window is mapped and the first Expose event arrives */
    XEvent event;
    while (1) { XNextEvent(display, &event); if (event.type == Expose) break; }

    update_geometry();
}

/* ── frame encoding & display ────────────────────────────────────────── */

/* Encode a single chunk as a pixel frame and display it in the X11 window.
   The frame has two regions:
   1. Header (always 3 bpp / 1 bit per channel) — carries sync marker,
      geometry, chunk metadata, payload length, and CRC32.
   2. Payload (at the configured bits_per_pixel) — the actual data bytes.
   The header is always at 3 bpp so the receiver can locate and parse it
   without knowing the payload's bpp setting in advance. */
static void show_frame(const uint8_t *payload, size_t payload_len,
                       int chunk_idx, int total_chunks)
{
    int bits_per_chan = bits_per_pixel / 3;     /* bits encoded per color channel */
    int quantize_levels = 1 << bits_per_chan;   /* distinct values per channel */

    /* ── Build the 18-byte frame header ── */
    uint32_t payload_crc = crc32_compute(payload, payload_len);
    uint8_t header[HEADER_SIZE];
    memcpy(header, SYNC_MARKER, 4);                         /* [0..3]   sync pattern */
    put_le16(header + 4,  (uint16_t)pixels_per_row);        /* [4..5]   pixels per row */
    header[6] = (uint8_t)bits_per_pixel;                    /* [6]      bits per pixel */
    put_le16(header + 7,  (uint16_t)total_chunks);          /* [7..8]   total chunks */
    put_le16(header + 9,  (uint16_t)chunk_idx);             /* [9..10]  chunk index */
    put_le24(header + 11, (uint32_t)payload_len);           /* [11..13] payload length */
    put_le32(header + 14, payload_crc);                     /* [14..17] CRC32 */

    /* Allocate a full-window pixel buffer (zero-initialized = black background) */
    uint32_t *framebuf = calloc((size_t)win_width * win_height, sizeof(uint32_t));
    if (!framebuf) return;

    /* ── Pass 1: Encode header pixels at fixed 3 bpp (1 bit per channel) ── */
    size_t header_bit_count = HEADER_SIZE * 8;
    for (size_t pixel_idx = 0; pixel_idx < HDR_PIXEL_COUNT; pixel_idx++) {
        int col = (int)(pixel_idx % (size_t)pixels_per_row);
        int row = (int)(pixel_idx / (size_t)pixels_per_row);
        int draw_x = BORDER + col, draw_y = BORDER + row;
        if (draw_y >= win_height) break;

        uint32_t pixel = 0;
        for (int chan = 0; chan < 3; chan++) {
            size_t bit_pos = pixel_idx * 3 + (size_t)chan;
            unsigned bit_val = 0;
            if (bit_pos < header_bit_count) {
                size_t byte_idx = bit_pos / 8;
                int bit_in_byte = 7 - (int)(bit_pos % 8);
                if ((header[byte_idx] >> bit_in_byte) & 1)
                    bit_val = 1;
            }
            /* 1 bit per channel: map to either 0 or 255 */
            pixel |= (uint32_t)(bit_val ? 255 : 0) << (chan * 8);
        }
        framebuf[draw_y * win_width + draw_x] = pixel;
    }

    /* ── Pass 2: Encode payload pixels at configured bpp ──
       Each pixel's 3 channels each carry bits_per_chan bits. Channel values
       are quantized: e.g. at 2 bits/chan, values 0–3 map to 0, 85, 170, 255. */
    size_t payload_bit_count = payload_len * 8;
    for (size_t pixel_idx = 0; pixel_idx * (size_t)bits_per_pixel < payload_bit_count; pixel_idx++) {
        /* Payload pixels start immediately after the header pixels */
        size_t abs_pixel_idx = HDR_PIXEL_COUNT + pixel_idx;
        int col = (int)(abs_pixel_idx % (size_t)pixels_per_row);
        int row = (int)(abs_pixel_idx / (size_t)pixels_per_row);
        int draw_x = BORDER + col, draw_y = BORDER + row;
        if (draw_y >= win_height) break;

        uint32_t pixel = 0;
        for (int chan = 0; chan < 3; chan++) {
            unsigned chan_val = 0;
            /* Extract bits_per_chan bits for this channel from the payload */
            for (int bit_idx = 0; bit_idx < bits_per_chan; bit_idx++) {
                size_t bit_pos = pixel_idx * (size_t)bits_per_pixel
                               + (size_t)chan * (size_t)bits_per_chan
                               + (size_t)bit_idx;
                if (bit_pos < payload_bit_count) {
                    size_t byte_idx = bit_pos / 8;
                    int bit_in_byte = 7 - (int)(bit_pos % 8);
                    if ((payload[byte_idx] >> bit_in_byte) & 1)
                        chan_val |= 1u << (bits_per_chan - 1 - bit_idx);
                }
            }
            /* Scale quantized value to full 0–255 range */
            uint8_t color = (uint8_t)(chan_val * 255 / (quantize_levels - 1));
            pixel |= (uint32_t)color << (chan * 8);
        }
        framebuf[draw_y * win_width + draw_x] = pixel;
    }

    /* ── Blit the pixel buffer to the X11 window ── */
    XImage *ximage = XCreateImage(display, visual, (unsigned)color_depth, ZPixmap, 0,
                                  (char *)framebuf, (unsigned)win_width, (unsigned)win_height, 32, 0);
    if (ximage) {
        XPutImage(display, window, graphics_ctx, ximage, 0, 0, 0, 0,
                  (unsigned)win_width, (unsigned)win_height);
        XFlush(display);
        XDestroyImage(ximage); /* also frees framebuf */
    } else {
        free(framebuf);
    }
}

/* Display an all-black screen to signal idle state (no file being sent). */
static void show_idle(void)
{
    uint32_t *framebuf = calloc((size_t)win_width * win_height, sizeof(uint32_t));
    if (!framebuf) return;
    XImage *ximage = XCreateImage(display, visual, (unsigned)color_depth, ZPixmap, 0,
                                  (char *)framebuf, (unsigned)win_width, (unsigned)win_height, 32, 0);
    if (ximage) {
        XPutImage(display, window, graphics_ctx, ximage, 0, 0, 0, 0,
                  (unsigned)win_width, (unsigned)win_height);
        XFlush(display);
        XDestroyImage(ximage);
    } else {
        free(framebuf);
    }
}

/* ── wait for ACK (any keypress) ─────────────────────────────────────── */

/* Block until the operator presses any key in the X11 window.
   Returns 1 on keypress (ACK), 0 if the window was closed or quit signaled. */
static int wait_ack(void)
{
    XEvent event;
    while (!quit) {
        while (XPending(display)) {
            XNextEvent(display, &event);
            if (event.type == KeyPress) return 1;
            if (event.type == ClientMessage &&
                (Atom)event.xclient.data.l[0] == wm_delete_atom) { quit = 1; return 0; }
        }
        msleep(10);
    }
    return 0;
}

/* Drain any pending key events so stale keypresses don't skip future frames. */
static void drain_keys(void)
{
    XEvent event;
    while (XPending(display)) {
        XNextEvent(display, &event);
        if (event.type == ClientMessage &&
            (Atom)event.xclient.data.l[0] == wm_delete_atom) quit = 1;
    }
}

/* ── compress + split a file ─────────────────────────────────────────── */

/* Compress a file with tar+gzip, then split the archive into chunks that
   each fit in one frame. Chunks are named .chunk.aa, .chunk.ab, … (dot prefix
   so find_next_file() skips them — it ignores dotfiles).
   Returns 1 on success; caller is responsible for deleting chunk files. */
static int compress_and_split(const char *filepath, const char *watchdir,
                              int *out_count)
{
    /* Separate filepath into directory and basename for tar's -C flag */
    const char *basename = strrchr(filepath, '/');
    const char *parent_dir;
    char parent_buf[1024];
    if (basename) {
        size_t dir_len = (size_t)(basename - filepath);
        if (dir_len == 0) dir_len = 1;  /* handle root paths like "/file" */
        memcpy(parent_buf, filepath, dir_len);
        parent_buf[dir_len] = '\0';
        parent_dir = parent_buf;
        basename++;  /* skip the '/' */
    } else {
        parent_dir = ".";
        basename = filepath;
    }

    /* Pipe: tar compresses to stdout → split chops into max_payload-sized pieces */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "tar czf - -C '%s' '%s' | split -b %zu - '%s/.chunk.'",
             parent_dir, basename, max_payload, watchdir);

    int exit_code = system(cmd);
    if (exit_code != 0) {
        fprintf(stderr, "compress/split failed (exit %d)\n", exit_code);
        return 0;
    }

    /* Count how many chunk files were produced */
    DIR *dir_handle = opendir(watchdir);
    if (!dir_handle) { perror(watchdir); return 0; }
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir_handle)) != NULL) {
        if (strncmp(entry->d_name, ".chunk.", 7) == 0) count++;
    }
    closedir(dir_handle);

    *out_count = count;
    return 1;
}

/* ── scan drop directory for files ───────────────────────────────────── */

/* Scan the watch directory for the first non-dotfile regular file.
   Returns a malloc'd path string, or NULL if nothing is ready. */
static char *find_next_file(const char *watchdir)
{
    DIR *dir_handle = opendir(watchdir);
    if (!dir_handle) return NULL;
    struct dirent *entry;
    while ((entry = readdir(dir_handle)) != NULL) {
        /* Skip dotfiles (includes ".", "..", and our ".chunk.*" temp files) */
        if (entry->d_name[0] == '.') continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", watchdir, entry->d_name);
        struct stat file_stat;
        if (stat(path, &file_stat) == 0 && S_ISREG(file_stat.st_mode) && file_stat.st_size > 0) {
            closedir(dir_handle);
            char *result = malloc(strlen(path) + 1);
            strcpy(result, path);
            return result;
        }
    }
    closedir(dir_handle);
    return NULL;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *watchdir = "./drop";

    /* Parse command line: [-b bits_per_pixel] [watch_directory] */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bits_per_pixel = atoi(argv[++i]);
            if (bits_per_pixel < 3 || bits_per_pixel > 24 || bits_per_pixel % 3 != 0) {
                fprintf(stderr, "Invalid -b: %d (must be 3,6,9,12,15,18,21,24)\n",
                        bits_per_pixel);
                return 1;
            }
        } else {
            watchdir = argv[i];
        }
    }

    mkdir(watchdir, 0755);  /* ensure watch directory exists */

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    x11_init();
    show_idle();
    printf("Watching '%s' for files. Press Ctrl+C to exit.\n", watchdir);

    while (!quit) {
        char *filepath = find_next_file(watchdir);
        if (!filepath) {
            /* No files ready — poll every 500ms (50 × 10ms sleeps),
               draining stale X events between sleeps. */
            for (int i = 0; i < 50 && !quit; i++) {
                drain_keys();
                msleep(10);
            }
            continue;
        }

        printf("Found: %s\n", filepath);

        /* Re-query window size in case it was resized/maximized since last file */
        update_geometry();

        int num_chunks = 0;
        if (!compress_and_split(filepath, watchdir, &num_chunks) || num_chunks == 0) {
            fprintf(stderr, "  Failed to process, skipping.\n");
            free(filepath);
            continue;
        }

        printf("  %d chunk(s), streaming...\n", num_chunks);

        /* Display each chunk as a frame and wait for operator ACK */
        int success = 1;
        for (int chunk_num = 0; chunk_num < num_chunks && !quit; chunk_num++) {
            /* Chunk filenames use two-letter alphabetic suffixes: aa, ab, … az, ba, … */
            char chunk_path[1024];
            char suffix[3] = { (char)('a' + chunk_num / 26),
                               (char)('a' + chunk_num % 26), '\0' };
            snprintf(chunk_path, sizeof(chunk_path), "%s/.chunk.%s", watchdir, suffix);

            size_t chunk_len = 0;
            uint8_t *chunk_data = load_file(chunk_path, &chunk_len);
            if (!chunk_data) {
                fprintf(stderr, "  Can't read chunk %s\n", chunk_path);
                success = 0; break;
            }

            show_frame(chunk_data, chunk_len, chunk_num, num_chunks);
            printf("  Frame %d/%d displayed, waiting for ACK...\n",
                   chunk_num + 1, num_chunks);
            free(chunk_data);

            drain_keys();  /* clear stale keypresses before waiting */
            if (!wait_ack()) { success = 0; break; }
            printf("  ACK received.\n");
        }

        /* Remove temporary chunk files regardless of success */
        for (int chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
            char chunk_path[1024];
            char suffix[3] = { (char)('a' + chunk_num / 26),
                               (char)('a' + chunk_num % 26), '\0' };
            snprintf(chunk_path, sizeof(chunk_path), "%s/.chunk.%s", watchdir, suffix);
            unlink(chunk_path);
        }

        if (success && !quit) {
            unlink(filepath);
            printf("  Transfer complete, deleted '%s'.\n", filepath);
        }

        show_idle();
        free(filepath);
    }

    printf("Exiting.\n");
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    return 0;
}
