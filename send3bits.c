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
 *   ./sender [watch_dir]          # default: ./drop
 *
 * Frame header (17 bytes):
 *   [0..3]   CCSDS sync: 0x1A 0xCF 0xFC 0x1D
 *   [4..5]   pixels_per_row (LE) — auto-detected
 *   [6..7]   total_chunks (LE)
 *   [8..9]   chunk_index (LE)
 *   [10..12] payload_length (LE, 3 bytes)
 *   [13..16] CRC32 of payload (LE)
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

#define BORDER       5
#define HEADER_SIZE  17   /* 4 sync + 2 ppr + 2 total + 2 idx + 3 len + 4 crc32 */
#define INIT_W       800  /* initial window width; user can resize/maximize */
#define INIT_H       600  /* initial window height */

static const uint8_t SYNC[4] = { 0x1A, 0xCF, 0xFC, 0x1D };

static Display *dpy;
static Window   win;
static Visual  *vis;
static GC       gc;
static int      scr, depth;
static Atom     wm_delete;
static int      ppr;       /* pixels per row, auto-detected */
static int      max_rows;
static int      win_w, win_h;
static size_t   max_payload;

static volatile sig_atomic_t quit = 0;
static void on_signal(int s) { (void)s; quit = 1; }

/* ── CRC32 (same polynomial as zlib) ─────────────────────────────────── */

static uint32_t crc32_compute(const uint8_t *buf, size_t len)
{
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int b = 0; b < 8; b++)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}

/* ── helpers ─────────────────────────────────────────────────────────── */

static void msleep(int ms)
{
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static uint8_t *load_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    if (n <= 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t *buf = malloc((size_t)n);
    if (!buf || (long)fread(buf, 1, (size_t)n, f) != n) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

/* ── X11 setup ───────────────────────────────────────────────────────── */

/* Recompute geometry from current window dimensions */
static void update_geometry(void)
{
    XWindowAttributes wa;
    XGetWindowAttributes(dpy, win, &wa);
    win_w = wa.width;
    win_h = wa.height;
    ppr = win_w - 2 * BORDER;
    max_rows = win_h - 2 * BORDER;
    if (ppr < 1) ppr = 1;
    if (max_rows < 1) max_rows = 1;
    max_payload = (size_t)((ppr * 3) / 8 * max_rows) - HEADER_SIZE;
    printf("Window: %dx%d, usable: %d×%d px, payload: %zu bytes/frame\n",
           win_w, win_h, ppr, max_rows, max_payload);
}

static void x11_init(void)
{
    dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "Cannot open X display\n"); exit(1); }

    scr   = DefaultScreen(dpy);
    vis   = DefaultVisual(dpy, scr);
    depth = DefaultDepth(dpy, scr);
    gc    = DefaultGC(dpy, scr);

    win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr),
                              0, 0, INIT_W, INIT_H, 0, 0, 0);
    XStoreName(dpy, win, "sender");
    wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wm_delete, 1);
    XSelectInput(dpy, win, ExposureMask | KeyPressMask | StructureNotifyMask);
    XMapWindow(dpy, win);

    /* Wait for map */
    XEvent ev;
    while (1) { XNextEvent(dpy, &ev); if (ev.type == Expose) break; }

    update_geometry();
}

/* ── frame encoding & display ────────────────────────────────────────── */

static void show_frame(const uint8_t *payload, size_t payload_len,
                       int chunk_idx, int total_chunks)
{
    uint32_t crc = crc32_compute(payload, payload_len);
    uint8_t hdr[HEADER_SIZE];
    memcpy(hdr, SYNC, 4);
    hdr[4]  = ppr & 0xFF;
    hdr[5]  = (ppr >> 8) & 0xFF;
    hdr[6]  = total_chunks & 0xFF;
    hdr[7]  = (total_chunks >> 8) & 0xFF;
    hdr[8]  = chunk_idx & 0xFF;
    hdr[9]  = (chunk_idx >> 8) & 0xFF;
    hdr[10] = payload_len & 0xFF;
    hdr[11] = (payload_len >> 8) & 0xFF;
    hdr[12] = (payload_len >> 16) & 0xFF;
    hdr[13] = crc & 0xFF;
    hdr[14] = (crc >> 8) & 0xFF;
    hdr[15] = (crc >> 16) & 0xFF;
    hdr[16] = (crc >> 24) & 0xFF;

    size_t packed_len = HEADER_SIZE + payload_len;

    uint32_t *pixels = calloc((size_t)win_w * win_h, sizeof(uint32_t));
    if (!pixels) return;

    size_t bit_idx = 0;
    for (size_t i = 0; i < packed_len; i++) {
        uint8_t byte = (i < HEADER_SIZE) ? hdr[i] : payload[i - HEADER_SIZE];
        for (int b = 7; b >= 0; b--) {
            if ((byte >> b) & 1) {
                int chan = (int)(bit_idx % 3);
                int px   = (int)((bit_idx / 3) % (size_t)ppr);
                int py   = (int)((bit_idx / 3) / (size_t)ppr);
                int dx   = BORDER + px, dy = BORDER + py;
                if (dy < win_h) {
                    uint32_t shift = (chan == 0) ? 0 : (chan == 1) ? 8 : 16;
                    pixels[dy * win_w + dx] |= 0xFFu << shift;
                }
            }
            bit_idx++;
        }
    }

    XImage *img = XCreateImage(dpy, vis, (unsigned)depth, ZPixmap, 0,
                               (char *)pixels, (unsigned)win_w, (unsigned)win_h, 32, 0);
    if (img) {
        XPutImage(dpy, win, gc, img, 0, 0, 0, 0, (unsigned)win_w, (unsigned)win_h);
        XFlush(dpy);
        XDestroyImage(img); /* frees pixels */
    } else {
        free(pixels);
    }
}

static void show_idle(void)
{
    /* Black screen = idle/waiting for files */
    uint32_t *pixels = calloc((size_t)win_w * win_h, sizeof(uint32_t));
    if (!pixels) return;
    XImage *img = XCreateImage(dpy, vis, (unsigned)depth, ZPixmap, 0,
                               (char *)pixels, (unsigned)win_w, (unsigned)win_h, 32, 0);
    if (img) {
        XPutImage(dpy, win, gc, img, 0, 0, 0, 0, (unsigned)win_w, (unsigned)win_h);
        XFlush(dpy);
        XDestroyImage(img);
    } else {
        free(pixels);
    }
}

/* ── wait for ACK (any keypress) ─────────────────────────────────────── */

static int wait_ack(void)
{
    XEvent ev;
    while (!quit) {
        while (XPending(dpy)) {
            XNextEvent(dpy, &ev);
            if (ev.type == KeyPress) return 1;
            if (ev.type == ClientMessage &&
                (Atom)ev.xclient.data.l[0] == wm_delete) { quit = 1; return 0; }
        }
        msleep(10);
    }
    return 0;
}

/* Drain any pending key events so stale ACKs don't skip frames */
static void drain_keys(void)
{
    XEvent ev;
    while (XPending(dpy)) {
        XNextEvent(dpy, &ev);
        if (ev.type == ClientMessage &&
            (Atom)ev.xclient.data.l[0] == wm_delete) quit = 1;
    }
}

/* ── compress + split a file ─────────────────────────────────────────── */

static char *compress_and_split(const char *filepath, int *out_count)
{
    /* Create temp dir for chunks */
    char tmpdir[] = "/tmp/s3b_XXXXXX";
    if (!mkdtemp(tmpdir)) { perror("mkdtemp"); return NULL; }

    /* tar czf - <file> | split -b <max_payload> - <tmpdir>/chunk. */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "tar czf - -C '%s' '%s' | split -b %zu - '%s/chunk.'",
             /* dirname */ "", filepath, max_payload, tmpdir);

    /* We need dirname/basename separately. Find last slash. */
    const char *base = strrchr(filepath, '/');
    const char *dir;
    char dirbuf[1024];
    if (base) {
        size_t dlen = (size_t)(base - filepath);
        if (dlen == 0) dlen = 1;
        memcpy(dirbuf, filepath, dlen);
        dirbuf[dlen] = '\0';
        dir = dirbuf;
        base++;
    } else {
        dir = ".";
        base = filepath;
    }

    snprintf(cmd, sizeof(cmd),
             "tar czf - -C '%s' '%s' | split -b %zu - '%s/chunk.'",
             dir, base, max_payload, tmpdir);

    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "compress/split failed (exit %d)\n", rc);
        return NULL;
    }

    /* Count chunk files */
    DIR *d = opendir(tmpdir);
    if (!d) { perror(tmpdir); return NULL; }
    int count = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strncmp(ent->d_name, "chunk.", 6) == 0) count++;
    }
    closedir(d);

    *out_count = count;
    char *result = malloc(strlen(tmpdir) + 1);
    strcpy(result, tmpdir);
    return result;
}

/* ── scan drop directory for files ───────────────────────────────────── */

static char *find_next_file(const char *watchdir)
{
    DIR *d = opendir(watchdir);
    if (!d) return NULL;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", watchdir, ent->d_name);
        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
            closedir(d);
            char *result = malloc(strlen(path) + 1);
            strcpy(result, path);
            return result;
        }
    }
    closedir(d);
    return NULL;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *watchdir = (argc > 1) ? argv[1] : "./drop";

    /* Ensure watch directory exists */
    mkdir(watchdir, 0755);

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    x11_init();
    show_idle();
    printf("Watching '%s' for files. Press Ctrl+C to exit.\n", watchdir);

    while (!quit) {
        char *filepath = find_next_file(watchdir);
        if (!filepath) {
            /* Poll every 500ms */
            for (int i = 0; i < 50 && !quit; i++) {
                drain_keys();
                msleep(10);
            }
            continue;
        }

        printf("Found: %s\n", filepath);

        /* Query current window size so split matches actual geometry */
        update_geometry();

        int nchunks = 0;
        char *tmpdir = compress_and_split(filepath, &nchunks);
        if (!tmpdir || nchunks == 0) {
            fprintf(stderr, "  Failed to process, skipping.\n");
            free(filepath); free(tmpdir);
            continue;
        }

        printf("  %d chunk(s), streaming...\n", nchunks);

        /* Sort and send each chunk */
        int success = 1;
        for (int ci = 0; ci < nchunks && !quit; ci++) {
            /* Chunk filenames: chunk.aa, chunk.ab, ... */
            char chunkpath[1024];
            char suffix[3] = { (char)('a' + ci / 26), (char)('a' + ci % 26), '\0' };
            snprintf(chunkpath, sizeof(chunkpath), "%s/chunk.%s", tmpdir, suffix);

            size_t clen = 0;
            uint8_t *cdata = load_file(chunkpath, &clen);
            if (!cdata) {
                fprintf(stderr, "  Can't read chunk %s\n", chunkpath);
                success = 0; break;
            }

            show_frame(cdata, clen, ci, nchunks);
            printf("  Frame %d/%d displayed, waiting for ACK...\n", ci + 1, nchunks);
            free(cdata);

            drain_keys();  /* clear stale keys before waiting */
            if (!wait_ack()) { success = 0; break; }
            printf("  ACK received.\n");
        }

        /* Clean up temp chunks */
        char rmcmd[1024];
        snprintf(rmcmd, sizeof(rmcmd), "rm -rf '%s'", tmpdir);
        system(rmcmd);
        free(tmpdir);

        if (success && !quit) {
            unlink(filepath);
            printf("  Transfer complete, deleted '%s'.\n", filepath);
        }

        show_idle();
        free(filepath);
    }

    printf("Exiting.\n");
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    return 0;
}
