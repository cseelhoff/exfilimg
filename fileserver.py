#!/usr/bin/env python3
"""Minimal upload/download HTTP file server — no dependencies beyond stdlib.

Usage: python3 fileserver.py [--port PORT] [--dir DIR] [--bind ADDR]
"""

import os, html, argparse, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

DIR = "."

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        name = urllib.parse.unquote(self.path.lstrip("/"))
        if not name:
            return self._listing()
        safe = os.path.basename(name)
        path = os.path.join(DIR, safe)
        if not safe or not os.path.isfile(path):
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Length", os.path.getsize(path))
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def do_POST(self):
        ct = self.headers.get("Content-Type", "")
        boundary = None
        for p in ct.split(";"):
            p = p.strip()
            if p.startswith("boundary="):
                boundary = p[9:].strip('"')
        if not boundary:
            return self.send_error(400, "No boundary")

        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 500_000_000:
            return self.send_error(400, "Bad Content-Length")
        body = self.rfile.read(length)

        fence = ("--" + boundary).encode()
        for part in body.split(fence):
            if b"\r\n\r\n" not in part:
                continue
            hdr, data = part.split(b"\r\n\r\n", 1)
            if data.endswith(b"\r\n"):
                data = data[:-2]
            filename = None
            for tok in hdr.decode(errors="replace").split(";"):
                tok = tok.strip()
                if tok.startswith("filename="):
                    filename = os.path.basename(tok[9:].strip('"'))
            if not filename or not data:
                continue
            with open(os.path.join(DIR, filename), "wb") as f:
                f.write(data)
            self.log_message("Saved %s (%d bytes)", filename, len(data))

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def _listing(self):
        files = sorted(f for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f)))
        rows = "".join(
            f'<tr><td><a href="/{urllib.parse.quote(f)}">{html.escape(f)}</a></td>'
            f'<td>{os.path.getsize(os.path.join(DIR, f)):,}</td></tr>' for f in files
        )
        page = (
            "<!DOCTYPE html><html><head><title>Files</title>"
            "<style>body{font-family:monospace;margin:2em}"
            "td,th{padding:2px 12px;text-align:left}</style></head><body>"
            "<h2>Files</h2>"
            '<form method="POST" enctype="multipart/form-data">'
            '<input type="file" name="file" multiple> <input type="submit" value="Upload">'
            "</form><table><tr><th>Name</th><th>Size</th></tr>"
            f"{rows}</table></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(page))
        self.end_headers()
        self.wfile.write(page)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", type=int, default=80)
    ap.add_argument("-b", "--bind", default="0.0.0.0")
    ap.add_argument("-d", "--dir", default=".")
    args = ap.parse_args()
    DIR = os.path.abspath(args.dir)
    os.makedirs(DIR, exist_ok=True)
    print(f"Serving {DIR} on {args.bind}:{args.port}")
    HTTPServer((args.bind, args.port), Handler).serve_forever()
