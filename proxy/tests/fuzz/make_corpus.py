#!/usr/bin/env python3
"""Seed inputs for the fuzz targets, so they start from valid-looking data.

Usage: make_corpus.py OUT_DIR   (creates OUT_DIR/<target>/seed-N)
"""
import io
import os
import sys

from PIL import Image


def put(d, name, data):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "wb") as f:
        f.write(data)


def http_seeds(d):
    # first two bytes are the harness header: flags, piece size - 1
    def req(flags, piece, body):
        return bytes([flags, piece]) + body
    msgs_req = [
        b"GET /a.png?x=1 HTTP/1.1\r\nHost: example.com\r\nAccept: image/avif,image/webp,*/*\r\n\r\n",
        b"POST /up HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\n\r\nhello",
        b"POST /up HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n",
        b"GET /a HTTP/1.1\r\nHost: h\r\n\r\nGET /b HTTP/1.1\r\nHost: h\r\n\r\n",
        b"GET /ws HTTP/1.1\r\nHost: h\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\nraw bytes",
        b"POST /x HTTP/1.1\r\nContent-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\n",
    ]
    msgs_resp = [
        b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4\r\n\r\n\x89PNG",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n\x89PNG\r\n0\r\nX-T: 1\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nuntil close",
        b"HTTP/1.1 204 No Content\r\n\r\n",
        b"HTTP/1.1 304 Not Modified\r\nContent-Length: 10\r\n\r\n",
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\nframes",
        b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok",
        b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/10\r\nContent-Length: 4\r\n\r\nabcd",
    ]
    n = 0
    for m in msgs_req:
        for flags, piece in ((0, 255), (4, 7), (0, 0)):
            put(d, f"seed-{n}", req(flags, piece, m)); n += 1
    for m in msgs_resp:
        for flags, piece in ((1, 255), (3, 5), (5, 0)):
            put(d, f"seed-{n}", req(flags, piece, m)); n += 1


def detect_seeds(d):
    seeds = [b"\xff\xd8\xff\xe0", b"\x89PNG\r\n\x1a\n", b"RIFF\x00\x00\x00\x00WEBPVP8 ", b"GIF89a",
             b"\x00\x00\x00\x1cftypavif", b"BM", b"text/html; charset=utf-8", b"image/svg+xml",
             b"image/png", b"application/octet-stream", b"Image/JPEG;q=1",
             b"text/html,application/xhtml+xml,image/avif,image/webp,image/apng,*/*;q=0.8",
             b"image/avif", b"IMAGE/AVIF;q=0.9, image/webp", b"image/avifx, image/avif ,*/*"]
    for i, s in enumerate(seeds):
        put(d, f"seed-{i}", s)


def image_seeds(d):
    im = Image.new("RGB", (40, 30))
    px = im.load()
    for x in range(40):
        for y in range(30):
            px[x, y] = (x * 6, y * 8, (x * y) % 256)
    n = 0
    for fmt, kw in (("JPEG", {"quality": 80}), ("JPEG", {"quality": 30, "progressive": True}),
                    ("PNG", {}), ("WEBP", {"quality": 70}), ("WEBP", {"lossless": True})):
        b = io.BytesIO(); im.save(b, fmt, **kw); put(d, f"seed-{n}", b.getvalue()); n += 1
    for mode in ("L", "P", "RGBA", "LA", "1"):
        b = io.BytesIO(); im.convert(mode).save(b, "PNG"); put(d, f"seed-{n}", b.getvalue()); n += 1
    b = io.BytesIO(); im.convert("L").save(b, "JPEG"); put(d, f"seed-{n}", b.getvalue()); n += 1
    b = io.BytesIO(); im.convert("RGBA").save(b, "WEBP"); put(d, f"seed-{n}", b.getvalue()); n += 1


def toml_seeds(d):
    seeds = [b'[clasificacion]\numbral = 0.50\n\n[imagenes]\nsustitucion = "/etc/fpgate/substitute.png"\n',
             b"# comment only\n", b"[clasificacion]\numbral = 7\n", b'[imagenes]\nsustitucion = "unterminated\n',
             b"umbral = 0.3\n", b"[a]\n[b]\nx = 1\n[clasificacion]\numbral = .5 # trailing\n",
             b'[proxy]\npuerto = 8080\n\n[clasificacion]\numbral = 0.6   # strict\ntimeout_accel_ms = 10000\n']
    for i, s in enumerate(seeds):
        put(d, f"seed-{i}", s)


def creds_seeds(d):
    for i, s in enumerate([b"admin:20000:" + b"ab" * 16 + b":" + b"cd" * 32 + b"\n", b"x:1:2:3\n", b"::::\n",
                           b"admin:99999999:" + b"00" * 16 + b":" + b"ff" * 32]):
        put(d, f"seed-{i}", s)


def admin_seeds(d):
    png = io.BytesIO(); Image.new("RGB", (8, 8), (9, 9, 9)).save(png, "PNG"); png = png.getvalue()
    j = b"Content-Type: application/json\r\n"
    seeds = [
        b"GET / HTTP/1.1\r\nHost: fpgate.admin\r\n\r\n",
        b"GET /dashboard HTTP/1.1\r\nHost: h\r\nCookie: fpgate_session=" + b"a" * 32 + b"\r\n\r\n",
        b"GET /api/state HTTP/1.1\r\nHost: h\r\n\r\n",
        b"GET /api/blocklog?page=2&size=5 HTTP/1.1\r\nHost: h\r\n\r\n",
        b"GET /api/substitute HTTP/1.1\r\nHost: h\r\n\r\n",
        b"GET /ca.crt HTTP/1.1\r\nHost: h\r\n\r\n",
        b"GET /certificado HTTP/1.1\r\nHost: h\r\n\r\n",
        b'POST /api/login HTTP/1.1\r\nHost: h\r\n' + j + b'Content-Length: 40\r\n\r\n{"user":"admin","password":"correct horse"}',
        b'POST /api/threshold HTTP/1.1\r\nHost: h\r\n' + j + b'Content-Length: 14\r\n\r\n{"value":0.25}',
        b'POST /api/threshold HTTP/1.1\r\nHost: h\r\n' + j + b'Content-Length: 22\r\n\r\n{"value":"\\"x\\y"  ,}',
        b"POST /api/substitute HTTP/1.1\r\nHost: h\r\nContent-Type: image/png\r\nContent-Length: %d\r\n\r\n" % len(png) + png,
        b"POST /api/logout HTTP/1.1\r\nHost: h\r\nOrigin: https://h\r\nContent-Length: 2\r\n\r\n{}",
        b"POST /api/login HTTP/1.1\r\nHost: h\r\nOrigin: https://evil\r\nContent-Length: 0\r\n\r\n",
        b"POST /api/threshold HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\n",
    ]
    for i, s in enumerate(seeds):
        put(d, f"seed-{i}", s)


out = sys.argv[1]
for name, fn in (("http_stream", http_seeds), ("detect", detect_seeds), ("image", image_seeds),
                 ("toml", toml_seeds), ("creds", creds_seeds), ("admin", admin_seeds)):
    fn(os.path.join(out, name))
print("seeds written to", out)
