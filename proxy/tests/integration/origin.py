"""Tiny origin server used by the integration tests.

usage: origin.py PORT [CERT KEY]    (TLS when CERT and KEY are given)

/echo     -> 200, body is the request head as received (headers preserved?)
/chunked  -> 200, Transfer-Encoding: chunked
/big      -> 200, 3 MiB body of a repeating pattern (streaming / backpressure)
/post     -> echoes the POST body
/img/NAME -> serves $IMG_DIR/NAME, shaped by query parameters:
    mode=cl|chunked|close|smuggle   how the body is delimited (smuggle sends both
                                    Content-Length and Transfer-Encoding)
    ct=TYPE|none                    Content-Type override (default: from the extension)
    status=200|206|304
    enc=gzip                        adds a Content-Encoding header
    slow=1                          half of the body now, the rest two seconds later
HEAD is supported everywhere.
"""
import os
import ssl
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

IMG_DIR = os.environ.get("IMG_DIR", ".")
TYPES = {".jpg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
         ".webp": "image/webp", ".bin": "application/octet-stream", ".txt": "text/plain"}
REASONS = {200: "OK", 206: "Partial Content", 304: "Not Modified"}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, extra=()):
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _image(self, u, head):
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        name = u.path[len("/img/"):]
        with open(os.path.join(IMG_DIR, name), "rb") as f:
            data = f.read()
        mode, status = q.get("mode", "cl"), int(q.get("status", "200"))
        ct = q.get("ct", TYPES.get(os.path.splitext(name)[1], "application/octet-stream"))

        lines = [f"HTTP/1.1 {status} {REASONS[status]}"]
        if ct != "none":
            lines.append(f"Content-Type: {ct}")
        if "enc" in q:
            lines.append(f"Content-Encoding: {q['enc']}")
        if mode == "close":
            lines.append("Connection: close")
            self.close_connection = True
        elif mode == "chunked":
            lines.append("Transfer-Encoding: chunked")
        elif mode == "smuggle":
            lines += [f"Content-Length: {len(data)}", "Transfer-Encoding: chunked"]
        else:
            lines.append(f"Content-Length: {len(data)}")
        self.wfile.write(("\r\n".join(lines) + "\r\n\r\n").encode())
        if head or status == 304:
            return

        if mode == "chunked":
            cuts = [0, len(data) // 5, len(data) // 2, len(data)]
            for n, (a, b) in enumerate(zip(cuts, cuts[1:])):
                ext = b";ext=1" if n == 1 else b""
                self.wfile.write(b"%x%s\r\n" % (b - a, ext) + data[a:b] + b"\r\n")
                if "slow" in q and n == 0:
                    self.wfile.flush()
                    time.sleep(2)
            self.wfile.write(b"0\r\nX-Trailer: 1\r\n\r\n")
        elif "slow" in q:
            half = len(data) // 2
            self.wfile.write(data[:half])
            self.wfile.flush()
            time.sleep(2)
            self.wfile.write(data[half:])
        else:
            self.wfile.write(data)

    def do_HEAD(self):
        u = urlparse(self.path)
        if u.path.startswith("/img/"):
            self._image(u, True)
        else:
            self.send_error(404)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/img/"):
            self._image(u, False)
        elif self.path.startswith("/echo"):
            head = "".join(f"{k}: {v}\n" for k, v in self.headers.items())
            self._send((self.requestline + "\n" + head).encode())
        elif self.path == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (b"hello ", b"chunked ", b"world"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            self.wfile.write(b"0\r\n\r\n")
        elif self.path == "/big":
            self._send((b"0123456789abcdef" * 4096) * 48)
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self._send(self.rfile.read(n))


srv = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H)
if len(sys.argv) == 4:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(sys.argv[2], sys.argv[3])
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
srv.serve_forever()
