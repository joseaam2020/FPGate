"""Tiny origin server used by the integration tests.

/echo     -> 200, body is the request head as received (headers preserved?)
/chunked  -> 200, Transfer-Encoding: chunked
/big      -> 200, 3 MiB body of a repeating pattern (streaming / backpressure)
/post     -> echoes the POST body
"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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

    def do_GET(self):
        if self.path.startswith("/echo"):
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


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
