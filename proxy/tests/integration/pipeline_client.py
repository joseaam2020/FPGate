"""Sends several HTTP/1.1 requests in one write through an intercepted CONNECT
tunnel and checks every response, in order, against an expected SHA-256.

usage: pipeline_client.py PROXY_PORT HOST PORT CA_CERT PATH=SHA256|PATH=- ...

A "-" instead of a hash only checks that a 200 came back. Prints one line per
request and exits non-zero on the first difference.
"""
import hashlib
import socket
import ssl
import sys


def read_response(fp):
    status = fp.readline().decode().split(" ", 2)
    headers = {}
    while True:
        line = fp.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode().partition(":")
        headers[k.strip().lower()] = v.strip()
    if "chunked" in headers.get("transfer-encoding", ""):
        body = b""
        while True:
            size = int(fp.readline().split(b";")[0], 16)
            if size == 0:
                while fp.readline() not in (b"\r\n", b"\n"):
                    pass
                break
            body += fp.read(size)
            fp.readline()
    else:
        body = fp.read(int(headers.get("content-length", 0)))
    return int(status[1]), body


def main():
    proxy_port, host, port, ca = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    wanted = [a.rsplit("=", 1) for a in sys.argv[5:]]

    raw = socket.create_connection(("127.0.0.1", proxy_port), timeout=30)
    raw.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += raw.recv(1)
    assert b" 200 " in head.split(b"\r\n")[0], head

    ctx = ssl.create_default_context(cafile=ca)
    tls = ctx.wrap_socket(raw, server_hostname=host)
    tls.sendall(b"".join(f"GET {p} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode()
                         for p, _ in wanted))
    fp = tls.makefile("rb")
    for path, want in wanted:
        status, body = read_response(fp)
        got = hashlib.sha256(body).hexdigest()
        ok = status == 200 and (want == "-" or got == want)
        print(f"{'ok  ' if ok else 'DIFF'} {path} status={status} bytes={len(body)}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
