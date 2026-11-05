"""CONNECT through the proxy to a TLS service that is not HTTP and talk to it.

usage: opaque_client.py PROXY_PORT HOST PORT CA_CERT
Sends two binary messages and checks both replies come back intact.
"""
import socket
import ssl
import sys

proxy_port, host, port, ca = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
raw = socket.create_connection(("127.0.0.1", proxy_port), timeout=15)
raw.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
head = b""
while b"\r\n\r\n" not in head:
    head += raw.recv(1)
assert b" 200 " in head.split(b"\r\n")[0], head

tls = ssl.create_default_context(cafile=ca).wrap_socket(raw, server_hostname=host)
for msg in (b"\x29\x00\x07binary-protocol-hello\xff\xfe", b"\x10\x01second message"):
    tls.sendall(msg)
    got = b""
    while len(got) < len(msg) + 4:
        chunk = tls.recv(4096)
        if not chunk:
            sys.exit("connection closed before the reply arrived")
        got += chunk
    if got != b"ACK:" + msg:
        sys.exit(f"unexpected reply {got!r}")
print("ok both binary messages came back intact")
