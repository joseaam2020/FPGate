"""TLS server speaking a non-HTTP protocol (like Google's push channel on :5228).

usage: tls_echo.py PORT CERT KEY      replies b"ACK:" + whatever the client sent
"""
import socket
import ssl
import sys
import threading

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(sys.argv[2], sys.argv[3])
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", int(sys.argv[1])))
srv.listen(8)


def serve(raw):
    try:
        with ctx.wrap_socket(raw, server_side=True) as s:
            while True:
                data = s.recv(4096)
                if not data:
                    return
                s.sendall(b"ACK:" + data)
    except OSError:
        pass


while True:
    conn, _ = srv.accept()
    threading.Thread(target=serve, args=(conn,), daemon=True).start()
