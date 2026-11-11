#!/usr/bin/env python3
"""Protocol test of fpgate-accel against the contract in proxy/src/accel_proto.h, with the real
model package. Mirrors the behaviour of mock_accel (see proxy/tests/mock_accel).

usage: test_protocol.py FPGATE_ACCEL_BINARY PKG_DIR INDEX_FILE
"""
import array
import math
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

REQ = struct.Struct("<4sBBBBIHHII")
RESP = struct.Struct("<4sBBBBIfII")
OK, BUSY, HW_ERROR, UNSUPPORTED, DEGRADED = range(5)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

failed = 0


def check(name, cond):
    global failed
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed = 1


def exchange(path, rid, w=448, h=448, payload=None, typ=1, version=1, timeout=120.0, fmt=0, ch=3):
    """Returns (status, score, cycles, latency_us, request_id), or None if no reply."""
    if payload is None:
        payload = bytes(w * h * ch) if typ == 1 else b""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        s.sendall(REQ.pack(b"FPGA", version, typ, ch, fmt, rid, w, h, len(payload), 0) + payload)
        data = b""
        try:
            while len(data) < RESP.size:
                chunk = s.recv(RESP.size - len(data))
                if not chunk:
                    return None
                data += chunk
        except socket.timeout:
            return None
    _, _, _, status, _, rid_back, score, cycles, lat = RESP.unpack(data)
    return status, score, cycles, lat, rid_back


def score_from_logits_hex(hexes):
    lg = [struct.unpack("<f", struct.pack("<I", h))[0] for h in hexes]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    return sum(e[1:]) / sum(e)


def main():
    binary, pkg, index = sys.argv[1:4]
    vec0 = None
    for line in open(index):
        t = line.split()
        if t and t[0] == "vector" and t[1] == "0":
            vec0 = (t[2], [int(x) for x in t[4:8]])
            break
    f = array.array("f")
    with open(os.path.join(pkg, vec0[0]), "rb") as fh:
        f.frombytes(fh.read())
    plane = 448 * 448
    px = bytearray(plane * 3)
    for c in range(3):
        base = c * plane
        for i in range(plane):
            v = round((f[base + i] * STD[c] + MEAN[c]) * 255.0)
            px[i * 3 + c] = 0 if v < 0 else 255 if v > 255 else v
    ref = score_from_logits_hex(vec0[1])

    d = tempfile.mkdtemp()
    sock = os.path.join(d, "accel.sock")
    srv = subprocess.Popen([binary, sock, "--pkg", pkg, "--index", index, "--quiet"], stderr=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(sock):
            break
        time.sleep(0.1)
    try:
        r = exchange(sock, 7, typ=2, payload=b"")
        check("STATUS answers OK", r is not None and r[0] == OK)
        check("STATUS echoes request_id", r is not None and r[4] == 7)

        t0 = time.time()
        r = exchange(sock, 0x01020304, payload=bytes(px))
        dt = time.time() - t0
        check("classify answers OK", r is not None and r[0] == OK)
        if r:
            check("request_id echoed", r[4] == 0x01020304)
            check(f"score {r[1]:.5f} within 0.01 of the reference {ref:.5f}", abs(r[1] - ref) < 0.01)
            check("same decision as the reference at threshold 0.5", (r[1] >= 0.5) == (ref >= 0.5))
            check("latency_us is reported and consistent", 0 < r[3] <= (dt + 1) * 1e6)
            print(f"   classify took {dt:.1f} s, latency_us = {r[3]}")

        # same contract as the mock: anything but 448x448 RGB8 HWC is UNSUPPORTED, and the
        # connection layer keeps working afterwards
        r = exchange(sock, 1, w=224, h=224)
        check("224x224 -> UNSUPPORTED", r is not None and r[0] == UNSUPPORTED)
        r = exchange(sock, 2, fmt=1)
        check("unknown pixel format -> UNSUPPORTED", r is not None and r[0] == UNSUPPORTED)
        r = exchange(sock, 3, ch=1, payload=bytes(448 * 448))
        check("1 channel -> UNSUPPORTED", r is not None and r[0] == UNSUPPORTED)
        r = exchange(sock, 4, payload=bytes(100))
        check("payload_len not matching the size -> UNSUPPORTED", r is not None and r[0] == UNSUPPORTED)
        r = exchange(sock, 5, version=9, payload=b"")
        check("unknown protocol version -> UNSUPPORTED", r is not None and r[0] == UNSUPPORTED)
        r = exchange(sock, 6, typ=2, payload=b"")
        check("still serving after the errors", r is not None and r[0] == OK and r[4] == 6)
    finally:
        srv.terminate()
        srv.wait()

    print("ALL PASSED" if not failed else "SOME FAILED")
    sys.exit(failed)


if __name__ == "__main__":
    main()
