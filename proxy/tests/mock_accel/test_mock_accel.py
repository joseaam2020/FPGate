#!/usr/bin/env python3
"""Tests the mock-accel and the wire format shared with the C header.

usage: test_mock_accel.py PROTO_DUMP_BINARY
"""
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK = os.path.join(HERE, "mock_accel.py")
REQ = struct.Struct("<4sBBBBIHHII")
RESP = struct.Struct("<4sBBBBIfII")
OK, BUSY, HW_ERROR, UNSUPPORTED, DEGRADED = range(5)

failed = 0


def check(name, cond):
    global failed
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed = 1


def exchange(path, rid, w=4, h=4, payload=None, typ=1, version=1, timeout=5.0, fmt=0, ch=3):
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


def main():
    dump = sys.argv[1]

    # --- C header and Python agree on the bytes -------------------------------
    out = subprocess.run([dump], capture_output=True, check=True).stdout
    want = REQ.pack(b"FPGA", 1, 1, 3, 0, 0x01020304, 224, 336, 224 * 336 * 3, 0)
    want += RESP.pack(b"FPGR", 1, 1, DEGRADED, 0, 0x0A0B0C0D, 0.5, 123456789, 4200000)
    check("wire format: C header and Python struct produce identical bytes", out == want)

    tmp = tempfile.mkdtemp()
    sock, cfgp, logp = (os.path.join(tmp, n) for n in ("accel.sock", "cfg.json", "log.jsonl"))

    def setcfg(**kw):
        with open(cfgp, "w") as f:
            json.dump(kw, f)

    setcfg(score=0.83)
    mock = subprocess.Popen([sys.executable, MOCK, "--socket", sock, "--config", cfgp,
                             "--log", logp], stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if os.path.exists(sock):
                break
            time.sleep(0.1)

        r = exchange(sock, 77)
        check("classify returns the configured score, status OK and echoes the id",
              r is not None and r[0] == OK and abs(r[1] - 0.83) < 1e-6 and r[4] == 77)

        setcfg(score=0.1)
        r = exchange(sock, 78)
        check("config is re-read on every request", r is not None and abs(r[1] - 0.1) < 1e-6)

        setcfg(score=0.5, latency=0.3)
        t0 = time.monotonic()
        r = exchange(sock, 79)
        dt = time.monotonic() - t0
        check("configured latency is applied and reported",
              r is not None and dt >= 0.3 and r[3] >= 290000 and r[2] > 0)

        # Concurrent requests are serialised, like the single FPGA engine.
        t0 = time.monotonic()
        ts = [threading.Thread(target=exchange, args=(sock, 80 + i)) for i in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        check("two concurrent requests are processed one at a time",
              time.monotonic() - t0 >= 0.6)

        setcfg(score_from_pixels=True)
        white = exchange(sock, 90, 4, 4, b"\xff" * 48)
        black = exchange(sock, 91, 4, 4, b"\x00" * 48)
        check("score_from_pixels: white image scores 1.0, black 0.0",
              white and black and abs(white[1] - 1.0) < 1e-6 and black[1] == 0.0)

        setcfg(score=0.2)
        r = exchange(sock, 92, typ=2)
        check("status request answers OK when idle", r is not None and r[0] == OK and r[4] == 92)

        check("unknown protocol version is refused",
              (exchange(sock, 93, version=9) or (None,))[0] == UNSUPPORTED)
        check("payload length not matching the dimensions is refused",
              (exchange(sock, 94, 4, 4, b"\x00" * 10) or (None,))[0] == UNSUPPORTED)
        check("unsupported pixel format is refused",
              (exchange(sock, 95, fmt=7) or (None,))[0] == UNSUPPORTED)

        setcfg(fail="error")
        r = exchange(sock, 100)
        check("fail=error: status HW_ERROR", r is not None and r[0] == HW_ERROR)
        r = exchange(sock, 101, typ=2)
        check("status reports DEGRADED while failing", r is not None and r[0] == DEGRADED)

        setcfg(fail="close")
        check("fail=close: connection dropped without a reply", exchange(sock, 102) is None)

        setcfg(fail="hang")
        check("fail=hang: no reply within the client timeout", exchange(sock, 103, timeout=1.0) is None)
        r = exchange(sock, 104, typ=2)
        check("status answers BUSY while the accelerator is stuck", r is not None and r[0] == BUSY)

        # The log holds metadata of every classify request, never pixels.
        with open(logp) as f:
            entries = [json.loads(line) for line in f]
        ids = [e["id"] for e in entries]
        check("log records every classify request received",
              all(i in ids for i in (77, 78, 79, 80, 81, 90, 91, 100, 102, 103)))
        e90 = next(e for e in entries if e["id"] == 90)
        check("log carries the sha256 of the pixels and no pixel data",
              e90["sha256"] == hashlib.sha256(b"\xff" * 48).hexdigest() and "pixels" not in e90)
    finally:
        mock.kill()
        mock.wait()

    sys.exit(failed)


if __name__ == "__main__":
    main()
