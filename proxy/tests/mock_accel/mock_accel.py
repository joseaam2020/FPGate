#!/usr/bin/env python3
"""Mock of fpgate-accel: speaks the classify protocol over a Unix socket.

Like the real accelerator it processes one image at a time (requests are
serialised), so the proxy's queue and timeouts behave realistically.

usage: mock_accel.py --socket PATH [--config FILE.json] [--log FILE.jsonl]
                     [--score S] [--latency SECONDS]

The JSON config is re-read on every request, so a test can change behaviour
while the proxy is running (e.g. start failing, then recover):

  {"score": 0.83,               # fixed score
   "score_from_pixels": false,  # true: score = mean pixel / 255 (white image -> 1.0)
   "latency": 0.2,              # seconds spent "computing" (the design expects 3-7 s)
   "fail": "none"}              # none | hang | close | error

The log holds one JSON line per classify request: id, size and sha256 of the
pixels. Pixels themselves are never written anywhere.
"""
import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time

REQ = struct.Struct("<4sBBBBIHHII")     # must mirror src/accel_proto.h
RESP = struct.Struct("<4sBBBBIfII")
MAGIC_REQ, MAGIC_RESP = b"FPGA", b"FPGR"
VERSION = 1
T_CLASSIFY, T_STATUS = 1, 2
OK, BUSY, HW_ERROR, UNSUPPORTED, DEGRADED = range(5)
MAX_PAYLOAD = 16 * 1024 * 1024
HW_CLOCK_HZ = 150_000_000               # design target, used to fake hw_cycles


class State:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()

    def cfg(self):
        cfg = {"score": self.args.score, "latency": self.args.latency,
               "fail": "none", "score_from_pixels": False}
        if self.args.config:
            try:
                with open(self.args.config) as f:
                    cfg.update(json.load(f))
            except (OSError, ValueError):
                pass
        return cfg

    def log(self, entry):
        if not self.args.log:
            return
        with self.log_lock, open(self.args.log, "a") as f:
            f.write(json.dumps(entry) + "\n")


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def reply(conn, typ, status, rid, score=0.0, cycles=0, latency_us=0):
    conn.sendall(RESP.pack(MAGIC_RESP, VERSION, typ, status, 0, rid,
                           score, cycles, latency_us))


def serve(conn, st):
    with conn:
        while True:
            raw = recv_exact(conn, REQ.size)
            if raw is None:
                return
            magic, ver, typ, ch, fmt, rid, w, h, plen, _ = REQ.unpack(raw)
            if magic != MAGIC_REQ:
                return                  # framing cannot be trusted: drop
            if ver != VERSION or plen > MAX_PAYLOAD:
                reply(conn, typ, UNSUPPORTED, rid)
                return
            payload = recv_exact(conn, plen)
            if payload is None:
                return

            if typ == T_STATUS:
                cfg = st.cfg()
                state = BUSY if st.lock.locked() else (DEGRADED if cfg["fail"] != "none" else OK)
                reply(conn, typ, state, rid)
                continue
            if (typ != T_CLASSIFY or fmt != 0 or ch != 3 or w == 0 or h == 0
                    or plen != w * h * ch):
                reply(conn, typ, UNSUPPORTED, rid)
                continue

            with st.lock:               # one image at a time, like the FPGA
                t0 = time.monotonic()
                cfg = st.cfg()
                mean = sum(payload) / len(payload) / 255.0
                st.log({"ts": time.time(), "id": rid, "w": w, "h": h, "bytes": plen,
                        "sha256": hashlib.sha256(payload).hexdigest(), "mean": round(mean, 4)})
                fail = cfg["fail"]
                if fail == "hang":
                    time.sleep(3600)    # accelerator stuck: never answers
                if fail == "close":
                    return              # drops the connection mid-request
                time.sleep(float(cfg["latency"]))
                elapsed = time.monotonic() - t0
                if fail == "error":
                    reply(conn, typ, HW_ERROR, rid, 0.0, 0, int(elapsed * 1e6))
                    continue
                score = mean if cfg["score_from_pixels"] else float(cfg["score"])
                reply(conn, typ, OK, rid, score, int(elapsed * HW_CLOCK_HZ),
                      int(elapsed * 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--socket", required=True)
    ap.add_argument("--config")
    ap.add_argument("--log")
    ap.add_argument("--score", type=float, default=0.0)
    ap.add_argument("--latency", type=float, default=0.0)
    args = ap.parse_args()

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o600)
    srv.listen(16)
    st = State(args)
    print(f"mock-accel listening on {args.socket}", file=sys.stderr, flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn, st), daemon=True).start()


if __name__ == "__main__":
    main()
