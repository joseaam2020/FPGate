#!/bin/bash
# RS-01 / RS-05: no image, and nothing derived from one, is written to disk.
#
# Images that carry a unique marker (and one with a unique pixel colour) go through the proxy in every
# way it handles them: allowed, blocked, corrupt, too big to hold, compressed, plain and intercepted
# HTTPS. Then every file that changed anywhere on the machine is listed, the files the proxy owns are
# compared with the ones it is supposed to write, and all of them are searched for the marker, for the
# pixels and for the decoded RGB tensor.
# Usage: tests/integration/test_no_persist.sh [proxy-binary] [mock_accel]
BIN=${1:-build/native/fpgate-proxy}
MOCKC=${2:-build/mock_accel}
T=$(mktemp -d)
fail=0
PIDS=""

curl() { command curl --max-time 25 "$@"; }
check() { # name, exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
cleanup() { kill $PIDS $MOCK 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT

MARK=FPGATE-NOPERSIST-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')
mkdir -p "$T/img" "$T/home/etc" "$T/home/tmp" "$T/home/cwd"
python3 - "$T/img" "$MARK" <<'EOF'
import io, sys
from PIL import Image, PngImagePlugin
d, mark = sys.argv[1], sys.argv[2]

def png(name, size, colour):
    info = PngImagePlugin.PngInfo(); info.add_text("Comment", mark)
    Image.new("RGB", size, colour).save(f"{d}/{name}", pnginfo=info)

png("white.png", (800, 600), (255, 255, 255))        # blocked by the mock (score 1.0)
png("dark.png", (640, 480), (0, 0, 0))               # allowed (score 0.0)
png("tensor.png", (500, 400), (0x12, 0x34, 0x56))    # allowed; a distinctive pixel colour
Image.new("RGB", (900, 700), (255, 255, 255)).save(f"{d}/white.jpg", quality=90, comment=mark.encode())
Image.new("RGB", (300, 200), (255, 255, 255)).save(f"{d}/white.webp", quality=90)
data = open(f"{d}/white.jpg", "rb").read()
open(f"{d}/trunc.jpg", "wb").write(data[: len(data) // 2])                  # corrupt, still carries the marker
# bigger than the -L limit given below: streamed, never held
big = io.BytesIO(); Image.effect_noise((900, 900), 80).convert("RGB").save(big, "PNG", pnginfo=PngImagePlugin.PngInfo())
open(f"{d}/big.png", "wb").write(big.getvalue() + mark.encode())
EOF
export IMG_DIR="$T/img"

grep -q fpgate.test /etc/hosts 2>/dev/null || echo "127.0.0.1 fpgate.test" >> /etc/hosts
export FPGATE_CA_PASS=testpass
sh tools/gen_ca.sh "$T/ca" >/dev/null 2>&1
printf 'testpass\n' > "$T/pass"
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 30 \
    -keyout "$T/oca.key" -out "$T/oca.crt" -subj /CN=OriginCA \
    -addext basicConstraints=critical,CA:TRUE -addext keyUsage=critical,keyCertSign 2>/dev/null
openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
    -keyout "$T/o.key" -out "$T/o.csr" -subj /CN=origin 2>/dev/null
printf 'subjectAltName=IP:127.0.0.1,DNS:fpgate.test\n' > "$T/ext.cnf"
openssl x509 -req -in "$T/o.csr" -CA "$T/oca.crt" -CAkey "$T/oca.key" -CAcreateserial \
    -out "$T/o.crt" -days 30 -extfile "$T/ext.cnf" 2>/dev/null
python3 tests/integration/origin.py 18081 & PIDS="$PIDS $!"
python3 tests/integration/origin.py 18463 "$T/o.crt" "$T/o.key" & PIDS="$PIDS $!"
$MOCKC "$T/m.sock" --mean --quiet 2>/dev/null & MOCK=$!

H="$T/home"
printf 'correct horse\n' | $BIN --set-admin-password "$H/etc/admin.pass" 2>/dev/null
touch "$T/ref"; sleep 1.2          # everything written from here on is "new"

# The proxy runs with its own working directory, temp directory and configuration directory,
# so whatever it writes is easy to tell apart.
( cd "$H/cwd" && TMPDIR="$H/tmp" exec "$OLDPWD/$BIN" -p 18190 \
    -C "$T/ca/ca.crt" -K "$T/ca/ca.key" -P "$T/pass" -B "$T/oca.crt" \
    --accel "$T/m.sock" --block-log "$H/etc/blocks.db" --config "$H/etc/fpgate.toml" \
    --admin-credentials "$H/etc/admin.pass" --admin-port 18455 -L 500000 -V 2>"$T/p.log" ) &
P=$!
PIDS="$PIDS $P"
sleep 2.5
X="-x http://127.0.0.1:18190"
O=http://127.0.0.1:18081
Q=https://fpgate.test:18463
CAF="--cacert $T/ca/ca.crt"

# --- traffic: every path an image can take ---------------------------------------------------------------------------------
curl -s $X "$O/img/white.png" -o /dev/null                              # blocked
curl -s $X "$O/img/dark.png" -o /dev/null                               # allowed
curl -s $X "$O/img/tensor.png" -o /dev/null                             # allowed, distinctive pixels
curl -s $X "$O/img/white.jpg?mode=chunked" -o /dev/null                 # blocked, chunked
curl -s $X "$O/img/white.webp" -o /dev/null                             # blocked WebP
curl -s $X "$O/img/trunc.jpg" -o /dev/null                              # corrupt
curl -s $X "$O/img/big.png" -o /dev/null                                # over the size limit: streamed
curl -s $X "$O/img/white.png?enc=gzip" -o /dev/null                     # compressed: passes unexamined
curl -s $X $CAF "$Q/img/white.png" -o /dev/null                         # intercepted HTTPS, blocked
curl -s $X $CAF "$Q/img/tensor.png" -o /dev/null                        # intercepted HTTPS, allowed
curl -s $X $CAF "$Q/img/white.jpg" -o /dev/null
# --- memory: the traffic is over and the process is still alive; look through all of its readable memory
python3 - "$P" "$MARK" > "$T/mem.txt" <<'PY'
import re, sys
pid, mark = sys.argv[1], sys.argv[2].encode()
pix = bytes([0x12, 0x34, 0x56]) * 100
control = b"Connection Established"      # a string the program certainly holds: proves the scan reads memory
n = {"scanned": 0, "control": 0, "marker": 0, "pixels": 0}
with open(f"/proc/{pid}/maps") as maps, open(f"/proc/{pid}/mem", "rb", 0) as mem:
    for line in maps:
        m = re.match(r"([0-9a-f]+)-([0-9a-f]+) (\S+) ", line)
        if not m or "r" not in m.group(3):
            continue
        lo, hi = int(m.group(1), 16), int(m.group(2), 16)
        try:
            mem.seek(lo)
            data = mem.read(hi - lo)
        except Exception:
            continue
        n["scanned"] += len(data)
        for k, needle in (("control", control), ("marker", mark), ("pixels", pix)):
            n[k] += data.count(needle)
print(n["scanned"], n["control"], n["marker"], n["pixels"])
PY
read scanned control marker pixels < "$T/mem.txt"
kill -TERM $P; wait $P 2>/dev/null                                      # shutdown flushes the block log
sleep 0.5

[ "${scanned:-0}" -gt 1000000 ] && [ "${control:-0}" -gt 0 ]
check "the memory scan really read the process (${scanned:-0} bytes; the control string was found ${control:-0} times)" $?
[ "${marker:-1}" = 0 ] && [ "${pixels:-1}" = 0 ]
check "no image bytes and no decoded pixels remain in the process memory after the images were handled (marker: $marker, pixels: $pixels)" $?

[ "$(grep -c '^VERDICT' "$T/p.log")" -ge 8 ]
check "the traffic really went through the judge ($(grep -c '^VERDICT' "$T/p.log") verdicts, blocks and allows)" $?
grep -q 'verdict=block' "$T/p.log" && grep -q 'verdict=allow' "$T/p.log"; check "both outcomes happened" $?
grep -q 'skipped=oversize' "$T/p.log"; check "the over-limit image took the streaming path" $?

# --- 1. what the proxy wrote ------------------------------------------------------------------------------------------------------------
echo "   files the proxy owns:"; (cd "$H" && find . -type f -newer "$T/ref" | sort | sed 's/^/     /')
unexpected=$(cd "$H" && find . -type f -newer "$T/ref" | grep -vE '^\./etc/(blocks\.db(-wal|-shm)?|fpgate\.toml)$' | sort)
[ -z "$unexpected" ]; check "the proxy wrote only its block log (and settings): nothing in its working or temp directory${unexpected:+ [$unexpected]}" $?
[ -z "$(find "$H/tmp" "$H/cwd" -mindepth 1)" ]; check "its temp directory and working directory are empty" $?

# --- 2. nothing else on the machine changed ------------------------------------------------------------------------------------------------
others=$(find / -xdev -type f -newer "$T/ref" \
    -not -path '/proc/*' -not -path '/sys/*' -not -path '/dev/*' -not -path "$T/*" \
    -not -path '/work/*' -not -path '/root/.cache/*' 2>/dev/null | sort)
[ -z "$others" ]; check "no file anywhere else on the machine changed while it ran${others:+ [$(echo $others | tr '\n' ' ')]}" $?

# --- 3. what is in the files the proxy owns ----------------------------------------------------------------------------------------------------
files=$(cd "$H" && find . -type f)
bad=0
for f in $files; do
    grep -aq "$MARK" "$H/$f" && { echo "   marker found in $f"; bad=1; }
done
grep -aq "$MARK" "$T/p.log" && { echo "   marker found in the log"; bad=1; }
[ $bad -eq 0 ]; check "the image bytes (marker) are in no file it wrote, nor in its log" $?

python3 - "$H" "$T/p.log" <<'EOF'
import os, sys
pixels = bytes([0x12, 0x34, 0x56]) * 100          # the decoded colour of tensor.png, as raw RGB
bad = 0
paths = [sys.argv[2]]
for root, _, names in os.walk(sys.argv[1]):
    paths += [os.path.join(root, n) for n in names]
for p in paths:
    data = open(p, "rb").read()
    if pixels in data or bytes([0x12, 0x34, 0x56]) * 40 in data:
        print("   raw pixels found in", p); bad = 1
sys.exit(bad)
EOF
check "no decoded pixels (the RGB tensor sent to the accelerator) are in any of them" $?

row=$(python3 -c "
import sqlite3
c = sqlite3.connect('$H/etc/blocks.db')
print(c.execute('select count(*) from blocks').fetchone()[0])
print(','.join(r[1] for r in c.execute('pragma table_info(blocks)')))")
[ "$(echo "$row" | head -1)" -ge 4 ] && [ "$(echo "$row" | tail -1)" = "id,ts,host,score,threshold,reason" ]
check "the block log has the blocks, as metadata only ($(echo "$row" | head -1) rows; columns $(echo "$row" | tail -1))" $?

[ $fail -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit $fail
