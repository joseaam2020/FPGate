#!/bin/bash
# Image decoding on worker threads: the event loop stays free, order is kept,
# connections can die mid-decode. Usage: tests/integration/test_workers.sh [binary]
BIN=${1:-build/asan/fpgate-proxy}
T=$(mktemp -d)
fail=0
PIDS=""

curl() { command curl --max-time 25 "$@"; }
check() { # name, exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
cleanup() { kill $PIDS 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
sha() { sha256sum | cut -d' ' -f1; }
fsha() { sha256sum "$1" | cut -d' ' -f1; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }

mkdir -p "$T/img"
python3 - "$T/img" <<'EOF'
import random, sys
from PIL import Image, ImageDraw, ImageFilter
d = sys.argv[1]
def photo(w, h, seed):
    r = random.Random(seed)
    im = Image.frombytes("RGB", (w, h), r.randbytes(w * h * 3)).filter(ImageFilter.GaussianBlur(2))
    ImageDraw.Draw(im).rectangle([w // 4, h // 4, w // 2, h // 2], fill=(200, 30, 30))
    return im
photo(800, 600, 1).save(d + "/photo.jpg", quality=85)
photo(300, 200, 4).save(d + "/pic.png")
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
python3 tests/integration/origin.py 18443 "$T/o.crt" "$T/o.key" & PIDS="$PIDS $!"
CA="-C $T/ca/ca.crt -K $T/ca/ca.key -P $T/pass -B $T/oca.crt"
# P1: one worker, every decode takes 800 ms longer. P2: same delay but decoding inline.
# P3: one worker, 300 ms per decode, for the concurrency tests.
$BIN -p 18090 $CA -T 800 2>"$T/p1.log" & P1=$!
$BIN -p 18091 -W 0 -T 800 2>"$T/p2.log" & P2=$!
$BIN -p 18092 $CA -T 300 2>"$T/p3.log" & P3=$!
PIDS="$PIDS $P1 $P2 $P3"
sleep 2
O=http://127.0.0.1:18081
JPG=$(fsha "$T/img/photo.jpg"); PNG=$(fsha "$T/img/pic.png")

# --- 1. the point of the exercise ---------------------------------------------
curl -s -x http://127.0.0.1:18091 "$O/img/photo.jpg" -o /dev/null & J=$!
sleep 0.3
t_inline=$(curl -s -o /dev/null -w '%{time_total}' -x http://127.0.0.1:18091 "$O/echo")
wait $J
! lt "$t_inline" 0.35; check "baseline: decoding inside the event loop stalls other traffic (echo took ${t_inline}s)" $?

curl -s -x http://127.0.0.1:18090 "$O/img/photo.jpg" -o "$T/w.out" & J=$!
sleep 0.3
t_worker=$(curl -s -o /dev/null -w '%{time_total}' -x http://127.0.0.1:18090 "$O/echo")
wait $J
lt "$t_worker" 0.3; check "with a worker, other traffic is not delayed (echo took ${t_worker}s)" $?
[ "$(sha < "$T/w.out")" = "$JPG" ]; check "the image decoded off-loop is delivered unchanged" $?
grep -q 'IMG host=127.0.0.1 kind=jpeg src=800x600 .*prep=' "$T/p1.log"; check "worker result is logged" $?

# --- 2. the server closes while the worker is still busy -----------------------
got=$(curl -s -x http://127.0.0.1:18090 "$O/img/photo.jpg?mode=close" | sha)
[ "$got" = "$JPG" ]; check "EOF while the job is in flight: image still delivered, then connection closed" $?

# --- 3. the client gives up while the worker is still busy ------------------------
curl -s --max-time 0.3 -x http://127.0.0.1:18090 "$O/img/photo.jpg" -o /dev/null
sleep 1.5
[ "$(curl -s -x http://127.0.0.1:18090 "$O/echo" | head -1)" = "GET /echo HTTP/1.1" ]
check "client aborted mid-decode: proxy keeps serving" $?
kill -0 $P1 2>/dev/null; check "client aborted mid-decode: proxy alive" $?

# --- 4. order is preserved when the next response is already waiting -------------------
out=$(python3 tests/integration/pipeline_client.py 18090 fpgate.test 18443 "$T/ca/ca.crt" \
        "/img/photo.jpg=$JPG" "/echo=-" "/img/pic.png=$PNG" "/chunked=$(printf 'hello chunked world' | sha)" \
        "/img/photo.jpg?mode=chunked=$JPG" 2>&1)
echo "$out" | sed 's/^/     /'
[ "$(echo "$out" | grep -c '^ok')" -eq 5 ]; check "pipelined image/text/image/chunked/chunked-image on one tunnel: all intact, in order" $?

# --- 5. several clients at once -------------------------------------------------------------
H3="-x http://127.0.0.1:18092"
WAITS=""
for n in 1 2 3 4 5 6 7 8; do
    ( curl -s $H3 "$O/img/photo.jpg" | sha > "$T/par.$n" ) & WAITS="$WAITS $!"
done
wait $WAITS
ok=0; for n in 1 2 3 4 5 6 7 8; do [ "$(cat "$T/par.$n")" = "$JPG" ] && ok=$((ok + 1)); done
[ $ok -eq 8 ]; check "8 concurrent clients: all 8 images intact ($ok)" $?

# --- 6. more jobs than the queue holds --------------------------------------------------------
WAITS=""
for n in $(seq 1 45); do
    ( curl -s $H3 "$O/img/photo.jpg" | sha > "$T/many.$n" ) & WAITS="$WAITS $!"
done
wait $WAITS
ok=0; for n in $(seq 1 45); do [ "$(cat "$T/many.$n")" = "$JPG" ] && ok=$((ok + 1)); done
[ $ok -eq 45 ]; check "45 concurrent images (queue holds 32): all delivered intact ($ok)" $?
busy=$(grep -c 'skipped=busy' "$T/p3.log")
[ "$busy" -ge 1 ]; check "images beyond the queue are passed through unexamined and counted ($busy)" $?

# --- 7. clean shutdown with nothing left over ---------------------------------------------------
kill -0 $P1 $P3 2>/dev/null; check "proxies alive after all of the above" $?
kill -TERM $P1 $P2 $P3 2>/dev/null; wait $P1 $P2 $P3 2>/dev/null
if grep -E "AddressSanitizer|runtime error|LeakSanitizer|ThreadSanitizer" "$T"/p*.log; then
    echo "FAIL sanitizer report"; fail=1
else
    echo "PASS no sanitizer reports, no leaks at shutdown"
fi
exit $fail
