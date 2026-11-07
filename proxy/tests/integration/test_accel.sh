#!/bin/bash
# DES-3.3: judging images with fpgate-accel (here the mock): threshold, replacement,
# failure policy, degraded mode and recovery, full queue, cancellation.
# Usage: tests/integration/test_accel.sh [proxy-binary] [prep_tool] [mock_accel]
BIN=${1:-build/asan/fpgate-proxy}
PREP=${2:-build/prep_tool}
MOCKC=${3:-build/mock_accel}
T=$(mktemp -d)
fail=0
PIDS=""

curl() { command curl --max-time 25 "$@"; }
check() { # name, exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
cleanup() { kill $PIDS $MOCK1 $MOCK2 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
sha() { sha256sum | cut -d' ' -f1; }
fsha() { sha256sum "$1" | cut -d' ' -f1; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }
png_info() { python3 -c "
import io, sys
from PIL import Image
im = Image.open(io.BytesIO(sys.stdin.buffer.read())); print(im.format, im.size[0], im.size[1])"; }
mockcfg() { printf '%s' "$2" > "$T/$1.tmp"; mv "$T/$1.tmp" "$T/$1.json"; }
logcount() { grep -c -E "$2" "$T/$1.log"; }
wait_for() { # seconds, command...: polls until the command succeeds
    local end=$((SECONDS + $1)); shift
    while [ $SECONDS -lt $end ]; do "$@" && return 0; sleep 0.3; done; return 1
}

# --- fixtures -----------------------------------------------------------------------------
mkdir -p "$T/img"
python3 - "$T/img" <<'EOF'
import random, sys
from PIL import Image
d = sys.argv[1]
Image.new("RGB", (800, 600), (255, 255, 255)).save(d + "/white.png")
Image.new("RGB", (640, 480), (0, 0, 0)).save(d + "/black.png")
Image.new("RGB", (300, 300), (128, 128, 128)).save(d + "/gray128.png")    # score 0.50196
Image.new("RGB", (300, 300), (100, 100, 100)).save(d + "/gray100.png")    # score 0.39216
Image.new("RGB", (40, 30), (255, 255, 255)).save(d + "/tiny.png")
Image.new("RGB", (500, 400), (255, 255, 255)).save(d + "/white.webp", quality=90)
Image.new("RGB", (200, 150), (255, 255, 255)).save(d + "/anim.gif")
Image.new("RGB", (900, 700), (255, 255, 255)).save(d + "/white.jpg", quality=90)
data = open(d + "/white.jpg", "rb").read()
open(d + "/trunc.jpg", "wb").write(data[: len(data) // 2])
open(d + "/page.txt", "wb").write(b"hello " * 50)
Image.new("RGB", (64, 64), (200, 30, 30)).save(d + "/custom.png")
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

# Two mock accelerators: #1 answers fast, #2 takes a second per image.
mockcfg m1 '{"score_from_pixels": true}'
mockcfg m2 '{"score_from_pixels": true, "latency": 1.0}'
start_mock() { python3 tests/mock_accel/mock_accel.py --socket "$T/$1.sock" --config "$T/$1.json" \
                   --log "$T/$1.mock" 2>/dev/null & }
start_mock m1; MOCK1=$!
start_mock m2; MOCK2=$!

CA="-C $T/ca/ca.crt -K $T/ca/ca.key -P $T/pass -B $T/oca.crt"
# P1 defaults (threshold 0.5)         P2 short timeout, for hang/restart tests
# P3 slow accelerator, queue of 2     P4 fail-open + threshold 0.6
# P5 unexamined = block, custom image P6 corrupt = allow
$BIN -p 18090 $CA --accel $T/m1.sock --accel-timeout-ms 5000 2>$T/p1.log & P1=$!
$BIN -p 18091 $CA --accel $T/m1.sock --accel-timeout-ms 1000 2>$T/p2.log & P2=$!
$BIN -p 18092 $CA --accel $T/m2.sock --queue-max 2 --accel-timeout-ms 8000 2>$T/p3.log & P3=$!
$BIN -p 18093 $CA --accel $T/m1.sock --threshold 0.6 --on-failure allow 2>$T/p4.log & P4=$!
$BIN -p 18094 $CA --accel $T/m1.sock --on-unexamined block --substitute $T/img/custom.png 2>$T/p5.log & P5=$!
$BIN -p 18095 $CA --accel $T/m1.sock --on-corrupt allow 2>$T/p6.log & P6=$!
PIDS="$PIDS $P1 $P2 $P3 $P4 $P5 $P6"
sleep 2.5
O=http://127.0.0.1:18081
Q=https://fpgate.test:18443
hp() { echo "-x http://127.0.0.1:$1"; }
sp() { echo "-x http://127.0.0.1:$1 --cacert $T/ca/ca.crt"; }

# --- 1. the verdict --------------------------------------------------------------------------------
BLACK=$(fsha "$T/img/black.png")
curl -s $(hp 18090) "$O/img/black.png" | cmp -s - "$T/img/black.png"
check "score 0.0 < threshold: the original is delivered unchanged" $?
grep -q 'VERDICT host=127.0.0.1 verdict=allow reason=below_threshold score=0.000' "$T/p1.log"
check "the allowed image is logged with its score" $?

curl -s -D "$T/h.txt" $(hp 18090) "$O/img/white.png?etag=1" -o "$T/blocked.png"
head -1 "$T/h.txt" | grep -q ' 200 '; check "blocked image: still a 200, not an error page" $?
grep -qi '^content-type: image/png' "$T/h.txt"; check "blocked image: Content-Type is the replacement's" $?
grep -qi '^cache-control: no-store' "$T/h.txt"; check "blocked image: Cache-Control no-store" $?
grep -qi '^x-fpgate-block: nsfw' "$T/h.txt"; check "blocked image: X-FPGate-Block says why" $?
! grep -qiE '^(etag|last-modified|content-encoding):' "$T/h.txt"; check "blocked image: the original's ETag / Last-Modified are not carried over" $?
[ "$(grep -i '^content-length:' "$T/h.txt" | tr -dc 0-9)" -eq "$(stat -c %s "$T/blocked.png")" ]
check "blocked image: Content-Length matches the replacement body" $?
[ "$(png_info < "$T/blocked.png")" = "PNG 800 600" ]; check "blocked image: replacement is a PNG with the original's 800x600" $?
! cmp -s "$T/blocked.png" "$T/img/white.png"; check "blocked image: the original bytes are not delivered" $?
[ "$(stat -c %s "$T/blocked.png")" -lt 5000 ]; check "the replacement is small ($(stat -c %s "$T/blocked.png") bytes)" $?
grep -q 'VERDICT host=127.0.0.1 verdict=block reason=nsfw score=1.000' "$T/p1.log"; check "the block is logged" $?

curl -s -D "$T/h2.txt" $(sp 18090) "$Q/img/tiny.png" -o "$T/blocked2.png"
[ "$(png_info < "$T/blocked2.png")" = "PNG 40 30" ]; check "HTTPS: a 40x30 image gets a 40x30 replacement" $?
curl -s $(sp 18090) "$Q/img/white.webp" -o "$T/b3.png"; [ "$(png_info < "$T/b3.png" | cut -d' ' -f1)" = PNG ]
check "HTTPS: a WebP rated NSFW is replaced too" $?
curl -s $(hp 18090) "$O/img/white.jpg?mode=chunked" -o "$T/b4.png"; got=$(png_info < "$T/b4.png" 2>&1 | tail -1)
[ "$got" = "PNG 800 622" ]; check "a chunked 900x700 JPEG rated NSFW is replaced by a PNG scaled to 800x622, same shape (got: $got)" $?
curl -s $(hp 18090) "$O/img/white.png?mode=close" -o "$T/b5.png"; [ "$(png_info < "$T/b5.png" | cut -d' ' -f1)" = PNG ]
check "a close-delimited image rated NSFW is replaced" $?

# --- 2. threshold edges -------------------------------------------------------------------------------
curl -s $(hp 18090) "$O/img/gray128.png" -o "$T/g.out"; ! cmp -s "$T/g.out" "$T/img/gray128.png"
check "score 0.502 >= 0.50: replaced (the comparison is >=)" $?
curl -s $(hp 18090) "$O/img/gray100.png" | cmp -s - "$T/img/gray100.png"; check "score 0.392 < 0.50: delivered" $?
curl -s $(hp 18093) "$O/img/gray128.png" | cmp -s - "$T/img/gray128.png"
check "--threshold 0.6: the same 0.502 image is delivered" $?
curl -s $(hp 18093) "$O/img/white.png" -o "$T/g2.out"; ! cmp -s "$T/g2.out" "$T/img/white.png"
check "--threshold 0.6: a 1.0 image is still replaced" $?

# --- 3. what reaches the accelerator is exactly the preprocessed tensor --------------------------
"$PREP" "$T/img/black.png" 448 crop 0 "$T/black.rgb" >/dev/null
grep -q "$(fsha "$T/black.rgb")" "$T/m1.mock"; check "the accelerator received byte-for-byte the tensor prep_tool makes (RF-03)" $?
grep -q '"w": 448, "h": 448, "bytes": 602112' "$T/m1.mock"; check "request carried 448x448x3 = 602112 bytes of RGB" $?

# --- 4. the rest of the traffic is untouched ---------------------------------------------------------------
curl -s $(hp 18090) "$O/img/page.txt" | cmp -s - "$T/img/page.txt"; check "text passes without being judged" $?
out=$(python3 tests/integration/pipeline_client.py 18090 fpgate.test 18443 "$T/ca/ca.crt" \
        "/img/white.png=-" "/echo=-" "/img/black.png=$BLACK" "/chunked=$(printf 'hello chunked world' | sha)" 2>&1)
[ "$(echo "$out" | grep -c '^ok')" -eq 4 ]
check "pipelined [blocked image, text, allowed image, text] on one tunnel: all answered, in order" $?

# --- 5. unexaminable and damaged images, per policy ---------------------------------------------------
curl -s $(hp 18090) "$O/img/anim.gif" | cmp -s - "$T/img/anim.gif"; check "default: a GIF is delivered (--on-unexamined allow)" $?
grep -q 'verdict=allow reason=unsupported_format' "$T/p1.log"; check "the GIF decision is logged" $?
curl -s $(hp 18094) "$O/img/anim.gif" | cmp -s - "$T/img/custom.png"
check "--on-unexamined block + --substitute: the GIF is replaced by the custom image, as is" $?
curl -s -D "$T/h5.txt" $(hp 18094) "$O/img/white.png" -o /dev/null; grep -qi '^content-type: image/png' "$T/h5.txt"
check "--substitute: the custom image is sent with its own type" $?
curl -s $(hp 18090) "$O/img/trunc.jpg" -o "$T/c.out"; ! cmp -s "$T/c.out" "$T/img/trunc.jpg"
check "default: a truncated JPEG is blocked (--on-corrupt block)" $?
grep -q 'verdict=block reason=corrupt' "$T/p1.log"; check "the corrupt block is logged" $?
curl -s $(hp 18095) "$O/img/trunc.jpg" | cmp -s - "$T/img/trunc.jpg"; check "--on-corrupt allow: the truncated JPEG is delivered" $?

# --- 6. the accelerator misbehaves (proxy #2, 1 s timeout) --------------------------------------------------------
mockcfg m1 '{"score_from_pixels": true, "fail": "error"}'
curl -s $(hp 18091) "$O/img/black.png" -o "$T/e.out"; ! cmp -s "$T/e.out" "$T/img/black.png"
check "accelerator error: the image is blocked (RF-14, default policy)" $?
grep -q 'verdict=block reason=accel_error' "$T/p2.log"; check "logged as accel_error" $?
grep -q 'accel: degraded mode' "$T/p2.log"; check "a hardware error puts the proxy in degraded mode" $?
s=$SECONDS; curl -s $(hp 18091) "$O/img/black.png" -o "$T/e2.out"; d=$((SECONDS - s))
! cmp -s "$T/e2.out" "$T/img/black.png" && [ $d -le 2 ]; check "degraded: the next image is blocked at once (${d}s), without waiting for the accelerator" $?
grep -q 'verdict=block reason=accel_down' "$T/p2.log"; check "logged as accel_down" $?
t=$(curl -s -o /dev/null -w '%{time_total}' $(hp 18091) "$O/echo"); lt "$t" 0.5
check "degraded: text is not delayed (${t}s)" $?

mockcfg m1 '{"score_from_pixels": true}'
ok() { curl -s $(hp 18091) "$O/img/black.png" | cmp -s - "$T/img/black.png"; }
wait_for 12 ok; check "the accelerator recovers: probing brings the proxy back and the image is delivered" $?
grep -q 'accel: answering again, back to normal' "$T/p2.log"; check "the recovery is logged" $?

mockcfg m1 '{"score_from_pixels": true, "fail": "close"}'
curl -s $(hp 18091) "$O/img/black.png" -o "$T/e3.out"; ! cmp -s "$T/e3.out" "$T/img/black.png"
check "connection dropped mid-request: the image is blocked" $?

mockcfg m1 '{"score_from_pixels": true}'
wait_for 12 ok; check "recovers again after the connection drop" $?

mockcfg m1 '{"score_from_pixels": true, "fail": "hang"}'
s=$SECONDS; curl -s $(hp 18091) "$O/img/black.png" -o "$T/h1.out"; d=$((SECONDS - s))
! cmp -s "$T/h1.out" "$T/img/black.png" && [ $d -ge 1 ] && [ $d -le 4 ]
check "accelerator hangs: the image is blocked after the 1 s timeout (${d}s)" $?
grep -q 'verdict=block reason=accel_timeout' "$T/p2.log"; check "logged as accel_timeout" $?
sleep 3
ok2() { ! grep -q 'back to normal' <(tail -n 3 "$T/p2.log"); }
curl -s $(hp 18091) "$O/img/black.png" -o "$T/h2.out"; ! cmp -s "$T/h2.out" "$T/img/black.png"
check "still stuck (status says BUSY): the proxy stays degraded" $?
# the stuck mock holds its lock forever: replace it, as an operator would restart the service
kill $MOCK1; wait $MOCK1 2>/dev/null; rm -f "$T/m1.sock"
mockcfg m1 '{"score_from_pixels": true}'
start_mock m1; MOCK1=$!
wait_for 15 ok; check "after the accelerator is restarted the proxy recovers on its own" $?

# --- 7. fail-open on request (proxy #4) --------------------------------------------------------------------------------
mockcfg m1 '{"score_from_pixels": true, "fail": "error"}'
curl -s $(hp 18093) "$O/img/white.png" | cmp -s - "$T/img/white.png"
check "--on-failure allow: with a broken accelerator the image is delivered" $?
grep -q 'verdict=allow reason=accel_error' "$T/p4.log"; check "and the decision is logged as allow" $?
mockcfg m1 '{"score_from_pixels": true}'

# --- 8. more images than the accelerator can take (proxy #3: 1 s each, 2 may wait) ---------------------------------
W=""
for n in 1 2 3 4 5 6; do
    ( curl -s $(hp 18092) "$O/img/white.png?n=$n" -o "$T/q.$n" ) & W="$W $!"
done
t0=$SECONDS
sleep 0.5
t=$(curl -s -o /dev/null -w '%{time_total}' $(hp 18092) "$O/echo"); lt "$t" 0.5
check "text is not delayed while images queue for the accelerator (${t}s)" $?
wait $W
ok=0; for n in 1 2 3 4 5 6; do [ -s "$T/q.$n" ] && ok=$((ok + 1)); done
[ $ok -eq 6 ]; check "all 6 concurrent images were answered ($ok)" $?
[ "$(logcount p3 'reason=queue_full')" -eq 3 ]; check "1 in flight + 2 waiting: the other 3 were blocked as queue_full ($(logcount p3 'reason=queue_full'))" $?
[ "$(logcount p3 'reason=nsfw')" -eq 3 ]; check "the 3 that fit were judged by the accelerator" $?

# --- 9. the client gives up while its image waits ------------------------------------------------------------------
curl -s --max-time 0.4 $(hp 18092) "$O/img/white.png?n=cancel" -o /dev/null
sleep 2.5
[ "$(curl -s $(hp 18092) "$O/echo" | head -1)" = "GET /echo HTTP/1.1" ]; check "client aborted while its image was at the accelerator: proxy keeps serving" $?
curl -s $(hp 18092) "$O/img/white.png?n=after" -o "$T/after.out"; ! cmp -s "$T/after.out" "$T/img/white.png"
check "and the next image is judged normally" $?

# --- 10. the C stand-in that runs on the board speaks the same protocol --------------------------------------
"$MOCKC" "$T/c.sock" --mean --latency-ms 50 --quiet 2>/dev/null & MOCKC_PID=$!
sleep 0.5
$BIN -p 18096 $CA --accel $T/c.sock 2>$T/p7.log & P7=$!
sleep 1.5
curl -s $(hp 18096) "$O/img/black.png" | cmp -s - "$T/img/black.png"; check "C stand-in: a dark image is delivered" $?
curl -s $(hp 18096) "$O/img/white.png" -o "$T/cw.out"; ! cmp -s "$T/cw.out" "$T/img/white.png" && [ "$(png_info < "$T/cw.out")" = "PNG 800 600" ]
check "C stand-in: a bright image is replaced" $?
grep -q 'verdict=block reason=nsfw score=1.000' "$T/p7.log"; check "C stand-in: the verdict is logged with its score" $?
kill $P7 $MOCKC_PID 2>/dev/null; wait $P7 $MOCKC_PID 2>/dev/null

# --- 11. nothing leaks, nothing crashes ----------------------------------------------------------------------------------
kill -0 $P1 $P2 $P3 $P4 $P5 $P6 2>/dev/null; check "all proxies alive" $?
kill -TERM $P1 $P2 $P3 $P4 $P5 $P6 2>/dev/null; wait $P1 $P2 $P3 $P4 $P5 $P6 2>/dev/null
if grep -E "AddressSanitizer|runtime error|LeakSanitizer|ThreadSanitizer" "$T"/p*.log; then
    echo "FAIL sanitizer report"; fail=1
else
    echo "PASS no sanitizer reports, no leaks at shutdown"
fi
exit $fail
