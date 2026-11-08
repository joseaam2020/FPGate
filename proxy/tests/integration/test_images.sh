#!/bin/bash
# Image extraction through the proxy (DES-3.2): plain HTTP and intercepted HTTPS.
# Usage: tests/integration/test_images.sh [binary]
BIN=${1:-build/asan/fpgate-proxy}
T=$(mktemp -d)
fail=0
PIDS=""

curl() { command curl --max-time 20 "$@"; }
check() { # name, exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
cleanup() { kill $PIDS 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT

sha() { sha256sum | cut -d' ' -f1; }
fsha() { sha256sum "$1" | cut -d' ' -f1; }
# number of log lines of the first proxy matching a pattern
count() { grep -c -E "$1" "$T/p1.log"; }

# --- fixtures ---------------------------------------------------------------
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
photo(640, 480, 2).save(d + "/prog.jpg", quality=85, progressive=True)
photo(1600, 1200, 3).save(d + "/large.jpg", quality=95)
rgba = photo(300, 200, 4).convert("RGBA"); rgba.putalpha(128); rgba.save(d + "/pic.png")
photo(200, 200, 5).save(d + "/anim.gif")
photo(200, 200, 6).save(d + "/pic.webp")
data = open(d + "/photo.jpg", "rb").read()
open(d + "/trunc.jpg", "wb").write(data[: len(data) // 2])
open(d + "/as.bin", "wb").write(data)                  # a JPEG served as octet-stream
open(d + "/fake.jpg", "wb").write(b"not an image at all " * 200)
open(d + "/tiny.bin", "wb").write(b"\x01\x02\x03")
open(d + "/page.txt", "wb").write(b"hello " * 500)
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
$BIN -p 18090 -C "$T/ca/ca.crt" -K "$T/ca/ca.key" -P "$T/pass" -B "$T/oca.crt" 2>"$T/p1.log" & P1=$!
# Second instance with a 100 KB analysis limit, third with inspection off.
$BIN -p 18091 -L 100000 2>"$T/p2.log" & P2=$!
$BIN -p 18092 -N 2>"$T/p3.log" & P3=$!
$BIN -p 18093 -V -C "$T/ca/ca.crt" -K "$T/ca/ca.key" -P "$T/pass" -B "$T/oca.crt" 2>"$T/p4.log" & P4=$!
python3 tests/integration/tls_echo.py 18444 "$T/o.crt" "$T/o.key" & PIDS="$PIDS $!"
PIDS="$PIDS $P1 $P2 $P3 $P4"
sleep 2

H="-x http://127.0.0.1:18090"
S="-x http://127.0.0.1:18090 --cacert $T/ca/ca.crt"
O=http://127.0.0.1:18081
Q=https://fpgate.test:18443

# --- 1. delivery is byte-identical, whatever the framing ----------------------
for f in photo.jpg prog.jpg pic.png large.jpg; do
    want=$(fsha "$T/img/$f")
    for mode in cl chunked close; do
        got=$(curl -s $H "$O/img/$f?mode=$mode" | sha)
        [ "$got" = "$want" ]; check "HTTP  $f ($mode) delivered unchanged" $?
        got=$(curl -s $S "$Q/img/$f?mode=$mode" | sha)
        [ "$got" = "$want" ]; check "HTTPS $f ($mode) delivered unchanged" $?
    done
done

# --- 2. RF-02: every jpg/png is detected and decoded ---------------------------
jpg=$(count 'IMG host=[^ ]+ kind=jpeg src=(800x600|640x480|1600x1200) .*prep=')
png=$(count 'IMG host=[^ ]+ kind=png src=300x200 .*prep=')
[ "$jpg" -eq 18 ]; check "RF-02: 18 JPEG responses detected and decoded (got $jpg)" $?
[ "$png" -eq 6 ]; check "RF-02: 6 PNG responses detected and decoded (got $png)" $?
count 'IMG host=fpgate.test' >/dev/null; [ "$(count 'IMG host=fpgate.test')" -eq 12 ]
check "HTTPS images are attributed to the CONNECT host" $?

# --- 3. wrong or missing headers: the bytes decide ---------------------------------
before=$(count 'kind=jpeg .*prep=')
got=$(curl -s $H "$O/img/as.bin" | sha)
[ "$got" = "$(fsha "$T/img/photo.jpg")" ]; check "JPEG served as octet-stream: delivered" $?
[ "$(count 'kind=jpeg .*prep=')" -eq $((before + 1)) ]; check "JPEG served as octet-stream: detected by magic bytes" $?
curl -s $H "$O/img/photo.jpg?ct=none" | cmp -s - "$T/img/photo.jpg"; check "JPEG without Content-Type: delivered" $?
[ "$(count 'kind=jpeg .*prep=')" -eq $((before + 2)) ]; check "JPEG without Content-Type: detected by magic bytes" $?
curl -s $H "$O/img/pic.png?ct=image/jpeg" | cmp -s - "$T/img/pic.png"; check "PNG labelled image/jpeg: delivered" $?
[ "$(count 'kind=png src=300x200 .*prep=')" -eq 7 ]; check "PNG labelled image/jpeg: decoded as PNG (magic wins)" $?

# --- 4. things that are not decodable pass through, and are accounted for ---------
curl -s $H "$O/img/anim.gif" | cmp -s - "$T/img/anim.gif"; check "GIF delivered unchanged" $?
curl -s $H "$O/img/pic.webp" | cmp -s - "$T/img/pic.webp"; check "WebP delivered unchanged" $?
[ "$(count 'kind=other .*error=unsupported_format')" -eq 1 ]; check "GIF logged as an unsupported format" $?
[ "$(count 'kind=webp src=200x200 .*prep=')" -eq 1 ]; check "WebP is decoded like JPEG and PNG" $?
for mode in cl chunked close; do
    curl -s $S "$Q/img/pic.webp?mode=$mode" | cmp -s - "$T/img/pic.webp"; check "HTTPS WebP ($mode) delivered unchanged" $?
done
[ "$(count 'kind=webp src=200x200 .*prep=')" -eq 4 ]; check "all four WebP responses were decoded" $?
curl -s $H "$O/img/trunc.jpg" | cmp -s - "$T/img/trunc.jpg"; check "truncated JPEG delivered unchanged" $?
[ "$(count 'kind=jpeg .*error=corrupt')" -ge 1 ]; check "truncated JPEG logged as corrupt" $?
curl -s $H "$O/img/fake.jpg" | cmp -s - "$T/img/fake.jpg"; check "image/jpeg that is not an image: delivered" $?
[ "$(count 'error=not_an_image')" -ge 1 ]; check "image/jpeg that is not an image: logged" $?

# --- 5. responses that must not be touched ------------------------------------------
curl -s $H "$O/img/page.txt" | cmp -s - "$T/img/page.txt"; check "text/plain delivered unchanged" $?
curl -s $H "$O/img/tiny.bin" | cmp -s - "$T/img/tiny.bin"; check "3-byte octet-stream delivered unchanged" $?
h=$(curl -s -I $H "$O/img/photo.jpg"); echo "$h" | grep -qi '^content-length: '; check "HEAD answers with headers and no body (no hang)" $?
code=$(curl -s -o /dev/null -w '%{http_code}' $H "$O/img/photo.jpg?status=304"); [ "$code" = 304 ]; check "304 passes" $?
before=$(count 'IMG .*(src=|error=)')
curl -s $H "$O/img/photo.jpg?status=206" | cmp -s - "$T/img/photo.jpg"; check "206 partial content delivered unchanged" $?
[ "$(count 'IMG .*(src=|error=)')" -eq "$before" ]; check "206 is not analysed (policy: pass without analysis)" $?
grep -q 'type=image/jpeg skipped=partial' "$T/p1.log"; check "206: the skip is logged with its reason (partial)" $?
curl -s $H "$O/img/photo.jpg?enc=gzip" | cmp -s - "$T/img/photo.jpg"; check "Content-Encoding image delivered unchanged" $?
[ "$(count 'IMG .*(src=|error=)')" -eq "$before" ]; check "Content-Encoding image is not analysed" $?
grep -q 'type=image/jpeg skipped=encoded' "$T/p1.log"; check "Content-Encoding: the skip is logged with its reason (encoded)" $?

# --- 6. keep-alive: many responses on one intercepted connection --------------------
n0=$(count 'IMG .*(src=|error=)')
out=$(curl -s $S "$Q/img/photo.jpg" "$Q/echo" "$Q/img/pic.png?mode=chunked" "$Q/img/prog.jpg" "$Q/chunked" | sha)
want=$( { cat "$T/img/photo.jpg"; echo; } >/dev/null; \
        { cat "$T/img/photo.jpg"; \
          curl -s $S "$Q/echo"; cat "$T/img/pic.png"; cat "$T/img/prog.jpg"; printf 'hello chunked world'; } | sha)
[ "$out" = "$want" ]; check "keep-alive: image, text, chunked image, image, chunked text in one tunnel" $?
[ "$(count 'IMG .*(src=|error=)')" -eq $((n0 + 3)) ]; check "keep-alive: the three images in that tunnel were all analysed" $?

# --- 7. the reactor is not blocked while an image is held ------------------------------
curl -s $H "$O/img/photo.jpg?slow=1" -o "$T/slow.out" & SLOW=$!
sleep 0.7
t=$(curl -s -o /dev/null -w '%{time_total}' $H "$O/echo")
ok=$(awk -v t="$t" 'BEGIN { print (t < 0.5) ? 0 : 1 }')
[ "$ok" -eq 0 ]; check "other traffic is not delayed by a held image (echo took ${t}s)" $?
wait $SLOW
cmp -s "$T/slow.out" "$T/img/photo.jpg"; check "the slowly delivered image arrives intact" $?

# --- 8. limits and switches -----------------------------------------------------------------
H2="-x http://127.0.0.1:18091"
curl -s $H2 "$O/img/large.jpg" | cmp -s - "$T/img/large.jpg"; check "image over -L limit delivered unchanged" $?
! grep -qE 'IMG .*(src=|error=)' "$T/p2.log"; check "image over -L limit is streamed, not analysed" $?
grep -q 'type=image/jpeg skipped=oversize' "$T/p2.log"; check "image over -L limit: the skip is logged with its reason (oversize)" $?
H3="-x http://127.0.0.1:18092"
curl -s $H3 "$O/img/photo.jpg?mode=chunked" | cmp -s - "$T/img/photo.jpg"; check "-N: image delivered unchanged" $?
! grep -q 'IMG ' "$T/p3.log"; check "-N: no inspection at all" $?

# --- 9. ambiguous framing is refused, not guessed ----------------------------------------------
curl -s $H "$O/img/photo.jpg?mode=smuggle" -o "$T/sm.out"; rc=$?
{ [ $rc -ne 0 ] || ! cmp -s "$T/sm.out" "$T/img/photo.jpg"; }; check "Content-Length + Transfer-Encoding: connection refused" $?
grep -q 'framing error' "$T/p1.log"; check "framing error logged" $?


# --- 10. the log says which page an image belongs to (-V) --------------------------------------
H4="-x http://127.0.0.1:18093"
S4="-x http://127.0.0.1:18093 --cacert $T/ca/ca.crt"
curl -s -e "http://news.example.org/story?id=7" $H4 "$O/img/photo.jpg?token=secret" -o /dev/null
grep -q 'IMG host=127.0.0.1 kind=jpeg .* path=/img/photo.jpg ref=news.example.org/story$' "$T/p4.log"
check "-V: image line carries its path and the page that requested it (plain HTTP)" $?
! grep -qE 'secret|id=7' "$T/p4.log"; check "-V: query strings are never logged" $?
curl -s -e "https://fpgate.test/wiki/Alps?x=1" $S4 "$Q/img/pic.png" -o /dev/null
grep -q 'IMG host=fpgate.test kind=png .* path=/img/pic.png ref=fpgate.test/wiki/Alps$' "$T/p4.log"
check "-V: same over an intercepted HTTPS tunnel" $?
curl -s $S4 "$Q/echo" "$Q/img/photo.jpg" -o /dev/null -o /dev/null
grep -qE 'RESP host=fpgate.test status=200 type=- bytes=[0-9]+ action=(stream|sniffed-not-image) path=/echo' "$T/p4.log"
check "-V: non-image responses get a RESP line" $?
grep -q 'RESP host=fpgate.test status=200 type=image/jpeg .* action=stream' "$T/p4.log" || \
    grep -q 'IMG host=fpgate.test kind=jpeg .* path=/img/photo.jpg' "$T/p4.log"
check "-V: the image after a text response in the same tunnel is attributed to its own path" $?
! grep -q 'path=' "$T/p1.log"; check "without -V nothing about paths is logged" $?

# --- 11. TLS that is not HTTP must pass through, not be cut -------------------------------------
fe0=$(grep -c 'framing error' "$T/p1.log")
python3 tests/integration/opaque_client.py 18090 fpgate.test 18444 "$T/ca/ca.crt" >"$T/opq.out" 2>&1
grep -q '^ok' "$T/opq.out"; check "non-HTTP protocol inside TLS passes through untouched ($(tail -1 "$T/opq.out" | cut -c1-60))" $?
[ "$(grep -c 'framing error' "$T/p1.log")" -eq "$fe0" ]; check "non-HTTP protocol is not reported as a framing error" $?


# --- 12. Accept loses image/avif, nothing else changes ---------------------------------------------
A='Accept: image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
want='Accept: image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
out=$(curl -s $H "$O/echo" -H "$A" -H 'X-Keep: 1')
echo "$out" | grep -qxF "$want" && echo "$out" | grep -qF 'X-Keep: 1'
check "plain HTTP: Accept loses image/avif, other headers intact" $?
out=$(curl -s $S "$Q/echo" -H "$A" -H 'X-Keep: 1')
echo "$out" | grep -qxF "$want" && echo "$out" | grep -qF 'X-Keep: 1'
check "intercepted HTTPS: Accept loses image/avif, other headers intact" $?
out=$(curl -s $S "$Q/echo" -H 'Accept: image/webp,*/*')
echo "$out" | grep -qxF 'Accept: image/webp,*/*'; check "an Accept without AVIF is forwarded untouched" $?
out=$(curl -s $S "$Q/echo" -H 'Accept: image/avif')
echo "$out" | grep -qxF 'Accept: */*'; check "an Accept that only allowed AVIF becomes */*" $?
[ "$(curl -s $S -d 'payload-xyz' -H "$A" "$Q/post")" = "payload-xyz" ]
check "a POST whose head was shortened keeps its body intact" $?
out=$(python3 tests/integration/pipeline_client.py 18090 fpgate.test 18443 "$T/ca/ca.crt" \
        "/img/pic.webp=$(fsha "$T/img/pic.webp")" "/echo=-" "/img/photo.jpg=$(fsha "$T/img/photo.jpg")" 2>&1)
[ "$(echo "$out" | grep -c '^ok')" -eq 3 ]; check "pipelined WebP, text and JPEG on one tunnel are all intact" $?

kill -0 $P1 2>/dev/null; check "proxy still alive" $?
if grep -E "AddressSanitizer|runtime error|LeakSanitizer" "$T"/p*.log; then echo "FAIL sanitizer report"; fail=1; fi
exit $fail
