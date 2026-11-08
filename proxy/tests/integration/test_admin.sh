#!/bin/bash
# DES-3.4: the administration interface (HTTPS) and http://fpgate.setup.
# Login, sessions, threshold, stand-in image, block log, public certificate page.
# Usage: tests/integration/test_admin.sh [proxy-binary] [mock_accel]
BIN=${1:-build/asan/fpgate-proxy}
MOCKC=${2:-build/mock_accel}
T=$(mktemp -d)
fail=0
PIDS=""

curl() { command curl --max-time 25 "$@"; }
check() { # name, exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
cleanup() { [ $fail -ne 0 ] && [ -f "$T/p.log" ] && { echo "--- proxy log (tail) ---"; tail -25 "$T/p.log"; }; kill $PIDS $MOCK 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
jget() { python3 -c "
import json, sys
d = json.load(sys.stdin)
for k in sys.argv[1].split('.'): d = d[int(k)] if k.isdigit() else d[k]
print(d)" "$1"; }

mkdir -p "$T/img"
python3 - "$T/img" <<'EOF'
import sys
from PIL import Image
d = sys.argv[1]
Image.new("RGB", (800, 600), (255, 255, 255)).save(d + "/white.png")
Image.new("RGB", (640, 480), (0, 0, 0)).save(d + "/black.png")
Image.new("RGB", (300, 300), (128, 128, 128)).save(d + "/gray128.png")
Image.new("RGB", (64, 64), (200, 30, 30)).save(d + "/custom.png")
Image.new("RGB", (64, 64), (30, 30, 200)).save(d + "/custom.jpg", quality=90)
open(d + "/fake.png", "wb").write(b"this is not an image" * 10)
open(d + "/big.png", "wb").write(open(d + "/custom.png", "rb").read() + b"\0" * (3 * 1024 * 1024))
EOF
export IMG_DIR="$T/img"

export FPGATE_CA_PASS=testpass
sh tools/gen_ca.sh "$T/ca" >/dev/null 2>&1
printf 'testpass\n' > "$T/pass"
python3 tests/integration/origin.py 18081 & PIDS="$PIDS $!"
$MOCKC "$T/m.sock" --mean --quiet 2>/dev/null & MOCK=$!

# --- 0. the credentials file -----------------------------------------------------------------------------
printf 'short\n' | $BIN --set-admin-password "$T/bad" 2>"$T/e.log"; rc=$?
[ $rc -ne 0 ] && [ ! -e "$T/bad" ] && grep -q '8 to' "$T/e.log"
check "a password under 8 characters is refused and nothing is written" $?
printf 'correct horse\n' | $BIN --set-admin-password "$T/admin.pass" 2>/dev/null
check "--set-admin-password writes the credentials file" $?
[ "$(stat -c %a "$T/admin.pass")" = 600 ]; check "the credentials file is mode 600" $?
! grep -q 'correct horse' "$T/admin.pass"; check "the password is not stored in the file, only a salted hash" $?
[ "$(cut -d: -f1 "$T/admin.pass")" = admin ] && [ "$(cut -d: -f2 "$T/admin.pass")" -ge 10000 ]
check "format user:iterations:salt:hash with PBKDF2 iterations" $?

# --- start -----------------------------------------------------------------------------------------------------
CA="-C $T/ca/ca.crt -K $T/ca/ca.key -P $T/pass -B $T/ca/ca.crt"
cat > "$T/fpgate.toml" <<'EOF'
# written by hand: comments and unrelated keys must survive the admin page
[proxy]
puerto = 8080

[clasificacion]
umbral = 0.60   # strict-ish
timeout_accel_ms = 10000
EOF
$BIN -p 18090 $CA --accel "$T/m.sock" --block-log "$T/blocks.db" --config "$T/fpgate.toml" \
    --admin-credentials "$T/admin.pass" --admin-port 18443 2>"$T/p.log" & P=$!
PIDS="$PIDS $P"
sleep 2.5
A="https://127.0.0.1:18443"
CURL="--cacert $T/ca/ca.crt"
J="Content-Type: application/json"

grep -q 'admin interface' "$T/p.log"; check "the interface announces itself at start-up" $?
grep -q 'umbral\|threshold 0.60' "$T/p.log"; check "the threshold in the file (0.60) overrides the default" $?

# --- 1. HTTPS with a certificate from the FPGate CA ---------------------------------------------------------------
curl -s $CURL "$A/" -o "$T/login.html" -D "$T/h.txt"
head -1 "$T/h.txt" | grep -q ' 200 '; check "the login page is served over HTTPS, verified against the FPGate CA" $?
grep -q 'api/login' "$T/login.html"; check "the page is the login template" $?
grep -qi '^x-frame-options: deny' "$T/h.txt" && grep -qi '^content-security-policy:' "$T/h.txt" &&
    grep -qi '^cache-control: no-store' "$T/h.txt"
check "security headers: no framing, CSP, no caching" $?
curl -s --max-time 5 "http://127.0.0.1:18443/" -o /dev/null; [ $? -ne 0 ] || true
echo | openssl s_client -connect 127.0.0.1:18443 -tls1_1 >/dev/null 2>&1; [ $? -ne 0 ]
check "TLS 1.1 is refused" $?
echo | openssl s_client -connect 127.0.0.1:18443 -CAfile "$T/ca/ca.crt" -verify_ip 127.0.0.1 2>&1 | grep -q 'Verify return code: 0'
check "the certificate carries the address the browser used (IP SAN)" $?

# A browser that uses the proxy asks it for the proxy's own interface (CONNECT): plain tunnel, no interception.
curl -s -x http://127.0.0.1:18090 $CURL "$A/" -o "$T/viaproxy.html" -D "$T/h.txt"
head -1 "$T/h.txt" | grep -q ' 200 ' && grep -q 'api/login' "$T/viaproxy.html"
check "the interface is reachable through the proxy itself (a browser configured to use it)" $?
grep -q 'CONNECT 127.0.0.1:18443' "$T/p.log" && ! grep -q 'MITM 127.0.0.1' "$T/p.log"
check "that CONNECT is a plain tunnel: the proxy does not intercept its own interface" $?

# https://fpgate.admin through the proxy: the interface itself, whatever the board's address or name is.
curl -s -x http://127.0.0.1:18090 $CURL "https://fpgate.admin/" -o "$T/vhost.html" -D "$T/h.txt"
head -1 "$T/h.txt" | grep -q ' 200 ' && grep -q 'api/login' "$T/vhost.html"
check "https://fpgate.admin through the proxy serves the login page, certificate verified against the FPGate CA" $?
curl -s -x http://127.0.0.1:18090 $CURL -c "$T/jar2" -X POST -H "$J" -d '{"user":"admin","password":"correct horse"}' \
    "https://fpgate.admin/api/login" -o /dev/null -w '%{http_code}' | grep -q 200
check "logging in through https://fpgate.admin works (the Origin check sees the same name)" $?
curl -s -x http://127.0.0.1:18090 $CURL -b "$T/jar2" "https://fpgate.admin/api/state" | jget status | grep -q normal
check "and the session works there" $?
curl -s -x http://127.0.0.1:18090 $CURL -b "$T/jar2" -X POST "https://fpgate.admin/api/logout" -H "$J" -d '{}' -o /dev/null
curl -s -x http://127.0.0.1:18090 "http://fpgate.admin/" -o /dev/null -D "$T/h.txt"
head -1 "$T/h.txt" | grep -q ' 302 ' && grep -qi '^location: https://fpgate.admin/' "$T/h.txt"
check "plain http://fpgate.admin is redirected to https" $?
host=$(hostname)
curl -s --resolve "$host:18443:127.0.0.1" $CURL "https://$host:18443/" -o /dev/null -w '%{http_code}' | grep -q 200
check "directly by the machine's host name the certificate also matches (the address may change, the name does not)" $?

# --- 2. nothing without a session ---------------------------------------------------------------------------------------
bad=0
for p in /api/state /api/blocklog /api/substitute; do
    [ "$(curl -s $CURL -o /dev/null -w "%{http_code}" "$A$p")" = 401 ] || bad=1
done
[ $bad -eq 0 ]; check "the API answers 401 without a session" $?
[ "$(curl -s $CURL -o /dev/null -w '%{http_code}' -X POST -H "$J" -d '{"value":0.1}' "$A/api/threshold")" = 401 ]
check "changing the threshold without a session is refused" $?
curl -s $CURL -D "$T/h.txt" -o /dev/null "$A/dashboard"; head -1 "$T/h.txt" | grep -q ' 302 ' && grep -qi '^location: /' "$T/h.txt"
check "/dashboard without a session redirects to the login" $?

# --- 3. login ------------------------------------------------------------------------------------------------------------------
curl -s $CURL -X POST -H "$J" -d '{"user":"admin","password":"wrong password"}' "$A/api/login" -o "$T/r.json" -w '%{http_code}' > "$T/code"
[ "$(cat "$T/code")" = 401 ] && grep -q 'incorrectos' "$T/r.json"
check "a wrong password: 401 with a plain-language message" $?
! grep -qi 'sqlite\|errno\|stack' "$T/r.json"; check "the message is not a raw error" $?
curl -s $CURL -X POST -H "$J" -d '{"user":"root","password":"correct horse"}' "$A/api/login" -o /dev/null -w '%{http_code}' | grep -q 401
check "a wrong user name: 401" $?
curl -s $CURL -X POST -H "$J" -d '{"user":"admin"}' "$A/api/login" -o /dev/null -w '%{http_code}' | grep -q 400
check "a missing field: 400" $?
curl -s $CURL -X POST -H "$J" -H "Origin: https://evil.example" -d '{"user":"admin","password":"correct horse"}' "$A/api/login" \
    -o /dev/null -w '%{http_code}' | grep -q 403
check "a cross-site Origin is refused (403)" $?

curl -s $CURL -c "$T/jar" -D "$T/h.txt" -X POST -H "$J" -d '{"user":"admin","password":"correct horse"}' "$A/api/login" -o /dev/null
head -1 "$T/h.txt" | grep -q ' 200 '; check "the right password logs in" $?
grep -qi '^set-cookie: fpgate_session=[0-9a-f]\{32\}; .*HttpOnly' "$T/h.txt" && grep -qi 'secure' "$T/h.txt" && grep -qi 'samesite=strict' "$T/h.txt"
check "the session cookie is random, HttpOnly, Secure and SameSite=Strict" $?
S="-b $T/jar"

curl -s $CURL $S "$A/dashboard" -o "$T/dash.html" -D "$T/h.txt"; head -1 "$T/h.txt" | grep -q ' 200 ' && grep -q 'thresholdSlider' "$T/dash.html"
check "with the session the dashboard is served" $?
curl -s $CURL $S -D "$T/h.txt" -o /dev/null "$A/"; head -1 "$T/h.txt" | grep -q ' 302 '
check "the login page sends a logged-in administrator to the dashboard" $?

# --- 4. state and threshold (RF-09, RI-02) ----------------------------------------------------------------------------------
curl -s $CURL $S "$A/api/state" -o "$T/state.json"
[ "$(jget threshold < "$T/state.json")" = 0.6 ] && [ "$(jget status < "$T/state.json")" = normal ] && [ "$(jget persistent < "$T/state.json")" = True ]
check "state: threshold 0.6 from the file, status normal, persistent" $?

curl -s -x http://127.0.0.1:18090 "http://127.0.0.1:18081/img/white.png" -o "$T/o1.png"          # score 1.0 >= 0.6: blocked
! cmp -s "$T/o1.png" "$T/img/white.png"; check "at 0.60 a white image (1.0) is replaced" $?
curl -s -x http://127.0.0.1:18090 "http://127.0.0.1:18081/img/black.png" | cmp -s - "$T/img/black.png"
check "at 0.60 a black image is delivered" $?

bad=0
for body in '{"value":1.5}' '{"value":-0.1}' '{"value":"abc"}' '{}' '{"value":nan}'; do
    c=$(curl -s $CURL $S -X POST -H "$J" -d "$body" "$A/api/threshold" -o "$T/r.json" -w '%{http_code}')
    [ "$c" = 400 ] && grep -q 'umbral' "$T/r.json" || { echo "  bad threshold $body -> $c"; bad=1; }
done
[ $bad -eq 0 ]; check "invalid thresholds are rejected with a clear message" $?
[ "$(curl -s $CURL $S "$A/api/state" | jget threshold)" = 0.6 ]; check "a rejected value leaves the threshold in force" $?

curl -s -x http://127.0.0.1:18090 "http://127.0.0.1:18081/img/gray128.png" | cmp -s - "$T/img/gray128.png"
check "at 0.60 a mid-grey image (0.502) is delivered" $?
curl -s $CURL $S -X POST -H "$J" -d '{"value":0.4}' "$A/api/threshold" -o /dev/null -w '%{http_code}' | grep -q 200
check "a valid threshold is accepted" $?
curl -s -x http://127.0.0.1:18090 "http://127.0.0.1:18081/img/gray128.png" -o "$T/g.out"; ! cmp -s "$T/g.out" "$T/img/gray128.png"
check "the next classification already uses 0.40 (the same grey image is now replaced), no restart" $?
[ "$(curl -s $CURL $S -X POST -H "$J" -d '{"value":1.0}' "$A/api/threshold" -o /dev/null -w '%{http_code}')" = 200 ] &&
    [ "$(curl -s $CURL $S -X POST -H "$J" -d '{"value":0}' "$A/api/threshold" -o /dev/null -w '%{http_code}')" = 200 ]
check "the limits 0 and 1 are valid" $?
grep -q '^umbral = 0$' "$T/fpgate.toml" && grep -q 'puerto = 8080' "$T/fpgate.toml" && grep -q 'timeout_accel_ms = 10000' "$T/fpgate.toml" &&
    grep -q '^# written by hand' "$T/fpgate.toml"
check "the TOML file is updated; its comments and other keys are kept" $?
curl -s $CURL $S -X POST -H "$J" -d '{"value":0.3}' "$A/api/threshold" -o /dev/null

# SIGHUP re-reads the file
sed -i 's/^umbral = .*/umbral = 0.9/' "$T/fpgate.toml"
kill -HUP $P; sleep 1
[ "$(curl -s $CURL $S "$A/api/state" | jget threshold)" = 0.9 ]; check "SIGHUP applies an edit made to the file" $?
grep -q 'configuration reloaded' "$T/p.log"; check "the reload is logged" $?
sed -i 's/^umbral = .*/umbral = 7/' "$T/fpgate.toml"
kill -HUP $P; sleep 1
[ "$(curl -s $CURL $S "$A/api/state" | jget threshold)" = 0.9 ]; check "an invalid file is refused on reload: the previous value stays" $?
grep -q 'NOT reloaded' "$T/p.log"; check "and the log says why" $?
curl -s $CURL $S -X POST -H "$J" -d '{"value":0.5}' "$A/api/threshold" -o /dev/null

# --- 5. the stand-in image (RU-03, RU-04) -------------------------------------------------------------------------------------
curl -s $CURL $S "$A/api/substitute" -D "$T/h.txt" -o "$T/prev.png"
grep -qi '^content-type: image/png' "$T/h.txt"; check "the preview of the default stand-in is a PNG" $?
[ "$(jget substitute.custom < "$T/state.json")" = False ]; check "state: no custom image yet" $?

bad=0
for f in fake.png big.png; do
    c=$(curl -s $CURL $S -X POST -H "Content-Type: image/png" --data-binary @"$T/img/$f" "$A/api/substitute" -o "$T/r.json" -w '%{http_code}')
    [ "$c" = 400 -o "$c" = 413 ] && grep -q 'JPG, PNG o WebP' "$T/r.json" || grep -q 'demasiado grande' "$T/r.json" || { echo "  $f -> $c"; bad=1; }
done
[ $bad -eq 0 ]; check "a file that is not an image, and one over 2 MB, are rejected with a clear message" $?
[ ! -e "$T/substitute.png" ] && [ ! -e "$T/.upload" ]; check "nothing is left behind after a rejected upload" $?

c=$(curl -s $CURL $S -X POST -H "Content-Type: image/png" --data-binary @"$T/img/custom.png" "$A/api/substitute" -w '%{http_code}' -o /dev/null)
[ "$c" = 200 ] && cmp -s "$T/substitute.png" "$T/img/custom.png"; check "a PNG is accepted and stored next to the configuration" $?
grep -q "sustitucion = \"$T/substitute.png\"" "$T/fpgate.toml"; check "the file records it" $?
curl -s -x http://127.0.0.1:18090 --max-time 25 "http://127.0.0.1:18081/img/white.png?v=2" -o "$T/o2.png"
curl -s $CURL $S -X POST -H "$J" -d '{"value":0.1}' "$A/api/threshold" -o /dev/null
curl -s -x http://127.0.0.1:18090 "http://127.0.0.1:18081/img/white.png?v=3" | cmp -s - "$T/img/custom.png"
check "the next blocked image is the uploaded one, as it is" $?
curl -s $CURL $S -X POST -H "Content-Type: image/jpeg" --data-binary @"$T/img/custom.jpg" "$A/api/substitute" -o /dev/null
[ -e "$T/substitute.jpg" ] && [ ! -e "$T/substitute.png" ]; check "replacing it with a JPEG keeps a single stored image" $?

# --- 6. the block log (RF-11, RU-02) ---------------------------------------------------------------------------------------------
sleep 1
curl -s $CURL $S "$A/api/blocklog?page=1&size=2" -o "$T/log.json"
[ "$(jget enabled < "$T/log.json")" = True ] && [ "$(jget total < "$T/log.json")" -ge 3 ] && [ "$(jget rows < "$T/log.json" | grep -c host)" -ge 1 ]
check "the block log is paginated and has the blocks made above" $?
[ "$(jget rows.0.host < "$T/log.json")" = 127.0.0.1 ] && [ "$(jget rows.0.reason < "$T/log.json")" = nsfw ]
check "a row carries the host and the reason, nothing else about the image" $?
! grep -q 'img/' "$T/log.json"; check "no URL path in the API" $?
curl -s $CURL $S "$A/api/blocklog?page=1&size=2" | jget pages | grep -qE '^[2-9]'; check "pages are counted from the total" $?
curl -s $CURL $S "$A/api/blocklog?page=999&size=2" | jget rows | grep -q '^\[\]$'; check "a page beyond the end is empty, not an error" $?

# --- 7. certificate pages (RU-05, CU-04) -------------------------------------------------------------------------------------------------
curl -s $CURL "$A/ca.crt" -D "$T/h.txt" -o "$T/ca-dl.crt"
cmp -s "$T/ca-dl.crt" "$T/ca/ca.crt" && grep -qi '^content-disposition: attachment; filename="fpgate-ca.crt"' "$T/h.txt"
check "ca.crt downloads without login, and is exactly the public CA" $?
! grep -q 'PRIVATE' "$T/ca-dl.crt"; check "no private key in the download" $?
curl -s $CURL "$A/certificado" -o "$T/cert.html"; grep -q 'fpgate-ca.crt' "$T/cert.html" && grep -q -- '--proxy-server="127.0.0.1:18090"' "$T/cert.html"
check "the certificate page is public and shows the proxy address to use with Chrome" $?

curl -s -x http://127.0.0.1:18090 "http://fpgate.setup/" -o "$T/setup.html" -D "$T/h.txt"
head -1 "$T/h.txt" | grep -q ' 200 ' && grep -q -- '--proxy-server="127.0.0.1:18090"' "$T/setup.html"
check "http://fpgate.setup through the proxy serves the certificate page" $?
curl -s -x http://127.0.0.1:18090 "http://fpgate.setup/ca.crt" | cmp -s - "$T/ca/ca.crt"; check "http://fpgate.setup/ca.crt is the CA" $?
[ "$(curl -s -x http://127.0.0.1:18090 -o /dev/null -w '%{http_code}' "http://fpgate.setup/api/state")" = 404 ]
check "fpgate.setup does not expose the API" $?
[ "$(curl -s -x http://127.0.0.1:18090 -o /dev/null -w '%{http_code}' -X POST "http://fpgate.setup/ca.crt")" = 405 ]
check "fpgate.setup only answers GET" $?

# --- 8. logout and lockout ----------------------------------------------------------------------------------------------------------------
curl -s $CURL $S -X POST -H "$J" -d '{}' "$A/api/logout" -o /dev/null
[ "$(curl -s $CURL $S -o /dev/null -w '%{http_code}' "$A/api/state")" = 401 ]; check "after logout the old cookie is dead" $?

for i in 1 2 3 4 5; do curl -s $CURL -X POST -H "$J" -d '{"user":"admin","password":"nope nope"}' "$A/api/login" -o /dev/null; done
c=$(curl -s $CURL -X POST -H "$J" -d '{"user":"admin","password":"correct horse"}' "$A/api/login" -o "$T/r.json" -w '%{http_code}')
[ "$c" = 429 ] && grep -q 'Espera' "$T/r.json"; check "five wrong passwords lock logins for a while, even the right one (429)" $?

kill -TERM $P; wait $P 2>/dev/null

# --- 9. without credentials -----------------------------------------------------------------------------------------------------------------------
$BIN -p 18091 $CA --admin-port 18444 2>"$T/p2.log" & P2=$!
PIDS="$PIDS $P2"
sleep 2
curl -s $CURL --max-time 3 "https://127.0.0.1:18444/" -o /dev/null; [ $? -ne 0 ]; check "no --admin-credentials: the interface is off" $?
curl -s -x http://127.0.0.1:18091 "http://fpgate.setup/ca.crt" | cmp -s - "$T/ca/ca.crt"
check "but http://fpgate.setup/ca.crt still works" $?
kill -TERM $P2; wait $P2 2>/dev/null
$BIN -p 18092 --admin-credentials "$T/admin.pass" 2>"$T/p3.log"; rc=$?
[ $rc -ne 0 ] && grep -q 'needs the CA' "$T/p3.log"; check "credentials without a CA are refused at start-up" $?
$BIN -p 18093 $CA --admin-credentials "$T/none" 2>"$T/p4.log"; rc=$?
[ $rc -ne 0 ] && grep -q 'cannot open' "$T/p4.log"; check "a missing credentials file is refused at start-up" $?
printf 'x\n' > "$T/damaged"; $BIN -p 18094 $CA --admin-credentials "$T/damaged" 2>"$T/p5.log"; rc=$?
[ $rc -ne 0 ]; check "a damaged credentials file is refused at start-up" $?

[ $fail -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit $fail
