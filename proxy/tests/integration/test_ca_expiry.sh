#!/bin/bash
# RS-07: a CA that is expired, or expires while the proxy runs, must stop the interception:
# HTTPS goes through as a blind tunnel (the browser sees the real server's certificate and
# nothing is inspected), the administration interface refuses TLS, and plain HTTP keeps working.
# Usage: tests/integration/test_ca_expiry.sh [proxy-binary]
BIN=${1:-build/asan/fpgate-proxy}
T=$(mktemp -d)
fail=0
PIDS=""

curl() { command curl --max-time 20 "$@"; }
check() { # name, exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
cleanup() { [ $fail -ne 0 ] && for l in "$T"/*.log; do echo "--- $l"; tail -12 "$l"; done; kill $PIDS 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT

command -v faketime >/dev/null || { echo "faketime is needed to make an expired CA"; exit 2; }

mkdir -p "$T/img"
python3 - "$T/img" <<'EOF'
import sys
from PIL import Image
Image.new("RGB", (800, 600), (255, 255, 255)).save(sys.argv[1] + "/white.png")
EOF
export IMG_DIR="$T/img"

# A CA created in the past, valid for one day starting `offset` seconds before now - 1 day.
make_ca() { # dir, seconds from now until it expires
    local dir=$1 left=$2
    local start; start=$(date -d "@$(( $(date +%s) - 86400 + left ))" '+%Y-%m-%d %H:%M:%S')
    mkdir -p "$dir"
    FPGATE_CA_PASS=testpass faketime "$start" sh tools/gen_ca.sh "$dir" 1 >/dev/null 2>&1
}
ca_left() { openssl x509 -in "$1" -noout -enddate | cut -d= -f2; }

grep -q fpgate.test /etc/hosts 2>/dev/null || echo "127.0.0.1 fpgate.test" >> /etc/hosts
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
python3 tests/integration/origin.py 18453 "$T/o.crt" "$T/o.key" & PIDS="$PIDS $!"
printf 'correct horse\n' | $BIN --set-admin-password "$T/admin.pass" 2>/dev/null

Q=https://fpgate.test:18453
# Through the proxy, verifying only against the ORIGIN's CA: it succeeds only if the proxy did not
# substitute its own certificate (a blind tunnel). Verifying against the FPGate CA is the reverse.
via_origin_ca() { curl -s -x http://127.0.0.1:$1 --cacert "$T/oca.crt" "$Q/echo" -o /dev/null -w '%{http_code}'; }
via_fpgate_ca() { curl -s -x http://127.0.0.1:$1 --cacert "$2" "$Q/echo" -o /dev/null -w '%{http_code}'; }

# --- 1. a CA that is already expired when the proxy starts -----------------------------------------------------------
make_ca "$T/old" -172800            # expired two days ago
[ "$(openssl x509 -in "$T/old/ca.crt" -noout -checkend 0 >/dev/null 2>&1; echo $?)" = 1 ]
check "the test CA really is expired ($(ca_left "$T/old/ca.crt"))" $?

$BIN -p 18190 -C "$T/old/ca.crt" -K "$T/old/ca.key" -P "$T/pass" -B "$T/oca.crt" \
    --admin-credentials "$T/admin.pass" --admin-port 18454 2>"$T/p1.log" & P1=$!
PIDS="$PIDS $P1"
sleep 2.5
grep -q 'CA expired or not yet valid' "$T/p1.log"
check "start-up logs that the interception is off, and why" $?
grep -q 'admin: the interface needs the CA' "$T/p1.log"; check "the administration interface refuses to start without a valid CA" $?
[ "$(via_origin_ca 18190)" = 200 ]; check "HTTPS still works, as a blind tunnel: the browser sees the server's own certificate" $?
[ "$(via_fpgate_ca 18190 "$T/old/ca.crt")" != 200 ]; check "and it does not present a certificate from the expired CA" $?
! grep -q '^MITM' "$T/p1.log"; check "nothing was intercepted" $?
curl -s -x http://127.0.0.1:18190 "http://127.0.0.1:18081/img/white.png" -o /dev/null
grep -q 'IMG host=127.0.0.1' "$T/p1.log"; check "plain HTTP is still inspected (only the HTTPS interception depends on the CA)" $?
kill -TERM $P1; wait $P1 2>/dev/null

# --- 2. a CA that expires while the proxy is running --------------------------------------------------------------------------
make_ca "$T/soon" 22                # valid for about 22 more seconds
$BIN -p 18191 -C "$T/soon/ca.crt" -K "$T/soon/ca.key" -P "$T/pass" -B "$T/oca.crt" \
    --admin-credentials "$T/admin.pass" --admin-port 18455 2>"$T/p2.log" & P2=$!
PIDS="$PIDS $P2"
sleep 2
grep -q 'CA valid, HTTPS interception enabled' "$T/p2.log"; check "a CA with time left starts normally" $?
[ "$(via_fpgate_ca 18191 "$T/soon/ca.crt")" = 200 ]; check "while it is valid, HTTPS is intercepted (certificate from the FPGate CA)" $?
grep -q '^MITM fpgate.test' "$T/p2.log"; check "and the log shows the interception" $?
[ "$(curl -s --cacert "$T/soon/ca.crt" -o /dev/null -w '%{http_code}' https://127.0.0.1:18455/)" = 200 ]
check "the administration interface answers TLS while the CA is valid" $?

echo "   waiting for the CA to expire (the check runs every 10 s)..."
for i in $(seq 1 45); do
    grep -q 'interception disabled' "$T/p2.log" && break
    sleep 1
done
grep -q 'CA expired or not yet valid' "$T/p2.log"
check "once it expires the proxy notices by itself and logs it" $?
before=$(grep -c '^MITM' "$T/p2.log")
[ "$(via_origin_ca 18191)" = 200 ]; check "new HTTPS connections now pass as blind tunnels" $?
[ "$(grep -c '^MITM' "$T/p2.log")" = "$before" ]; check "no new interception after the expiry" $?
curl -s --cacert "$T/soon/ca.crt" https://127.0.0.1:18455/ -o /dev/null; [ $? -ne 0 ]
check "the administration interface stops answering TLS with a certificate from an expired CA" $?
[ "$(curl -s -x http://127.0.0.1:18191 'http://fpgate.setup/ca.crt' -o /dev/null -w '%{http_code}')" = 200 ]
check "http://fpgate.setup keeps serving the CA, so a renewed one can be installed" $?
kill -TERM $P2; wait $P2 2>/dev/null

[ $fail -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit $fail
