#!/bin/bash
# HTTPS interception tests. Usage: tests/integration/test_tls.sh [binary]
# Needs: openssl, curl, python3, and a "fpgate.test" entry (added if missing).
BIN=${1:-build/asan/fpgate-proxy}
T=$(mktemp -d)
fail=0
PIDS=""

curl() { command curl --max-time 15 "$@"; }
check() { # name, exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
cleanup() { kill $PIDS 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT

grep -q fpgate.test /etc/hosts 2>/dev/null || echo "127.0.0.1 fpgate.test" >> /etc/hosts

# --- certificates --------------------------------------------------------
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

# --- servers -------------------------------------------------------------
python3 tests/integration/origin.py 18081 & PIDS="$PIDS $!"
python3 tests/integration/origin.py 18443 "$T/o.crt" "$T/o.key" & PIDS="$PIDS $!"
$BIN -p 18090 -C "$T/ca/ca.crt" -K "$T/ca/ca.key" -P "$T/pass" -B "$T/oca.crt" 2>"$T/p1.log" & P1=$!
# Second proxy trusts the wrong bundle: upstream verification must fail.
$BIN -p 18091 -C "$T/ca/ca.crt" -K "$T/ca/ca.key" -P "$T/pass" -B "$T/ca/ca.crt" 2>"$T/p2.log" & P2=$!
PIDS="$PIDS $P1 $P2"
sleep 2
X1="-x http://127.0.0.1:18090 --cacert $T/ca/ca.crt"

# --- tests ---------------------------------------------------------------
out=$(curl -s $X1 https://127.0.0.1:18443/echo -H 'X-Test: abc')
echo "$out" | grep -qi '^x-test: abc'; check "HTTPS by IP: decrypted, headers preserved" $?

out=$(curl -s $X1 https://fpgate.test:18443/echo)
echo "$out" | grep -q '^GET /echo HTTP/1.1'; check "HTTPS by hostname (SAN DNS leaf)" $?

curl -sv $X1 https://fpgate.test:18443/echo 2>&1 | grep -q 'issuer: CN=FPGate CA'; check "leaf is issued by the FPGate CA" $?

curl -s -x http://127.0.0.1:18090 https://127.0.0.1:18443/echo >/dev/null 2>&1
[ $? -eq 60 ]; check "client without the CA rejects the leaf (curl exit 60)" $?

curl -s $X1 --tlsv1.2 --tls-max 1.2 https://fpgate.test:18443/echo | grep -q '^GET'; check "client TLS 1.2" $?
curl -s $X1 --tlsv1.3 https://fpgate.test:18443/echo | grep -q '^GET'; check "client TLS 1.3" $?

s=$(echo | timeout 15 openssl s_client -proxy 127.0.0.1:18090 -connect fpgate.test:18443 -tls1_1 \
        -cipher 'DEFAULT:@SECLEVEL=0' -CAfile "$T/ca/ca.crt" 2>&1)
echo "$s" | grep -q 'alert protocol version'; check "client TLS 1.1 rejected (protocol_version alert)" $?

curl -s $X1 https://fpgate.test:18443/big | cmp -s - <(python3 -c 'import sys;sys.stdout.buffer.write((b"0123456789abcdef"*4096)*48)')
check "3 MiB over HTTPS intact" $?

[ "$(curl -s $X1 -d 'payload-xyz' https://fpgate.test:18443/post)" = "payload-xyz" ]; check "POST over HTTPS" $?

curl -s $X1 https://fpgate.test:18443/echo https://fpgate.test:18443/echo | grep -c '^GET' | grep -q '^2$'
check "two requests through the proxy" $?

[ "$(curl -s --proxytunnel -x http://127.0.0.1:18090 http://127.0.0.1:18081/chunked)" = "hello chunked world" ]
check "non-TLS CONNECT goes through a blind tunnel" $?

curl -s --cacert "$T/ca/ca.crt" -x http://127.0.0.1:18091 https://127.0.0.1:18443/echo 2>/dev/null | grep -q '^GET'
[ $? -ne 0 ]; check "untrusted upstream certificate is refused" $?
grep -q 'upstream' "$T/p2.log"; check "refusal logged with the reason" $?

FPGATE_CA_PASS=wrong $BIN -p 18092 -C "$T/ca/ca.crt" -K "$T/ca/ca.key" -B "$T/oca.crt" 2>/dev/null
[ $? -ne 0 ]; check "wrong CA password: proxy refuses to start" $?

kill -0 $P1 2>/dev/null; check "proxy still alive" $?
if grep -E "AddressSanitizer|runtime error" "$T"/p*.log; then echo "FAIL sanitizer report"; fail=1; fi
exit $fail
