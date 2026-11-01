#!/bin/sh
# Plain-HTTP and CONNECT integration tests. Usage: tests/integration/test_http.sh [binary]
BIN=${1:-build/asan/fpgate-proxy}
PROXY=127.0.0.1:18080
ORIGIN=127.0.0.1:18081
fail=0

python3 tests/integration/origin.py 18081 & OPID=$!
$BIN -p 18080 2>proxy.log & PPID_=$!
trap 'kill $OPID $PPID_ 2>/dev/null; wait 2>/dev/null' EXIT
sleep 1

check() { # name, condition-exit-code
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}

curl -s -x $PROXY http://$ORIGIN/echo -H 'X-Test: abc' -H 'Accept-Language: es' >out.txt
grep -qi '^x-test: abc' out.txt && grep -qi '^accept-language: es' out.txt; check "headers preserved" $?
grep -q '^GET /echo HTTP/1.1' out.txt; check "request line rewritten to origin form" $?
! grep -qi '^proxy-connection' out.txt; check "proxy-connection stripped" $?

[ "$(curl -s -x $PROXY http://$ORIGIN/chunked)" = "hello chunked world" ]; check "chunked response" $?

curl -s -x $PROXY http://$ORIGIN/big | cmp -s - <(python3 -c 'import sys;sys.stdout.buffer.write((b"0123456789abcdef"*4096)*48)') 2>/dev/null
check "3 MiB body intact" $?

[ "$(curl -s -x $PROXY -d 'payload-123' http://$ORIGIN/post)" = "payload-123" ]; check "POST body" $?

[ "$(curl -s -x $PROXY --proxytunnel http://$ORIGIN/chunked)" = "hello chunked world" ]; check "CONNECT tunnel" $?

[ "$(curl -s -o /dev/null -w '%{http_code}' -x $PROXY http://127.0.0.1:1/)" = "502" ]; check "unreachable upstream -> 502" $?
[ "$(curl -s -o /dev/null -w '%{http_code}' $PROXY/echo)" = "400" ]; check "non-proxy request -> 400" $?

kill -0 $PPID_ 2>/dev/null; check "proxy still alive" $?
rm -f out.txt
exit $fail
