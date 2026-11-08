#!/bin/bash
# DES-3.4: the block log. Every block leaves one row with host, score, threshold and
# reason; nothing else reaches the file (RS-05); allowed images leave no row.
# Usage: tests/integration/test_blocklog.sh [proxy-binary] [mock_accel]
BIN=${1:-build/asan/fpgate-proxy}
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
# Runs a query on the log and prints the rows.
q() { python3 -c "
import sqlite3, sys
c = sqlite3.connect('$T/blocks.db')
for r in c.execute(sys.argv[1]): print('|'.join('' if v is None else str(v) for v in r))" "$1"; }

mkdir -p "$T/img"
python3 - "$T/img" <<'EOF'
import sys
from PIL import Image
d = sys.argv[1]
Image.new("RGB", (800, 600), (255, 255, 255)).save(d + "/white.png")
Image.new("RGB", (640, 480), (0, 0, 0)).save(d + "/black.png")
Image.new("RGB", (200, 150), (255, 255, 255)).save(d + "/anim.gif")
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
$MOCKC "$T/m.sock" --mean --quiet 2>/dev/null & MOCK=$!

CA="-C $T/ca/ca.crt -K $T/ca/ca.key -P $T/pass -B $T/oca.crt"
$BIN -p 18090 $CA --accel "$T/m.sock" --threshold 0.6 --accel-timeout-ms 3000 \
    --block-log "$T/blocks.db" 2>"$T/p.log" & P=$!
sleep 2
O=http://127.0.0.1:18081
Q=https://fpgate.test:18443

# --- 1. what gets a row --------------------------------------------------------------------------------
curl -s -x http://127.0.0.1:18090 "$O/img/white.png?secret=token123" -o /dev/null
curl -s -x http://127.0.0.1:18090 --cacert "$T/ca/ca.crt" "$Q/img/white.png?user=bob" -o /dev/null
curl -s -x http://127.0.0.1:18090 "$O/img/black.png" -o /dev/null           # allowed: no row
curl -s -x http://127.0.0.1:18090 "$O/img/anim.gif" -o /dev/null           # unexamined, allowed: no row
kill -TERM $P; wait $P 2>/dev/null                                        # shutdown flushes the queue

[ "$(q 'select count(*) from blocks')" = 2 ]; check "two blocks, two rows; allowed images leave none" $?
[ "$(q "select count(*) from blocks where host='127.0.0.1'")" = 1 ]; check "plain HTTP: the host is stored" $?
[ "$(q "select count(*) from blocks where host='fpgate.test'")" = 1 ]
check "HTTPS: the host is stored without the port" $?
[ "$(q "select count(*) from blocks where reason='nsfw' and score>0.99 and abs(threshold-0.6)<1e-6")" = 2 ]
check "score and the threshold in force are recorded with reason nsfw" $?
now=$(date +%s)
ts=$(q 'select min(ts) from blocks'); [ $((now - ts)) -lt 60 ] && [ $((now - ts)) -ge 0 ]
check "timestamps are epoch seconds, UTC" $?

# --- 2. privacy (RS-05) ----------------------------------------------------------------------------------------
cols=$(q "select group_concat(name) from pragma_table_info('blocks')")
[ "$cols" = "id,ts,host,score,threshold,reason" ]; check "the table has no column for URLs or pixels ($cols)" $?
! grep -aq -e 'token123' -e 'user=bob' -e 'img/white' "$T/blocks.db" "$T/blocks.db-wal" 2>/dev/null
check "no path, query string or file name reaches the database file" $?
[ "$(stat -c %s "$T/blocks.db")" -lt 100000 ]; check "the database is small (no image data)" $?

# --- 3. failures are logged too, without a score ------------------------------------------------------------------
$BIN -p 18091 $CA --accel "$T/missing.sock" --block-log "$T/blocks.db" 2>"$T/p2.log" & P2=$!
PIDS="$PIDS $P2"
sleep 2
curl -s -x http://127.0.0.1:18091 "$O/img/white.png" -o /dev/null
kill -TERM $P2; wait $P2 2>/dev/null
[ "$(q "select count(*) from blocks where score is null and reason='accel_down'")" = 1 ]
check "accelerator down: the block is logged with reason accel_down and no score" $?
[ "$(q 'select count(*) from blocks')" = 3 ]; check "a second proxy run appends to the same file" $?

# --- 4. a bad path stops the start-up, not the traffic later -------------------------------------------------
$BIN -p 18092 --block-log "$T/no/such/dir/blocks.db" 2>"$T/p3.log"; rc=$?
[ $rc -ne 0 ] && grep -q 'blocklog: cannot open' "$T/p3.log"
check "an unusable --block-log path is refused at start-up with a clear message" $?

[ $fail -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit $fail
