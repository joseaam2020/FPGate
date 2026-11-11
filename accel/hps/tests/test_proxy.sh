#!/bin/bash
# End to end on a PC: the real proxy, the real fpgate-accel and a web server that serves real model inputs.
#
# The 20 reference tensors of the e1_r7 package are turned back into the 448x448 RGB8 pictures they came from
# (the uint8 -> float32 normalisation is exact, so this recovers them), served over HTTP, fetched through the
# proxy, judged by fpgate-accel, and the proxy's verdict and score are compared with the reference decision.
#
# usage: tests/test_proxy.sh PROXY ACCEL PACKAGE_DIR INDEX VECTOR_DIR N [accel options...]
#   e.g. tests/test_proxy.sh ../../proxy/build/native/fpgate-proxy build/native/fpgate-accel \
#            /repo/model/quant/release/e1_r7 pkg/e1_r7.idx /repo/model/quant/release/e1_r7/vectors 6 --sim
PROXY=$1; ACCEL=$2; PK=$3; IDX=$4; VEC=$5; N=${6:-6}; shift 6
ACCEL_OPTS="$*"
HERE=$(cd "$(dirname "$0")" && pwd)
ORIGIN=$HERE/../../../proxy/tests/integration/origin.py
T=$(mktemp -d)
PIDS=""
fail=0
cleanup() { kill $PIDS 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
check() { if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }

# --- pictures from the reference tensors -------------------------------------------------------------
mkdir -p "$T/img"
python3 - "$VEC" "$T/img" "$N" <<'EOF'
import sys
from array import array
from PIL import Image
vec, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
S = 448
for i in range(n):
    a = array("f")
    with open("%s/img_%02d.f32" % (vec, i), "rb") as f:
        a.frombytes(f.read())
    assert len(a) == 3 * S * S
    planes = []
    for c in range(3):
        p = a[c * S * S:(c + 1) * S * S]
        planes.append([min(255, max(0, int(round((x * std[c] + mean[c]) * 255.0)))) for x in p])
    data = bytearray(S * S * 3)
    for c in range(3):
        data[c::3] = bytes(planes[c])
    Image.frombytes("RGB", (S, S), bytes(data)).save("%s/img_%02d.png" % (out, i))
EOF
check "pictures rebuilt from $N reference tensors" $?

# --- reference decisions: the CPU backend on the same tensors -----------------------------------------
"$ACCEL" --selftest "$N" --pkg "$PK" --index "$IDX" 2>/dev/null | grep -E '^vector' > "$T/ref.txt"
[ "$(wc -l < "$T/ref.txt")" -eq "$N" ]; check "reference run produced $N decisions" $?

# --- the servers --------------------------------------------------------------------------------------
IMG_DIR="$T/img" python3 "$ORIGIN" 18081 & PIDS="$PIDS $!"
"$ACCEL" "$T/accel.sock" --pkg "$PK" --index "$IDX" $ACCEL_OPTS 2>"$T/accel.log" & PIDS="$PIDS $!"
for _ in $(seq 1 100); do [ -S "$T/accel.sock" ] && break; sleep 0.3; done
[ -S "$T/accel.sock" ]; check "fpgate-accel is listening ($ACCEL_OPTS)" $?
"$PROXY" -p 18090 --accel "$T/accel.sock" --accel-timeout-ms 120000 2>"$T/proxy.log" & PIDS="$PIDS $!"
sleep 2

# --- every picture through the proxy ------------------------------------------------------------------
for i in $(seq 0 $((N - 1))); do
    f=$(printf 'img_%02d.png' "$i")
    start=$(date +%s.%N)
    command curl -s --max-time 150 -x http://127.0.0.1:18090 "http://127.0.0.1:18081/img/$f" -o "$T/got_$f"
    el=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
    line=$(grep -E 'VERDICT' "$T/proxy.log" | sed -n "$((i + 1))p")
    verdict=$(echo "$line" | sed -n 's/.*verdict=\([a-z]*\).*/\1/p')
    score=$(echo "$line" | sed -n 's/.* score=\([0-9.]*\).*/\1/p')
    want=$(sed -n "$((i + 1))p" "$T/ref.txt")
    label=$(echo "$want" | sed -n 's/.*(\(NSFW\|SFW\)).*/\1/p')
    refscore=$(echo "$want" | sed -n 's/.*(ref \([0-9.]*\)).*/\1/p')
    exp=allow; [ "$label" = NSFW ] && exp=block
    ok=0
    [ "$verdict" = "$exp" ] || ok=1
    # score printed by the proxy has 3 decimals: compare within 0.01 of the reference score
    awk -v a="$score" -v b="$refscore" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.01) }' || ok=1
    # a blocked picture must come back replaced, an allowed one unchanged
    if [ "$exp" = allow ]; then cmp -s "$T/got_$f" "$T/img/$f" || ok=1; else ! cmp -s "$T/got_$f" "$T/img/$f" || ok=1; fi
    check "$f: $label -> proxy verdict=$verdict score=$score (reference $refscore) in ${el}s" $ok
done
grep -E "VERDICT.*reason=(accel_down|accel_error|accel_timeout)" "$T/proxy.log" >/dev/null; [ $? -ne 0 ]; check "no accelerator failures in the proxy log" $?
[ $fail -eq 0 ] && echo "ALL PASSED" || { echo "FAILED"; tail -5 "$T/accel.log"; exit 1; }
