#!/bin/sh
# Board gate for the FPGA build: runs the 20-image selftest with the FPGA linear
# layers and attention, then compares the logits with the CPU reference.
#
# Usage on the board, from the directory that holds fpgate-accel, pkg/ and e1_r7.idx
# (ref_logits_20.txt must sit next to this script):
#   sh gate.sh [extra fpgate-accel flags]
#
# Prints:
#   RESULT gate: PASS|FAIL (<n> lines differ)
#   RESULT time: <min>-<max> s/img
# followed by the timing summary the program prints itself.
# Set FPGATE_ACCEL to use another binary (default ./fpgate-accel).

HERE=$(cd "$(dirname "$0")" && pwd)
REF="$HERE/ref_logits_20.txt"
BIN=${FPGATE_ACCEL:-./fpgate-accel}

if [ ! -f "$REF" ]; then
    echo "RESULT gate: FAIL (reference not found: $REF)"
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "RESULT gate: FAIL (binary not found or not executable: $BIN)"
    exit 1
fi

TMP=$(mktemp -d 2>/dev/null || echo "/tmp/gate.$$")
mkdir -p "$TMP"

"$BIN" --pkg pkg --index e1_r7.idx --selftest 20 --logits --fpga --fpga-attn "$@" \
    > "$TMP/run.txt" 2> "$TMP/err.txt"
rc=$?

grep '^    logits ' "$TMP/run.txt" > "$TMP/got.txt"
diff "$REF" "$TMP/got.txt" > "$TMP/diff.txt"
n_diff=$(grep -c '^[<>]' "$TMP/diff.txt")

if [ "$rc" -eq 0 ] && [ "$n_diff" -eq 0 ]; then
    echo "RESULT gate: PASS (0 lines differ)"
else
    echo "RESULT gate: FAIL ($n_diff lines differ, fpgate-accel exit $rc)"
    tail -5 "$TMP/err.txt"
fi

times=$(grep '^vector ' "$TMP/run.txt" | sed -n 's/.* \([0-9.][0-9.]*\) s$/\1/p')
if [ -n "$times" ]; then
    range=$(echo "$times" | awk 'NR == 1 { mn = mx = $1 }
        { if ($1 < mn) mn = $1; if ($1 > mx) mx = $1 }
        END { printf "%s-%s", mn, mx }')
    echo "RESULT time: $range s/img"
else
    echo "RESULT time: n/a"
fi

grep 'time by stage' "$TMP/run.txt"
grep 'decisions identical' "$TMP/run.txt"

rm -rf "$TMP"
[ "$rc" -eq 0 ] && [ "$n_diff" -eq 0 ]
