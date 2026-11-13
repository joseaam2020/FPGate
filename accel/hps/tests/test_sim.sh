#!/bin/sh
# Overlap and pipeline tests on the software engine (--sim): every mode must give the same logits, bit for
# bit, as the sequential mode, and every layer must equal the CPU result (--check).
# usage: tests/test_sim.sh BINARY PACKAGE_DIR INDEX [IMAGES]
BIN=$1; PK=$2; IDX=$3; N=${4:-2}
tmp=$(mktemp -d)
fail=0
run() { # name, options
    name=$1; shift
    "$BIN" --selftest "$N" --pkg "$PK" --index "$IDX" --sim ${CHK---check} --logits "$@" > "$tmp/$name.out" 2>&1
    rc=$?
    grep -E '^    logits' "$tmp/$name.out" > "$tmp/$name.logits"
    got=$(grep -c '^    logits' "$tmp/$name.out")
    if [ $rc -ne 0 ] || [ "$got" -ne "$N" ]; then
        echo "FAIL $name: exit status $rc, $got of $N logits lines (the binary crashed or did not finish)"; fail=1
    fi
    if grep -qE 'differ from the CPU|DIFFER FROM|MISMATCH|accel.*(failed|did not finish)|pipeline: ' "$tmp/$name.out"; then
        echo "FAIL $name: a layer differs from the CPU result or an error was reported"; fail=1
    fi
    echo "ran $name: $got logits lines (exit $rc)"
}
run base --no-overlap --no-pipeline
for mode in "layer:--no-pipeline" "pipe256:" "pipe160:--chunk 160" "pipe512:--chunk 512" "pipe48:--chunk 48" "cpupatch:--cpu-patch" "attn_seq:--fpga-attn --no-att-overlap" "attn_ovl:--fpga-attn" "attn_pack:--fpga-attn --att-overlap-pack-only"; do
    name=${mode%%:*}; opts=${mode#*:}
    run "$name" $opts
    if ! cmp -s "$tmp/base.logits" "$tmp/$name.logits"; then
        echo "FAIL $name: logits differ from the sequential run"; diff "$tmp/base.logits" "$tmp/$name.logits" | head -4; fail=1
    else
        echo "PASS $name: logits identical to the sequential run"
    fi
done
# without --check the pipeline fuses the MLP tail (PS_MLP); --check keeps the separate steps, so test it here
for mode in "fused:" "fused160:--chunk 160" "nofuse:--no-mlp-fuse"; do
    name=${mode%%:*}; opts=${mode#*:}
    CHK= run "$name" $opts
    if ! cmp -s "$tmp/base.logits" "$tmp/$name.logits"; then
        echo "FAIL $name: logits differ from the sequential run"; diff "$tmp/base.logits" "$tmp/$name.logits" | head -4; fail=1
    else
        echo "PASS $name: logits identical to the sequential run"
    fi
done
rm -rf "$tmp"
[ $fail -eq 0 ] && echo "ALL PASSED" || { echo "FAILED"; exit 1; }
