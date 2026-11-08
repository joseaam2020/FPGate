#!/bin/bash
# Runs every fuzz target for N seconds each (default 60) from freshly generated seeds.
# A crash leaves its input in build/fuzz/crashes/<target>/ and the script exits non-zero.
# Usage: tests/fuzz/run.sh [seconds]
SECS=${1:-60}
OUT=build/fuzz
SEEDS=$OUT/seeds
TARGETS="http_stream detect image toml creds admin"
fail=0

python3 tests/fuzz/make_corpus.py "$SEEDS" >/dev/null || exit 1
# Files the targets write go here, not into the project.
export TMPDIR=/tmp

for t in $TARGETS; do
    corpus=$OUT/corpus/$t
    crashes=$OUT/crashes/$t
    mkdir -p "$corpus" "$crashes"
    case $t in
        image) maxlen=65536 ;;
        admin) maxlen=16384 ;;
        *)     maxlen=4096 ;;
    esac
    echo "== $t ($SECS s)"
    # clang 14's AddressSanitizer dies at start-up on some runs when the kernel randomises
    # memory with 32 bits (WSL2, Docker Desktop). That shows as a segfault with an empty
    # log, before any input ran; it is not a finding, so such runs are simply repeated.
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        ASAN_OPTIONS=detect_leaks=1:abort_on_error=0         "$OUT/$t" "$corpus" "$SEEDS/$t" -max_total_time="$SECS" -max_len=$maxlen -timeout=20             -artifact_prefix="$crashes/" -print_final_stats=1 2>"$OUT/$t.log"
        rc=$?
        if [ $rc -eq 139 ] && ! grep -q "INFO: Seed" "$OUT/$t.log"; then
            echo "   (start-up crash of the sanitizer runtime, run $attempt; trying again)"
            continue
        fi
        break
    done
    grep -E "^(#[0-9]+\s+DONE|stat::number_of_executed_units|stat::new_units_added|Done )|cov:.*DONE" "$OUT/$t.log" | tail -3
    if [ $rc -ne 0 ] || ls "$crashes"/crash-* "$crashes"/leak-* "$crashes"/timeout-* "$crashes"/oom-* >/dev/null 2>&1; then
        echo "FAIL $t (exit $rc); see $OUT/$t.log and $crashes"
        tail -25 "$OUT/$t.log"
        fail=1
    else
        echo "PASS $t: no crash, leak or timeout in $SECS s"
    fi
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit $fail
