#!/bin/sh
# Generates the FPGate CA: an EC P-256 key encrypted with a password (RS-02)
# and a self-signed certificate. Usage: tools/gen_ca.sh [outdir] [days]
# The password comes from $FPGATE_CA_PASS, or is asked for interactively.
set -e
OUT=${1:-.}
DAYS=${2:-365}
mkdir -p "$OUT"

if [ -n "$FPGATE_CA_PASS" ]; then
    PASS_OPT="-passout env:FPGATE_CA_PASS"
else
    PASS_OPT=""
fi

# shellcheck disable=SC2086
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -sha256 \
    -days "$DAYS" -keyout "$OUT/ca.key" -out "$OUT/ca.crt" \
    -subj "/CN=FPGate CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    $PASS_OPT
chmod 600 "$OUT/ca.key"
echo "CA written to $OUT (ca.crt is public; ca.key must never leave the proxy)"
