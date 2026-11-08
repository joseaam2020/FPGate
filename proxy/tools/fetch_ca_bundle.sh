#!/bin/sh
# Downloads the Mozilla CA bundle (via curl.se) used by the proxy to verify
# the real servers. Usage: tools/fetch_ca_bundle.sh [outfile]
set -e
OUT=${1:-certs/ca-bundle.pem}
mkdir -p "$(dirname "$OUT")"
curl -fsSL https://curl.se/ca/cacert.pem -o "$OUT"
echo "wrote $OUT ($(grep -c 'BEGIN CERTIFICATE' "$OUT") certificates)"
