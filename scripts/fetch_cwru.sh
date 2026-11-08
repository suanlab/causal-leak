#!/usr/bin/env bash
# Fetch the Case Western Reserve University bearing data used by
# src/run_audit_semisynthetic.py: 12 kHz drive-end recordings of the normal
# baseline and of seeded inner-race, ball and outer-race (centred, 6 o'clock)
# faults of 0.007, 0.014 and 0.021 inch, each at motor loads 0-3 hp.
# Source: https://engineering.case.edu/bearingdatacenter (public).
# Destination: $CWRU_DIR (default: ../external/cwru relative to the repo).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CWRU_DIR:-$HERE/../external/cwru}"
mkdir -p "$DEST"
IDS="97 98 99 100 105 106 107 108 118 119 120 121 130 131 132 133 \
169 170 171 172 185 186 187 188 197 198 199 200 \
209 210 211 212 222 223 224 225 234 235 236 237"
for id in $IDS; do
  f="$DEST/$id.mat"
  if [ -s "$f" ] && [ "$(stat -c %s "$f")" -gt 100000 ]; then continue; fi
  curl -fsS -m 120 -o "$f.part" "https://engineering.case.edu/sites/default/files/$id.mat"
  size=$(stat -c %s "$f.part")
  if [ "$size" -lt 100000 ]; then
    echo "fetch_cwru: $id.mat is only $size bytes; aborting" >&2
    exit 1
  fi
  mv "$f.part" "$f"
done
n=$(ls "$DEST"/*.mat | wc -l)
echo "fetch_cwru: $n files in $DEST"
[ "$n" -eq 40 ] || { echo "fetch_cwru: expected 40 files" >&2; exit 1; }
