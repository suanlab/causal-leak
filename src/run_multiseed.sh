#!/usr/bin/env bash
# Phase 1 / E1 — Multi-seed re-runs of every stochastic experiment.
#
# For each seed in {0, 1, 2, 3, 42}:
#   - run_irm_mlp.py        (ERM / IRM / GroupDRO on 57 causal features)
#   - run_irm_cnn.py        (ERM / IRM / GroupDRO on raw 513 spectrum)
#   - run_pga.py            (Permutation-Guided Attention sweep)
#   - run_augmentation.py   (5 CNN recipes + 1 MLP recipe)
#
# Failure behaviour: hard fail on any non-zero exit. Existing
# `<base>_seed<S>.json` artefacts (and the un-suffixed seed-42 artefacts)
# are reused via `SKIP_EXISTING=1`; pass `FORCE=1` to re-run everything.
#
# Interpreter resolution: $PYBIN if set, else ../.venv/bin/python, else
# .venv/bin/python, else `python3`.
set -euo pipefail

SEEDS=(42 0 1 2 3)
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SRC_DIR")"
LOG_DIR="$ROOT_DIR/results/multiseed_logs"
RESULTS_DIR="$ROOT_DIR/results"
mkdir -p "$LOG_DIR"

if [[ -z "${PYBIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/../.venv/bin/python" ]]; then
    PYBIN="$ROOT_DIR/../.venv/bin/python"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYBIN="$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null; then
    PYBIN="python3"
  else
    echo "ERROR: no Python interpreter found; set \$PYBIN explicitly." >&2
    exit 1
  fi
fi

SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"

artifact_for() {
    local script="$1"
    local seed="$2"
    local base
    base="$(basename "$script" .py | sed 's/^run_//')"
    if [[ "$seed" == "42" ]]; then
        echo "$RESULTS_DIR/${base}_results.json"
    else
        echo "$RESULTS_DIR/${base}_results_seed${seed}.json"
    fi
}

run_one() {
    local seed="$1"
    local script="$2"
    local tag
    tag="$(basename "$script" .py)"
    local log="$LOG_DIR/${tag}_seed${seed}.log"
    local artefact
    artefact="$(artifact_for "$script" "$seed")"

    if [[ "$FORCE" != "1" && "$SKIP_EXISTING" == "1" && -s "$artefact" ]]; then
        echo "[$(date +%H:%M:%S)] seed=$seed  $tag  SKIP (artefact exists: $artefact)"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] seed=$seed  $tag  ->  $log"
    CAUSAL_SEED="$seed" "$PYBIN" -u "$SRC_DIR/$script" > "$log" 2>&1
}

START=$(date +%s)
echo "=== Multi-seed sweep launched at $(date) ==="
echo "PYBIN=$PYBIN  SKIP_EXISTING=$SKIP_EXISTING  FORCE=$FORCE"
for seed in "${SEEDS[@]}"; do
    echo "--- seed $seed ---"
    run_one "$seed" run_irm_mlp.py
    run_one "$seed" run_augmentation.py
    run_one "$seed" run_pga.py
    run_one "$seed" run_irm_cnn.py
done
END=$(date +%s)
echo "=== Multi-seed sweep finished in $((END - START)) s ==="

# Verify all expected artefacts exist before aggregation.
missing=0
for seed in "${SEEDS[@]}"; do
    for script in run_irm_mlp.py run_irm_cnn.py run_pga.py run_augmentation.py; do
        a="$(artifact_for "$script" "$seed")"
        if [[ ! -s "$a" ]]; then
            echo "MISSING artefact: $a" >&2
            missing=$((missing + 1))
        fi
    done
done
if [[ $missing -gt 0 ]]; then
    echo "FAIL: $missing artefacts missing — aggregation aborted." >&2
    exit 1
fi

echo "=== Aggregating multi-seed summary ==="
"$PYBIN" -u "$SRC_DIR/aggregate_multiseed.py" > "$LOG_DIR/aggregate.log" 2>&1
echo "=== Done ==="
