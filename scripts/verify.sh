#!/usr/bin/env bash
# Verification gate for the DMKD article (Identifier Leakage and
# Selection-Defined Classes in a Sensor Benchmark).
#
# Code-level checks always run and need neither the AI Hub data nor the
# manuscript: the sources compile, retired phrases are absent, every figure
# the article uses exists, and src/cited_numbers.py prints every cited number
# from the committed result files. When the manuscript source is present
# (paper-dmkd/, kept locally), its build log is checked and its claim checker
# is run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PAPER_DIR="${PAPER_DIR:-paper-dmkd}"
PAPER_BASE="${PAPER_BASE:-paper}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "../.venv/bin/python" ]]; then
    PYTHON="../.venv/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  elif command -v python3 >/dev/null; then
    PYTHON="python3"
  else
    echo "ERROR: no Python interpreter found." >&2
    exit 1
  fi
fi

"${PYTHON}" -m compileall -q src

PAPER_LOG="${PAPER_DIR}/${PAPER_BASE}.log"
PAPER_PDF="${PAPER_DIR}/${PAPER_BASE}.pdf"
PAPER_TEX="${PAPER_DIR}/${PAPER_BASE}.tex"

# --- code-level checks (always run) ---

# 1. stale phrase guard — strings retired in earlier revisions.
shopt -s nullglob
STALE_TARGETS=(README.md results/*.md src/*.py scripts/reproduce_all.sh)
shopt -u nullglob
[[ -f "${PAPER_TEX}" ]] && STALE_TARGETS+=("${PAPER_TEX}")
if grep -n -E 'weighted_avg|cnn_best|make_fig0|bibtex cycle|final deployment recipe' \
   "${STALE_TARGETS[@]}"; then
  echo "FAIL: stale phrase still present in repository" >&2
  exit 1
fi

# 2. figures used by the article
for fig in figB1_ladder_alarm figB2_dispatch figB3_models_alarm figB5_operating; do
  if [[ ! -s "results/figs/${fig}.pdf" ]]; then
    echo "FAIL: results/figs/${fig}.pdf missing or empty" >&2
    exit 1
  fi
done

# 3. every cited number is computable from the committed result files
if ! "${PYTHON}" src/cited_numbers.py > /dev/null; then
  echo "FAIL: src/cited_numbers.py could not compute the cited numbers" >&2
  exit 1
fi

# --- manuscript checks (only when the LaTeX source is present locally) ---
if [[ -f "${PAPER_TEX}" ]]; then
  if [[ ! -f "${PAPER_LOG}" || ! -f "${PAPER_PDF}" ]]; then
    echo "ERROR: ${PAPER_DIR} build artifacts missing — run 'make paper' first." >&2
    exit 1
  fi
  if grep -n -E 'undefined|Author undefined' "${PAPER_LOG}"; then
    echo "FAIL: undefined references or citations in ${PAPER_LOG}" >&2
    exit 1
  fi
  if pdftotext "${PAPER_PDF}" - | grep -n -E 'Figure \?|Table \?|Section \?|\[\?\]|\?\?'; then
    echo "FAIL: ${PAPER_PDF} contains rendered '?' placeholders" >&2
    exit 1
  fi
  if [[ -f "${PAPER_DIR}/verify_claims.py" ]]; then
    PYABS="${PYTHON}"
    [[ "${PYTHON}" == */* ]] && PYABS="$(cd "$(dirname "${PYTHON}")" && pwd)/$(basename "${PYTHON}")"
    (cd "${PAPER_DIR}" && "${PYABS}" verify_claims.py)
  fi
else
  echo "Manuscript source (${PAPER_DIR}/) not in this checkout — skipping manuscript checks."
fi

echo "Verification passed."
