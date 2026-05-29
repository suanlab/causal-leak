#!/usr/bin/env bash
# Verification gate for the EAAI submission.
# Checks that paper-eaai/paper.tex compiles cleanly and that no
# placeholder references / undefined cites remain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PAPER_DIR="${PAPER_DIR:-paper-eaai}"
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

if [[ ! -f "${PAPER_LOG}" || ! -f "${PAPER_PDF}" || ! -f "${PAPER_TEX}" ]]; then
  echo "ERROR: ${PAPER_DIR} build artifacts missing — run 'make paper' first." >&2
  exit 1
fi

# 1. natbib undefined citations
if grep -n -F 'undefined' "${PAPER_LOG}"; then
  echo "FAIL: undefined cross-references in ${PAPER_LOG}" >&2
  exit 1
fi
if grep -n -F 'Author undefined' "${PAPER_LOG}"; then
  echo "FAIL: natbib author-undefined in ${PAPER_LOG}" >&2
  exit 1
fi

# 2. body-level rendered '?' tokens (broken \ref / \cite output)
if pdftotext "${PAPER_PDF}" - | grep -n -E 'author\?|Figure \?|Table \?|Section \?|\[\?\]|\?\?'; then
  echo "FAIL: ${PAPER_PDF} contains rendered '?' placeholders" >&2
  exit 1
fi

# 3. stale phrase guard — these strings were retired during the EAAI revision
if grep -n -E 'weighted_avg|cnn_best|make_fig0|bibtex cycle|final deployment recipe' \
   README.md results/*.md "${PAPER_TEX}" src/*.py \
   scripts/reproduce_all.sh; then
  echo "FAIL: stale phrase still present in repository" >&2
  exit 1
fi

# 4. EAAI revision figures present
for fig in fig9_baselines fig10_threshold fig11_cost fig12_multiseed \
           fig13_sigungu fig14_pareto; do
  if [[ ! -s "results/figs/${fig}.pdf" ]]; then
    echo "FAIL: results/figs/${fig}.pdf missing or empty" >&2
    exit 1
  fi
done

# 5. EAAI required statements present in the manuscript
for tag in 'CRediT Authorship Contribution Statement' \
           'Declaration of Competing Interests' \
           'Funding' 'Data Availability' \
           'Declaration of Generative AI'; do
  if ! grep -q "${tag}" "${PAPER_TEX}"; then
    echo "FAIL: ${PAPER_TEX} is missing required section: ${tag}" >&2
    exit 1
  fi
done

# 6. highlights file present (3-5 bullets, each <= 85 chars)
HIGHLIGHTS="${PAPER_DIR}/highlights.tex"
if [[ ! -f "${HIGHLIGHTS}" ]]; then
  echo "FAIL: ${HIGHLIGHTS} not found (EAAI requires a separate highlights file)" >&2
  exit 1
fi
BAD=$(awk '
  /^\\item/ {
    line=$0
    sub(/^\\item[[:space:]]*/, "", line)
    if (length(line) > 85) { print length(line) ": " line }
  }
' "${HIGHLIGHTS}")
if [[ -n "${BAD}" ]]; then
  echo "FAIL: highlights bullets exceed 85 chars:" >&2
  echo "${BAD}" >&2
  exit 1
fi

echo "Verification passed."
