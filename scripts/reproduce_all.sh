#!/usr/bin/env bash
# End-to-end reproduction script for the EAAI submission.
#
# Phase 1  (original Sensors paper, kept for replication):
#   matrix / LORO / freq / causal-bands / IRM-MLP / IRM-CNN / hierarchical
#   / cross-region / robust-bands / IRM-sweep / augmentation / PGA / ensemble
#
# Phase 2  (EAAI revision E2-E7):
#   sigungu OOD (E2) / extra tabular baselines (E3) / domain alignment (E4)
#   / cost metrics (E5+E7) / compute cost benchmark (E6)
#
# Multi-seed sweep (E1) is run via the dedicated `make multiseed` target.
#
# Failure behaviour: hard-fail on the first non-zero exit so missing artifacts
# can never be hidden from downstream summaries.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

PAPER_DIR="${PAPER_DIR:-paper-eaai}"

mkdir -p results results/figs

run_py() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  "${PYTHON}" "$@" 2>&1 | tee "results/${name}.log"
}

echo "Repository: ${ROOT}"
echo "Python: $(${PYTHON} -c 'import sys; print(sys.executable)')"
echo "Paper directory: ${PAPER_DIR}"
echo "Started: $(date -Is)"

# ---------- Phase 1 ----------
run_py run_matrix         src/run_matrix.py
run_py run_loro           src/run_loro.py
run_py run_freq           src/run_freq_analysis.py
run_py run_causal_bands   src/run_causal_bands.py
run_py run_irm_mlp        src/run_irm_mlp.py
run_py run_irm_cnn        src/run_irm_cnn.py
run_py run_hierarchical   src/run_hierarchical.py
run_py run_cross_region   src/run_cross_region_causal.py
run_py run_robust_bands   src/run_robust_bands.py
run_py run_irm_sweep      src/run_irm_sweep.py
run_py run_augmentation   src/run_augmentation.py
run_py run_pga            src/run_pga.py
run_py run_ensemble       src/run_ensemble.py

# ---------- Phase 2 (EAAI revision) ----------
run_py run_sigungu_ood       src/run_sigungu_ood.py
run_py run_extra_baselines   src/run_extra_baselines.py
run_py run_domain_alignment  src/run_domain_alignment.py
run_py run_cost_metrics      src/run_cost_metrics.py
run_py run_compute_cost      src/run_compute_cost.py

# ---------- analysis + figures ----------
run_py analyze_results       src/analyze_results.py
run_py make_figures          src/make_figures.py
run_py make_fig5_causal      src/make_fig5_causal.py
run_py make_fig6_irm         src/make_fig6_irm.py
run_py make_fig7_robust      src/make_fig7_robust.py
run_py make_fig8_boost       src/make_fig8_boost.py
run_py make_fig9_baselines   src/make_fig9_baselines.py
run_py make_fig10_threshold  src/make_fig10_threshold.py
run_py make_fig11_cost       src/make_fig11_cost.py
run_py make_fig12_multiseed  src/make_fig12_multiseed.py
run_py make_fig13_sigungu    src/make_fig13_sigungu.py
run_py make_fig14_pareto     src/make_fig14_pareto.py

# ---------- manuscript (optional) ----------
# The LaTeX manuscript is not part of the public code release; this step
# runs only when the source tree is present locally.
if [[ -f "${PAPER_DIR}/paper.tex" ]]; then
  echo
  echo "==> paper (${PAPER_DIR})"
  (
    cd "${PAPER_DIR}"
    pdflatex -interaction=nonstopmode -halt-on-error paper.tex
    bibtex paper
    pdflatex -interaction=nonstopmode -halt-on-error paper.tex
    pdflatex -interaction=nonstopmode -halt-on-error paper.tex
    pdflatex -interaction=nonstopmode -halt-on-error highlights.tex
  ) 2>&1 | tee "${ROOT}/results/run_paper.log"
  echo
  echo "Outputs: ${PAPER_DIR}/paper.pdf, ${PAPER_DIR}/highlights.pdf"
else
  echo
  echo "==> paper: skipped (${PAPER_DIR}/ not present in this checkout)"
fi

echo
echo "Finished: $(date -Is)"
