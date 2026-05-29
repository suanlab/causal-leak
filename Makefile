SHELL := /usr/bin/env bash

# Canonical interpreter: shared venv at ../.venv (EAAI revision)
PYTHON ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

# Canonical manuscript target: paper-eaai/ (Engineering Applications of AI submission)
PAPER_DIR := paper-eaai
PAPER_BASE := paper

.PHONY: all experiments multiseed figures paper verify clean-cache help

all: experiments multiseed figures verify

experiments:
	PYTHON="$(PYTHON)" scripts/reproduce_all.sh

# E1 — 5-seed re-runs of every stochastic experiment (IRM/GroupDRO/PGA/aug)
multiseed:
	PYBIN="$(PYTHON)" bash src/run_multiseed.sh
	$(PYTHON) src/aggregate_multiseed.py

# Regenerate every figure (Phase 1 + EAAI revision figs 9-14)
figures:
	$(PYTHON) src/analyze_results.py
	$(PYTHON) src/make_figures.py
	$(PYTHON) src/make_fig5_causal.py
	$(PYTHON) src/make_fig6_irm.py
	$(PYTHON) src/make_fig7_robust.py
	$(PYTHON) src/make_fig8_boost.py
	$(PYTHON) src/make_fig9_baselines.py
	$(PYTHON) src/make_fig10_threshold.py
	$(PYTHON) src/make_fig11_cost.py
	$(PYTHON) src/make_fig12_multiseed.py
	$(PYTHON) src/make_fig13_sigungu.py
	$(PYTHON) src/make_fig14_pareto.py

# Manuscript build — the LaTeX source is not part of the public code
# release, so this target is a no-op unless the source tree is present.
paper:
	@if [ -f $(PAPER_DIR)/$(PAPER_BASE).tex ]; then \
	  cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_BASE).tex \
	    && bibtex $(PAPER_BASE) \
	    && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_BASE).tex \
	    && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_BASE).tex \
	    && pdflatex -interaction=nonstopmode -halt-on-error highlights.tex ; \
	else \
	  echo "paper: skipped ($(PAPER_DIR)/ not present in this checkout)" ; \
	fi

verify:
	PYTHON="$(PYTHON)" PAPER_DIR="$(PAPER_DIR)" scripts/verify.sh

clean-cache:
	rm -rf src/__pycache__

help:
	@echo "Targets:"
	@echo "  experiments  - Phase 1 + EAAI E2-E7 experiments"
	@echo "  multiseed    - E1 5-seed sweep + aggregation"
	@echo "  figures      - Regenerate all figures (fig1-fig14)"
	@echo "  paper        - Build paper-eaai/paper.pdf + highlights.pdf"
	@echo "  verify       - Verify paper-eaai build and claims"
	@echo "  all          - Full end-to-end reproduction"
