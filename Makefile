SHELL := /usr/bin/env bash

# Canonical interpreter: shared venv at ../.venv
PYTHON ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

# Manuscript target: paper-dmkd/ (Data Mining and Knowledge Discovery). The
# LaTeX source is kept locally and is not part of the public release.
PAPER_DIR := paper-dmkd
PAPER_BASE := paper

.PHONY: all reproduce experiments multiseed figures paper verify numbers \
        provenance cwru clean-cache help

all: experiments multiseed figures verify

# One command for the article: data lineage, public bearing data, every
# experiment and figure, then the cited numbers and the gate.
reproduce: provenance cwru experiments numbers verify

provenance:
	$(PYTHON) src/check_provenance.py

cwru:
	scripts/fetch_cwru.sh

numbers:
	$(PYTHON) src/cited_numbers.py

experiments:
	PYTHON="$(PYTHON)" scripts/reproduce_all.sh

# E1 — 5-seed re-runs of every stochastic experiment (IRM/GroupDRO/PGA/aug)
multiseed:
	PYBIN="$(PYTHON)" bash src/run_multiseed.sh
	$(PYTHON) src/aggregate_multiseed.py

# Regenerate every figure
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
	$(PYTHON) src/make_figB1_ladder.py
	$(PYTHON) src/make_figB2_falsealarms.py
	$(PYTHON) src/make_figB3_models.py
	$(PYTHON) src/make_figB4_tradeoff.py
	FIG_TERM=alarm $(PYTHON) src/make_figB1_ladder.py
	FIG_TERM=alarm $(PYTHON) src/make_figB3_models.py
	$(PYTHON) src/make_figB2_dispatch.py
	$(PYTHON) src/make_figB5_operating.py

# Manuscript build — the LaTeX source is not part of the public code
# release, so this target is a no-op unless the source tree is present.
paper:
	@if [ -f $(PAPER_DIR)/$(PAPER_BASE).tex ]; then \
	  cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_BASE).tex \
	    && bibtex $(PAPER_BASE) \
	    && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_BASE).tex \
	    && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_BASE).tex ; \
	else \
	  echo "paper: skipped ($(PAPER_DIR)/ not present in this checkout)" ; \
	fi

verify:
	PYTHON="$(PYTHON)" PAPER_DIR="$(PAPER_DIR)" scripts/verify.sh

clean-cache:
	rm -rf src/__pycache__

help:
	@echo "Targets:"
	@echo "  reproduce    - provenance + CWRU + experiments + cited numbers + gate"
	@echo "  provenance   - hash and row-check the AI Hub release (needs the data)"
	@echo "  cwru         - fetch the public CWRU bearing data"
	@echo "  experiments  - every experiment (scripts/reproduce_all.sh)"
	@echo "  numbers      - print every number the article cites"
	@echo "  multiseed    - 5-seed sweep + aggregation"
	@echo "  figures      - regenerate all figures"
	@echo "  paper        - build paper-dmkd/paper.pdf (local source only)"
	@echo "  verify       - code gate; manuscript checks when the source is present"
	@echo "  all          - experiments + multiseed + figures + verify"
