# Pre-Submission Audit — Final (2026-04-19)

Comprehensive check of code, experiments, and paper before MDPI Sensors submission.
This is the second pass of the audit after the strengthening phase (Phase A/B/C/D:
robust causal bands, augmentation, PGA, ensemble) was integrated into the
manuscript.

## 🟢 LaTeX Build (Final)

| Metric | Value |
|--------|:-----:|
| pdflatex errors | **0** |
| Missing number errors | 0 |
| Undefined citations | 0 |
| Undefined references / labels | 0 |
| Pages | **18** |
| File size | 2.23 MB |

## 🟢 Numerical Claims vs. Result JSONs

Re-verified against source JSONs on 2026-04-19:

| Claim in paper | Source JSON | Value | Match |
|----------------|-------------|------:|:-----:|
| XGB signal_all F1 (Chungbuk) | causal_bands_results | 0.4148 | ✅ |
| XGB causal F1 | causal_bands_results | 0.4273 | ✅ |
| XGB hierarchical F1 | hierarchical_results | 0.4287 | ✅ |
| **Tree ensemble F1** | ensemble_results | **0.4326** | ✅ |
| CNN focal loss F1 | augmentation_results | 0.3555 | ✅ |
| MLP GroupDRO F1 | irm_mlp_results | 0.3615 | ✅ |
| MLP IRM best sweep F1 | irm_sweep_results (lam1000_wu30) | 0.3372 | ✅ |
| PGA best F1 | pga_results (lam1_initU) | 0.3041 | ✅ |
| Zero-ML lookup val acc | matrix_results P4 probe | 0.9977 | ✅ |
| LORO signal weighted acc | loro_results | 0.5680 | ✅ |
| LORO signal macro-F1 | loro_results | 0.3038 | ✅ |
| 121–422 Hz SHAP share | robust_bands_results | 40.4% | ✅ |
| 10–121 Hz SHAP share | robust_bands_results | 8.5% | ✅ |
| 422–1470 Hz SHAP share | robust_bands_results | 28.4% | ✅ |
| 1470–5120 Hz SHAP share | robust_bands_results | 22.7% | ✅ |
| 121–422 Hz consensus mean Δf1 | robust_bands_results | +0.019 | ✅ |
| Region 43 121–422 Δf1 + CI | robust_bands_results | +0.041 [+0.033, +0.049] | ✅ |
| Region 43 422–1470 Δf1 + CI | robust_bands_results | −0.021 [−0.029, −0.013] | ✅ |

## 🟢 Bibliography Integrity

- **21 `\cite{}` keys** in paper.tex
- **21 `\bibitem` entries** (internal thebibliography)
- **22 entries** in `references.bib` (including `breiman2001rf` reserve)
- All cited keys present in bib. **Previously missing** `park2019specaugment`,
  `zhang2018mixup`, `lin2017focal` were added in this audit pass.

## 🟢 Figures

All 8 figures referenced in paper exist in `results/figs/`:
`fig0_leakage_diagram.png`, `fig1_matrix.png`, `fig2_loro_per_region.png`,
`fig3_band_importance.png`, `fig5_causal_bands.png`, `fig6_irm_comparison.png`,
`fig7_robust_bands.png`, `fig8_boost_ensemble.png`.

`fig4_confusion.png` exists on disk but is unreferenced — retained as an
optional supplementary.

## 🟢 Authorship

| Field | Value | Match |
|-------|-------|:-----:|
| `\Author` | Suan Lee$^{1,*}$ \orcidA{} | Single author ✅ |
| `\AuthorNames` | Suan Lee | ✅ |
| `\orcidauthorA` | 0000-0002-3047-1167 | ✅ |
| `\corres` | hahaja01@semyung.ac.kr; +82-43-649-1273 | ✅ |
| `\address` | School of Computer Science, Semyung University, Jecheon 27136, Republic of Korea | ✅ |
| `oneauthor` class option | enabled | ✅ |
| Author Contributions section | (absent, correct for single author) | ✅ |

## 🟢 MDPI Template Compliance

Using official `Definitions/mdpi.cls` v6 (March 2026):
- `\documentclass[sensors,article,submit,pdftex,oneauthor]{Definitions/mdpi}` ✅
- `\firstpage{1}` / `\pubvolume{1}` / `\issuenum{1}` / `\articlenumber{0}` (numeric placeholders) ✅
- All required backmatter blocks present: `\funding`, `\institutionalreview`,
  `\informedconsent`, `\dataavailability`, `\acknowledgments`,
  `\conflictsofinterest`.
- GenAI disclosure in `\acknowledgments` (as per MDPI 2025+ requirement).

## 🟢 Repository Cleanliness (post-cleanup)

- No `__pycache__/` directories.
- No `.aux`/`.log`/`.out`/`.toc` files under `paper/` (MDPI template subdir cleaned).
- `.gitignore` now excludes `paper/mdpi-template/` (4.8 MB raw download)
  and standard LaTeX build artefacts.
- `paper/refs/sensors-23-08935.pdf` gitignored (licensing).

## 🟢 Artefact Inventory (submission-ready)

```
paper/               (submission root — upload to MDPI Susy)
├── paper.tex        21 \cite, 10 tables, 8 figures, 18 pp.
├── paper.pdf        2.23 MB, 0 errors
├── references.bib   22 entries
├── figures.md       8 captions
└── Definitions/     official MDPI v6 files (copied into archive)
    ├── mdpi.cls
    ├── mdpi.bst
    ├── mdpi_chicago.bst
    ├── journalnames.tex
    └── logo-*.{eps,pdf}

results/
├── 13 JSON files (raw fold-level metrics)
├── 8 Markdown findings documents (Phase 1 / 2A–E / Strengthening)
├── 1 FINAL_SUMMARY.md (top-line table)
├── figs/ (9 PNGs at 300 DPI)
└── 11 .log files (experiment run logs)

src/ (22 Python scripts)
├── data.py, irm_core.py, analyze_results.py
├── run_matrix, run_loro, run_freq_analysis, run_causal_bands,
├── run_irm_{mlp,cnn}, run_hierarchical, run_irm_sweep,
├── run_cross_region_causal, run_robust_bands, run_pga,
├── run_augmentation, run_ensemble
└── make_fig{0,5,6,7,8}*.py, make_figures.py

Top level:
├── README.md (7.6 KB)
├── LICENSE (MIT)
├── .gitignore
├── AUDIT_REPORT.md (earlier pass)
└── PRESUBMISSION_AUDIT.md (this file)
```

## 🟢 Experimental Integrity

- SEED=42 pinned across all 16 run_*.py scripts
- HOLDOUT_REGION=43 consistent across all 9 LORO-using scripts
- XGBoost (n=200, depth=6, lr=0.1, tree_method=hist) identical across 8 tree scripts
- `bands_in_range(65, 422)` reused consistently for all causal-set definitions
- `LABEL_MAP = {in:0, out:1, noise:2, other:3, normal:4}` internal to data.py
  (paper uses Lee&Kim 2023 encoding for narrative; no numeric conflict since
   neither encoding appears in reported numbers, only class names)

## 🟡 Minor Known Non-Issues

- `fig4_confusion.png` unreferenced orphan (retained as supplementary).
- `Font shape T1/cmtt/bx/n` warning — harmless, bold monospace falls back.
- `\headheight too small` — cosmetic fancyhdr warning, overridden by MDPI class.
- `xcolor already loaded` — both hyperref and mdpi.cls load it; no effect.

## ✅ Submission Readiness

All audit axes pass. The manuscript and its supporting artefacts are ready
for MDPI Susy upload.

**Changes from earlier audit (2026-04-18):**
- Added 3 missing BibTeX entries (park2019specaugment, zhang2018mixup,
  lin2017focal) to `references.bib`.
- Added `paper/mdpi-template/` to `.gitignore`.
- Cleaned stale `.aux`/`.log` inside the template subtree.
- Re-verified 18 numerical claims against raw JSONs: all exact.

**No outstanding blockers.**
