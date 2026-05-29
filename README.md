# causal-leak

**A causal audit and out-of-distribution redesign of Lee & Kim (2023)**
*Machine Learning Model for Leak Detection Using Water Pipeline Vibration
Sensor.* Sensors **23**(21), 8935. [DOI: 10.3390/s23218935](https://doi.org/10.3390/s23218935)

This repository is the follow-up study to our 2023 Sensors paper. It
quantifies how much of the reported 99.79 % XGBoost accuracy on the
Korean AI Hub water-leak dataset is driven by identifier-label
leakage, derives the causal frequency signature of real leak noise
(121–422 Hz), and benchmarks invariance-regularised deep models
against causal-band XGBoost under a strict Leave-One-Region-Out
protocol.

## Headline result

| Model | # feat | OOD Acc | OOD Macro-F1 |
|-------|------:|-------:|------:|
| 5-line lookup (`sid` → majority + `lrate` override) | 0 | 0.9977 | – |
| XGBoost full (paper split) | 544 | 0.9402 | 0.9274 |
| XGBoost signal_all (LORO Chungbuk) | 534 | 0.2978 | 0.4148 |
| XGBoost causal 65–422 Hz (LORO Chungbuk) | 57 | 0.3013 | 0.4273 |
| Hierarchical (normal gate + causal typer) | 57 | 0.3041 | 0.4287 |
| **Pre-specified tree-only soft average (causal + hier)** | **57** | **0.3096** | **0.4326** |
| MLP GroupDRO | 57 | 0.3210 | 0.3615 |
| CNN + SpecAugment (validation-selected CNN recipe) | 534 | 0.3384 | 0.3544 |
| MLP IRM (validation-selected sweep) | 57 | 0.2931 | 0.3145 |
| PGA (best observed test ablation, λc=1) | 534 | 0.2897 | 0.3041 |

**See `paper-eaai/paper.tex` (the canonical submission target, formatted
for *Engineering Applications of Artificial Intelligence*) for the
full write-up. `paper/` contains the earlier Sensors-targeted draft
and is kept for reference only.**

## Eight key findings

1. **Zero-ML reproduction.** An `sid → majority-train-label` lookup
   with an `lrate ≠ 90 → normal` override yields **99.77 %** val
   accuracy — within 0.02 pp of our published 99.79 %. The 99.79 %
   headline is a split-protocol artefact, not a demonstration of
   vibration-spectrum capability.
2. **Shortcut collapse.** Switching to stratified 5-fold GroupKFold by
   site drops accuracy from 94 % to **55 %** (macro-F1 0.42). 12-fold
   LORO weighted accuracy is **56.8 %** (macro-F1 0.30).
3. **Causal frequency signature.** Grouped permutation importance on
   the only region with a realistic 5-class balance (Chungbuk)
   localises the generalisable leak signal to **121–422 Hz**, with
   **422–1470 Hz an active confounder** that our original XGBoost
   over-relied on.
4. **Causal feature selection beats invariance learning.** Restricting
   to 57 causal features improves OOD macro-F1 by +0.013 over full
   spectrum. No gradient-based invariance method (IRM, GroupDRO) on
   MLP or 1D-CNN closes the remaining gap.
5. **Triple-source convergence for 121–422 Hz.** Bootstrap 95% CIs
   across 4 LORO folds (wins 3/4), SHAP (40.4% attribution mass),
   and XGBoost gain (7.5%, mis-allocated) independently converge on
   121–422 Hz as the causal leak-noise band.
6. **Augmentation helps but cannot close the gap.** SpecAugment improves the
   CNN baseline from F1 0.334 to 0.354 (+2.0 pp); focal loss mainly improves
   accuracy; Mixup hurts on this dataset.
7. **PGA negative result.** A novel permutation-guided attention
   regulariser cannot beat plain feature selection, reinforcing the
   main finding.
8. **Deployment candidate.** A pre-specified tree-only soft average of
   causal flat + hierarchical XGBoost gives OOD macro-F1 (**0.4326**)
   with perfect OOD `normal` recall.

## Layout

```
causal-leak/
├── README.md                         this file
├── LICENSE                           MIT
├── REPLICATION.md                    full end-to-end recipe
├── EAAI_AUDIT.md                     pre-submission audit checklist
├── Dockerfile                        pinned environment
├── requirements.txt                  base Python deps
├── requirements-eaai.txt             EAAI revision deps
├── Makefile                          targets paper-eaai by default
├── paper-eaai/                       canonical submission (EAAI)
│   ├── paper.tex                     elsarticle single-column
│   ├── paper.pdf                     latest build
│   ├── highlights.tex                EAAI highlights file
│   ├── references.bib                41 verified entries
│   ├── claims.csv                    numeric-claim tracker
│   └── EAAI_AUDIT.md                 guide-for-authors checklist
├── paper/                            archival Sensors-targeted draft
├── scripts/
│   ├── reproduce_all.sh              full experiment + paper-eaai pipeline
│   └── verify.sh                     paper-eaai build / claim consistency
├── src/
│   ├── data.py                       loader (data root via $CAUSAL_LEAK_DATA_DIR)
│   ├── irm_core.py                   IRM / GroupDRO infrastructure
│   ├── run_matrix.py                 Phase 1 — shortcut matrix
│   ├── run_loro.py                   Phase 1 — LORO baselines
│   ├── run_freq_analysis.py          Phase 1 — band importance
│   ├── run_causal_bands.py           Phase 1 — causal-band retraining
│   ├── run_irm_mlp.py                Phase 1 — MLP ERM/IRM/GroupDRO
│   ├── run_irm_cnn.py                Phase 1 — CNN ERM/IRM/GroupDRO
│   ├── run_hierarchical.py           Phase 1 — hierarchical classifier
│   ├── run_robust_bands.py           Phase 1 — bootstrap + SHAP
│   ├── run_irm_sweep.py              Phase 1 — IRM lambda sweep
│   ├── run_pga.py                    Phase 1 — Permutation-Guided Attention
│   ├── run_augmentation.py           Phase 1 — CNN augmentation recipes
│   ├── run_ensemble.py               Phase 1 — soft-probability ensemble
│   ├── run_sigungu_ood.py            EAAI E2 — 시군구 LOCO
│   ├── run_extra_baselines.py        EAAI E3 — LightGBM/CatBoost/TabNet/RF
│   ├── run_domain_alignment.py       EAAI E4 — DeepCORAL/DANN/MMD
│   ├── run_cost_metrics.py           EAAI E5+E7 — calibration-only FAR / sweep
│   ├── run_compute_cost.py           EAAI E6 — train/infer/RAM/GPU benchmark
│   ├── run_multiseed.sh              EAAI E1 — 5-seed sweep wrapper
│   ├── aggregate_multiseed.py        EAAI E1 — mean ± std + Wilcoxon
│   ├── analyze_results.py            summary tables
│   └── make_fig*.py                  figure generators (1-14, vector PDF)
└── results/
    ├── FINAL_SUMMARY.md              top-line result table
    ├── FINDINGS_EAAI_E1_E7.md        EAAI revision findings
    ├── *.json                        raw fold-level metrics
    ├── *.log                         training logs
    ├── multiseed_logs/               per-seed-script logs (E1)
    └── figs/                         manuscript figures (fig1-fig14)
```

## Reproducing

```bash
# Environment (Python 3.12, CUDA 12.1 torch)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-eaai.txt

# Data — download from AI Hub (https://www.aihub.or.kr/, dataset 138)
# and either:
#   (a) place under ../kcc-water/Dataset/ so the default path works, or
#   (b) export CAUSAL_LEAK_DATA_DIR=/your/data/root before running.

# Full canonical pipeline targeting paper-eaai/ (Phase 1 + EAAI E2-E7
# + figures + LaTeX build). Hard-fails on any non-zero exit.
scripts/reproduce_all.sh

# Multi-seed sweep (E1, ~6 h on 2 GPUs):
make multiseed

# Explicit Python override:
PYTHON=.venv/bin/python scripts/reproduce_all.sh

# Verification only (no rebuild):
scripts/verify.sh

# Make targets:
make all          # experiments + multiseed + figures + paper + verify
make experiments  # Phase 1 + EAAI E2-E7
make multiseed    # EAAI E1 (5 seeds)
make figures      # regenerate every figure (fig1-fig14)
make paper        # build paper-eaai/paper.pdf + highlights.pdf
make verify       # gate-check paper-eaai build and claim consistency
```

See `REPLICATION.md` for the full end-to-end recipe (expected
runtimes, troubleshooting, Docker invocation).

## Citing this work

If you build on this follow-up, please cite both the original Sensors
paper and this repository:

```bibtex
@article{lee2023leak,
  author  = {Suan Lee and Byeonghak Kim},
  title   = {Machine Learning Model for Leak Detection Using Water Pipeline Vibration Sensor},
  journal = {Sensors},
  volume  = {23}, number = {21}, pages = {8935},
  year    = {2023},
  doi     = {10.3390/s23218935}
}

@misc{lee2026causal,
  author = {Suan Lee and Byeonghak Kim},
  title  = {From 99.79\% to Field Deployment: A Causal Audit and
            Out-of-Distribution Redesign of Our Water-Pipeline Leak
            Detection Pipeline},
  year   = {2026},
  note   = {In preparation. Code and data at causal-leak/}
}
```

## License

Code: MIT (see `LICENSE`).
Vibration data: separate license from AI Hub; not redistributed here.
