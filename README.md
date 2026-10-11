# causal-leak

**An audit and out-of-distribution re-evaluation of Lee & Kim (2023)**
*Machine Learning Model for Leak Detection Using Water Pipeline Vibration
Sensor.* Sensors **23**(21), 8935. [DOI: 10.3390/s23218935](https://doi.org/10.3390/s23218935)

This repository is the follow-up to our 2023 Sensors paper. It measures how
much of the reported 99.79 % accuracy on the Korean AI Hub water-leak benchmark
comes from recognising which sensor produced a recording, evaluates leak typing
with whole administrative regions held out, and compares models on the false
alarms a utility would have to dispatch crews for.

It is the reproducibility harness for a manuscript in preparation: experiment
code, raw result files, logs, and the scripts that render every figure.

> **Correction (October 2026).** An earlier version of this repository reported
> a 121–422 Hz "causal band" as carrying the generalisable leak signal. That band
> was selected using feature importance measured on the evaluation fold and then
> evaluated on the same fold. Under nested selection, with the evaluation region
> excluded from the choice, it is selected in 2 of 7 regions. Every result that
> depended on the resulting 57-feature set has been re-run; the retracted
> artefacts are kept in `results/retracted_fixed_band_65_422/`. See
> `results/FINDINGS_NESTED_BANDS.md`.

## Headline results

Five-class leak typing, gradient-boosted classifier unless stated.

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Sensor-identifier lookup (no spectrum used) | 0.9977 | – |
| Row-level cross-validation, with identifier columns | 0.945 | 0.933 |
| Group-CV by installation | 0.553 | 0.419 |
| Leave-one-region-out, signal only (weighted) | 0.568 | 0.304 |

Canonical held-out region (Chungbuk, region 43), leakage-free features:

| Model | Macro-F1 |
|---|---:|
| LightGBM | 0.4464 |
| Tree-only soft average (flat + hierarchical) | 0.4388 |
| Gradient-boosted, flat | 0.4363 |
| Gradient-boosted, hierarchical | 0.4335 |
| CNN, ERM (5 seeds) | 0.3455 ± 0.027 |
| MLP, GroupDRO (5 seeds) | 0.3115 ± 0.017 |
| DeepCORAL (1 seed) | 0.3100 |

## Findings

1. **Identifier shortcut.** Every training sensor carries a single label, and
   98 % of validation sensors also appear in training. A lookup on the sensor
   identifier plus one metadata rule reaches 99.77 % validation accuracy without
   reading the vibration spectrum — within 0.02 points of the published 99.79 %.
2. **Collapse under realistic evaluation.** Accuracy falls from 0.945 under
   row-level cross-validation to 0.553 with installations held out, and to a
   weighted 0.568 (macro-F1 0.304) with regions held out. Identifier columns are
   worth 5.3 points at row level and nothing once sensors are unseen.
3. **No stable "causal band".** With band selection nested inside the training
   regions, the 121–422 Hz band is selected in 2 of 7 regions. Selecting along a
   temporal axis instead retains it, so a band's generalisation claim depends on
   the axis named.
4. **Complexity does not help.** Neither invariance regularisers (IRM, GroupDRO;
   five seeds) nor domain-alignment methods (DeepCORAL, DANN, MMD-DG; one seed)
   outperform gradient-boosted trees on the same inputs.
5. **False alarms, tested at the sensor.** Over 22,121 normal recordings from
   1,239 sensors in the four regions able to measure it, the hierarchical
   classifier — alone or averaged with the flat one — raised false alarms on 4
   sensors against 14 for LightGBM (paired McNemar test, p = 0.021), at typing
   accuracy that could not be told apart. The false-alarm criterion was adopted
   after LightGBM led on the canonical fold; the evidence is concentrated in one
   region.
6. **Binary detection is not reading the site, as far as can be tested.** With
   each region held out, normal and leak recordings are separated perfectly
   inside each of the five installations holding at least 30 of both. No sensor
   carries both labels, so a difference in how leak and normal sensors were
   recorded cannot be excluded.

## Layout

```
causal-leak/
├── README.md, LICENSE (MIT), REPLICATION.md, Dockerfile, Makefile
├── requirements.txt, requirements-eaai.txt
├── scripts/
│   ├── reproduce_all.sh              full experiment + figure pipeline
│   └── verify.sh                     code / figure / phrase consistency gate
├── src/
│   ├── data.py                       loader (data root via $CAUSAL_LEAK_DATA_DIR)
│   ├── band_selection.py             nested, leakage-free band selection
│   │                                 + per-fold exclusion manifest and checker
│   ├── irm_core.py                   IRM / GroupDRO infrastructure
│   ├── run_matrix.py, run_loro.py    shortcut matrix, leave-one-region-out
│   ├── run_nested_causal_bands.py    nested band selection over 7 regions
│   ├── run_causal_bands.py           SUPERSEDED — documents the retracted protocol
│   ├── run_irm_*.py, run_pga.py,     invariance and deep baselines
│   │   run_augmentation.py
│   ├── run_hierarchical.py,          tree classifiers and ensemble
│   │   run_ensemble.py
│   ├── run_extra_baselines*.py       LightGBM / CatBoost / RF / TabNet
│   ├── run_domain_alignment.py       DeepCORAL / DANN / MMD-DG
│   ├── run_sigungu_ood.py            county-level holdout
│   ├── run_temporal_split.py         temporal holdout
│   ├── run_threshold_sweep_loro.py   calibrated detection thresholds
│   ├── run_operational_folds.py      sensor-level false-alarm comparison
│   ├── run_lrate_confound.py         binary-detection confound checks
│   ├── run_cost_metrics.py, run_compute_cost.py
│   ├── run_multiseed.sh, aggregate_multiseed.py
│   └── make_fig*.py                  figure generators (vector PDF)
└── results/
    ├── FINDINGS_NESTED_BANDS.md      band-selection correction
    ├── FINDINGS_RQ4_RQ6_RERUN.md     re-run of every affected experiment
    ├── FINDINGS_LRATE_CONFOUND.md    binary-detection confound checks
    ├── retracted_fixed_band_65_422/  artefacts computed with the retracted band
    ├── *.json, *.log                 raw fold-level metrics and logs
    └── figs/                         figures
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

# Full pipeline; hard-fails on any non-zero exit.
scripts/reproduce_all.sh

# Multi-seed sweep (~6 h on 2 GPUs):
make multiseed

# Make targets:
make all          # experiments + multiseed + figures + verify
make experiments  # every experiment
make multiseed    # 5-seed sweep + aggregation
make figures      # regenerate every figure
make verify       # gate-check code, figures, and phrase consistency
```

`run_nested_causal_bands.py` must run before any script that reads the
leakage-free feature set; `reproduce_all.sh` orders them accordingly. Nested
selections are memoised in `results/nested_band_cache.json`; delete it to force
recomputation. See `REPLICATION.md` for runtimes, troubleshooting and Docker.

## Citing this work

Please cite the original Sensors paper and this repository:

```bibtex
@article{lee2023leak,
  author  = {Suan Lee and Byeonghak Kim},
  title   = {Machine Learning Model for Leak Detection Using Water Pipeline Vibration Sensor},
  journal = {Sensors},
  volume  = {23}, number = {21}, pages = {8935},
  year    = {2023},
  doi     = {10.3390/s23218935}
}

@misc{lee2026causalleak,
  author = {Suan Lee},
  title  = {causal-leak: audit and out-of-distribution re-evaluation of
            vibration-based water leak detection},
  year   = {2026},
  note   = {Code and results: https://github.com/suanlab/causal-leak}
}
```

## License

Code: MIT (see `LICENSE`).
Vibration data: separate license from AI Hub; not redistributed here.
