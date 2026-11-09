# causal-leak

**An audit and out-of-distribution re-evaluation of Lee & Kim (2023)**
*Machine Learning Model for Leak Detection Using Water Pipeline Vibration
Sensor.* Sensors **23**(21), 8935. [DOI: 10.3390/s23218935](https://doi.org/10.3390/s23218935)

This repository is the follow-up to our 2023 Sensors paper. It measures how
much of the reported 99.79 % accuracy on the Korean AI Hub water-leak benchmark
comes from recognising which sensor produced a recording or re-learning the
sensor's own trigger. It evaluates leak typing with whole regions, counties or
later dates held out, and scores models under a crew-dispatch rule at
thresholds calibrated on training data.

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

| Evaluation | Accuracy | Macro-F1 (5) | Leak-class macro-F1 |
|---|---:|---:|---:|
| Sensor-identifier lookup + one `lrate` rule (no spectrum used) | 0.9977 | – | – |
| Row-level cross-validation, with identifier columns | 0.945 | 0.933 | 0.916 |
| Group-CV by installation, with identifier columns | 0.553 | 0.419 | 0.274 |
| Leave-one-region-out, signal only (weighted) | 0.568 | 0.301 | 0.298 |

*Leak-class macro-F1* is the mean F1 over the leak classes (in/out/noise/other)
present in the evaluation set. The normal class is excluded from the average
because the device's own trigger count separates it (finding 3). Its
reference is 1/K for K leak classes present, which is what a guess matched to
the test set's class shares scores. Under leave-one-region-out, leak-class F1
weights regions by leak recordings over the 11 regions with at least two leak
classes; its reference is 0.264.

Typing on the canonical held-out region (Chungbuk, region 43; reference 0.25),
leakage-free features:

| Model | Leak-class macro-F1 |
|---|---:|
| LightGBM | 0.311 |
| Tree-only soft average (flat + hierarchical) | 0.298 |
| CNN, ERM, early-stopped on held-out regions (5 seeds) | 0.303 ± 0.023 |
| CNN, IRM (5 seeds) | 0.291 ± 0.026 |
| CatBoost (5 seeds) | 0.281 ± 0.005 |
| CNN, ERM (5 seeds) | 0.275 ± 0.017 |
| DeepCORAL (1 seed) | 0.268 |
| MLP, GroupDRO (5 seeds) | 0.264 ± 0.014 |

## Findings

1. **Identifier shortcut.** Every training sensor carries a single label, and
   98 % of validation sensors also appear in training. A lookup on the sensor
   identifier covers 99.5 % of validation recordings at 100 % accuracy. Of the
   39 uncovered rows, 21 have `lrate` below 90 and are therefore normal.
   Marking them normal and counting the other 18 as errors gives 99.77 %
   validation accuracy without reading the vibration spectrum. That is within
   0.02 points of the published 99.79 %, or 99.85 % if the 18 rows get the
   most common training label (`src/run_identifier_audit.py`).
2. **Collapse under realistic evaluation.** Accuracy falls from 0.945 under
   row-level cross-validation to 0.553 with installations held out. With
   regions held out it is 0.562 with identifiers and 0.568 without.
   Identifier columns are worth 5.3 points at row level and nothing once
   sensors are unseen.
3. **Device-trigger shortcut.** The AI Hub page defines `lrate` as the share
   of 10 early-morning checks in which the device's leak response exceeded
   its threshold. Every in/out/noise/other recording has `lrate = 90`, and
   every normal one 30–70. The 20 `MAX` columns alternate between peak
   frequency and peak amplitude for each check. The device threshold is
   identified exactly (`lrate` = 10 x the number of peak amplitudes >= 125 on
   every normal recording), and the rule "non-normal if at least 9 of the 10
   peak amplitudes are >= 125" separates normal from non-normal recordings
   with 6 errors in 70,384; no recording has a count of 8. Every learned normal-vs-rest gate re-learns this rule,
   which explains its AUC of 1.000 everywhere. Binary detection on this
   benchmark therefore cannot be evaluated as leak detection
   (`src/run_device_rule.py`, `results/FINDINGS_LRATE_CONFOUND.md`).
4. **Leak typing near chance under holdout.** Typing among
   in/out/noise/other reaches leak-class macro-F1 0.298 with regions held out
   (reference 0.264). It is 0.234 over 16 county holdouts (reference 0.260),
   and 0.26–0.28 on recordings after a date cutoff (reference 0.25). On the six
   four-class regions it is 0.045 above the reference. That excess is above
   zero when sensors are resampled but not when sites are
   (`src/summarize_leak_f1.py`, `src/analyze_dispatch.py`).
5. **No model family changes this.** Gradient-boosted trees, invariance
   regularisers (IRM, GroupDRO), domain alignment (DeepCORAL, DANN, MMD-DG) and
   TabNet all score 0.24–0.31 on the canonical region against a reference of
   0.25. Re-running every network with early stopping on whole held-out
   training regions, instead of a random 10 % of recordings, moves CNN-ERM
   from 0.275 to 0.303 and the others by at most 0.04
   (`CAUSAL_VAL_MODE=region`, results tagged `_regionval`).
6. **No stable "causal band".** With band selection nested inside the training
   regions, the 121–422 Hz band is selected in 2 of 7 regions. Selecting along a
   temporal axis instead retains it, so a band's generalisation claim depends on
   the axis named.
7. **Crew dispatch.** A crew is sent when a recording is typed in or out.
   Thresholds are calibrated on training regions only, through region-grouped
   inner folds. Once the device has triggered, every candidate separates in/out
   leaks from noise/other with AUC 0.594–0.598 pooled over eight regions.
   Pooled over eight regions, calibrated miss rates transfer, but catching
   90 % of leaks sends crews to 78.5–81.0 % of noise/other recordings. Region
   by region, the 10 % target misses 2.4–50.2 % of leaks. Averaging scores per
   sensor does not help (AUC 0.59). Models differ
   by a few points, and the direction changes with the target, so no model is
   recommended. An earlier version of this repository recommended a tree
   ensemble over LightGBM on argmax false alarms (4 against 14 sensors). That
   comparison is withdrawn: under the dispatch rule it is a trade-off, and the
   normal-class false alarms it counted are all removed by the device's own
   trigger (`src/analyze_dispatch.py`, `results/FINDINGS_DISPATCH.md`).

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
│   ├── run_operational_folds.py      false alarms at argmax (superseded by dispatch)
│   ├── run_lrate_confound.py         binary-detection confound checks
│   ├── run_identifier_audit.py       sensor-roster lookup, no model
│   ├── run_device_rule.py            training-free device trigger-count rule
│   ├── run_dispatch_operating_points.py  per-recording probabilities, outer
│   │                                 fold + region-grouped inner calibration
│   ├── analyze_dispatch.py           dispatch rule at calibrated thresholds
│   ├── summarize_leak_f1.py          leak-class macro-F1 for every result
│   ├── sensor_stats.py               sensor-level McNemar, cluster bootstrap,
│   │                                 installation sign test
│   ├── audit_tests.py                audit statistics: A1 lift test, T1 threshold
│   │                                 identity, T3 cluster-bootstrap depletion
│   ├── run_sensor_holdout.py         new-sensor rung; sensor vs site grouping
│   ├── run_audit_inference.py        region-level inference, T2 sufficiency,
│   │                                 variance shares, site-level paired tests
│   ├── run_audit_semisynthetic.py    planted shortcuts on CWRU bearing data
│   │                                 (size and power of the audit tests)
│   ├── run_cost_metrics.py, run_compute_cost.py
│   ├── run_multiseed.sh, aggregate_multiseed.py
│   └── make_fig*.py                  figure generators (vector PDF)
└── results/
    ├── FINDINGS_NESTED_BANDS.md      band-selection correction
    ├── FINDINGS_RQ4_RQ6_RERUN.md     re-run of every affected experiment
    ├── FINDINGS_LRATE_CONFOUND.md    device trigger count and binary detection
    ├── dispatch_op/                  per-region probabilities for the dispatch run
    ├── retracted_fixed_band_65_422/  artefacts computed with the retracted band
    ├── superseded_grid10/            device rule from the step-10 threshold grid
    ├── audit_external_*.json         SUPERSEDED, not reported: MIMII grouped on a
    │                                 field whose purity equals the class prior,
    │                                 SECOM on a constructed identifier
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

# Public CWRU bearing data (40 files, ~134 MB) for the planted-shortcut
# experiment (src/run_audit_semisynthetic.py); default destination
# ../external/cwru, override with CWRU_DIR.
scripts/fetch_cwru.sh

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
