# End-to-End Replication Recipe

This document is the replication appendix for the article *Identifier
Leakage and Selection-Defined Classes in a Sensor Benchmark: An Audit of
Water-Leak Classification* (submitted to Data Mining and Knowledge Discovery).
It covers everything needed to reproduce every number the article cites from
a clean machine.

## Quick path

```bash
pip install -r requirements-lock.txt     # exact environment (Python 3.12.3)
make provenance    # hashes and row checks of the AI Hub release (needs the data)
make cwru          # public CWRU bearing data, ~134 MB, into ../external/cwru
make experiments   # every experiment and figure (scripts/reproduce_all.sh)
make numbers       # print every number the article cites, by section
make verify        # code gate (manuscript checks run only where the source is)
```

`make reproduce` runs the five in order. Without the AI Hub data,
`make numbers` and `make verify` still run: they read only the committed
result files. The same gate runs in CI (`.github/workflows/verify.yml`).

Every number in the article is printed by `src/cited_numbers.py` with the
section it supports. The manuscript's own checker, which matches those numbers
against the article's sentences, is kept with the (unreleased) manuscript
source and is available on request.

## Hardware reference profile

The benchmark numbers reported in the paper were measured on:

- CPU: 16-core AMD Ryzen 9 (or equivalent Intel)
- RAM: 64 GB
- GPU: 2 × NVIDIA TITAN Xp (12 GB each), CUDA 12.2 driver 535
- OS: Ubuntu 24.04 LTS

Less GPU memory is fine; deep experiments fit comfortably in 6 GB.
For the tree experiments only, a 16-core CPU machine without any
GPU suffices.

## Software environment

Two equivalent ways:

### Conda / venv

```bash
# Python 3.12 is required (PyTorch 2.5.1 cu121 wheels)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-eaai.txt   # or: pip install -r requirements-lock.txt (exact)
```

### Docker

```bash
docker build -t causal-leak:dmkd .
docker run --rm --gpus all \
  -v "$(pwd)/results":/work/causal-leak/results \
  causal-leak:dmkd bash src/run_multiseed.sh
```

## Dataset

The AI Hub water-leak vibration dataset (dataset ID 138) is distributed by the
National Information Society Agency of Korea through AI Hub; at the time of
writing the portal accepts applications from users in Korea only. Download the
Training and Validation archives and place them, with their extracted class
CSVs, under

    ../kcc-water/Dataset/Training/   (archive + in/out/noise/other/normal-training.csv)
    ../kcc-water/Dataset/Validation/ (archive + ...-validation.csv)

The analyses read two derived files in `../kcc-water/Dataset/`,
`train_with_site_name_4.csv` and `val_with_site_name_4.csv`, which add an
address to each site code. `src/check_provenance.py` fixes their lineage:
SHA-256 of the archives, the extracted CSVs and the derived files; each
extracted CSV byte-identical to an archive member; every derived row matched
to exactly one distributed row on (site, sid, ldate) with all 536 distributed
columns equal; and the four administrative codes equal to the digits of the
site code. Its output, `results/provenance_results.json`, is committed, so the
hashes can be compared against a fresh download. Override the location with
`CAUSAL_LEAK_DATA_DIR`.

The public CWRU bearing data for the planted-shortcut experiment are fetched
by `scripts/fetch_cwru.sh` (40 files; set `CWRU_DIR` to relocate).

## Reproducing every figure and table

A single shell command reproduces the entire results pipeline,
including the multi-seed sweep:

```bash
bash src/run_multiseed.sh
```

This runs:

1. Multi-seed re-runs of `run_irm_mlp.py`, `run_irm_cnn.py`,
   `run_pga.py`, `run_augmentation.py` for seeds {0, 1, 2, 3}.
   Combined with the existing seed-42 results, this populates the
   five-seed numbers reported throughout the paper.
2. Aggregation: `python src/aggregate_multiseed.py` produces
   `results/multiseed_summary.{json,md}`.

For the additional baseline experiments:

```bash
python src/run_sigungu_ood.py        # E2: 시군구-level OOD
python src/run_extra_baselines.py    # E3: LightGBM, CatBoost, TabNet, RF
python src/run_domain_alignment.py   # E4: DeepCORAL, DANN, MMD-DG
python src/run_cost_metrics.py       # E5+E7: detection-rate@1%FAR, threshold sweep
python src/run_compute_cost.py       # E6: train/infer time, RAM, GPU memory
```

Each script writes its results to `results/<name>_results.json`
together with a `.log` file capturing console output.

## Figure regeneration

```bash
python src/make_figures.py        # fig 1, 2, 3 (matrix, LORO, bands)
python src/make_fig5_causal.py    # fig 5 (causal-band retraining)
python src/make_fig6_irm.py       # fig 6 (IRM/GroupDRO comparison)
python src/make_fig7_robust.py    # fig 7 (bootstrap CIs + SHAP)
python src/make_fig8_boost.py     # fig 8 (augmentation + ensemble)
```

All output is in `results/figs/*.pdf` (vector format, no PNG dependencies).

## Expected runtime

| Component | Compute | Wall-clock |
|-----------|---------|-----------:|
| Multi-seed (E1, 4 seeds × 4 scripts) | 2 GPU + CPU | ~6 h |
| 시군구 OOD (E2) | CPU | ~3 h |
| Extra baselines (E3) | CPU + 1 GPU | ~30 min |
| Domain alignment (E4) | 1 GPU | ~30 min |
| Cost metrics (E5+E7) | CPU | ~20 min |
| Compute cost (E6) | CPU + 1 GPU | ~10 min |
| Device trigger rule, identifier audit | CPU | ~2 min |
| Dispatch run (8 regions × outer + 3 inner folds, 3 in parallel) | CPU | ~4 h |
| Dispatch run on all signal columns (`DISPATCH_FEATURES=all`) | CPU | ~4 h |
| Planted shortcuts on CWRU (`run_audit_semisynthetic.py`) | CPU | ~20 min |
| Region-level inference and sufficiency (`run_audit_inference.py`) | CPU | ~30 min |
| Provenance checks | CPU | ~2 min |
| `analyze_dispatch.py` (incl. 2,000-replicate cluster bootstraps) | CPU | ~10 min |
| Networks with held-out-region early stopping (`CAUSAL_VAL_MODE=region`, 5 seeds + alignment) | 2 GPU | ~3 h |
| Figures | CPU | <1 min |

## Troubleshooting

- `python: command not found` from `run_multiseed.sh`: set the
  `PYBIN` environment variable to your interpreter, e.g.
  `PYBIN=/path/to/.venv/bin/python bash src/run_multiseed.sh`.
- CUDA OOM on TITAN Xp: reduce the batch size in `RunConfig` to 264.
- LightGBM hangs under high CPU contention: reduce `n_jobs` to 4
  (already done in `run_extra_baselines.py`).
- Mismatched seed-42 results vs paper: ensure `data.py` was not
  modified; the data partition seed is fixed at 42 in
  `irm_core.py:SPLIT_SEED`.
