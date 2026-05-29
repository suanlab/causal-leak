# End-to-End Replication Recipe

This document is the replication appendix referenced from the EAAI
manuscript (Section "Hardware and Reproducibility"). It walks
through everything required to reproduce every numeric claim in the
paper from a clean machine.

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
pip install -r requirements-eaai.txt
```

### Docker

```bash
docker build -t causal-leak:eaai .
docker run --rm --gpus all \
  -v "$(pwd)/results":/work/causal-leak/results \
  causal-leak:eaai bash src/run_multiseed.sh
```

## Dataset

The AI Hub water-pipe vibration dataset is openly distributed by NIA
Korea AI Hub (dataset ID 138, accessed 5 October 2023). After
download, place the CSVs under

    ../kcc-water/Dataset/
        train_with_site_name_4.csv
        val_with_site_name_4.csv

relative to the `causal-leak/` folder. The loader in `src/data.py`
expects exactly this layout.

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

For the additional EAAI experiments:

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
| Figures | CPU | <1 min |

## Troubleshooting

- `python: command not found` from `run_multiseed.sh`: set the
  `PYBIN` environment variable to your interpreter, e.g.
  `PYBIN=/path/to/.venv/bin/python bash src/run_multiseed.sh`.
- CUDA OOM on TITAN Xp: reduce the batch size in `RunConfig` to 264.
- LightGBM hangs under high CPU contention: reduce `n_jobs` to 4
  (already done in the EAAI version of `run_extra_baselines.py`).
- Mismatched seed-42 results vs paper: ensure `data.py` was not
  modified; the data partition seed is fixed at 42 in
  `irm_core.py:SPLIT_SEED`.
