"""Phase 2E — IRM λ hyperparameter sweep on MLP + causal bands.

Tests whether IRM's Phase 2B failure was a tuning issue. Sweeps λ_max ∈
{1, 10, 100, 1000, 10000} and warmup ∈ {10, 30} on the MLP/causal setup.
Same LORO region 43 holdout.
"""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from irm_core import (DEVICE, RunConfig, make_splits, set_seed, train_and_eval)
from run_irm_mlp import MLP  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    print(f'DEVICE = {DEVICE}')
    set_seed(42)
    splits = make_splits(mode='causal')
    n_in = splits.X_train.shape[1]
    print(f'X_train={splits.X_train.shape}  envs={len(splits.env_ids)}')

    def ctor():
        return MLP(n_in=n_in)

    configs = []
    for lam in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        for warmup in [10, 30]:
            configs.append((lam, warmup))

    results = []
    for lam, warmup in configs:
        name = f'irm_lam{int(lam)}_wu{warmup}'
        print(f'\n=== {name} ===', flush=True)
        cfg = RunConfig(method='irm', epochs=100, batch_size=528,
                        lr=1e-3, weight_decay=1e-4,
                        irm_lambda_max=lam, irm_warmup_epochs=warmup,
                        early_stop_patience=25)
        out = train_and_eval(ctor, splits, cfg, log_prefix='  ')
        out['name'] = name
        out['lam'] = lam
        out['warmup'] = warmup
        print(
            f'  DONE {name}: test_f1={out["test"]["f1_macro"]:.4f}  '
            f'test_acc={out["test"]["acc"]:.4f}  '
            f'val_f1={out["val"]["f1_macro"]:.4f}  '
            f'({out["time_s"]:.0f}s)',
            flush=True,
        )
        results.append(out)

    with open(os.path.join(RESULTS_DIR, 'irm_sweep_results.json'),
              'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== IRM sweep summary ===')
    print(f'{"name":25s} {"val_f1":>10s} {"test_f1":>10s} {"test_acc":>10s}')
    for r in results:
        print(f'{r["name"]:25s} {r["val"]["f1_macro"]:10.4f} '
              f'{r["test"]["f1_macro"]:10.4f} {r["test"]["acc"]:10.4f}')


if __name__ == '__main__':
    main()
