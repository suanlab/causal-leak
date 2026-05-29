"""Phase 2B (a) — MLP on causal bands with ERM, IRM, and GroupDRO."""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from irm_core import (DEVICE, RunConfig, make_splits, set_seed, train_and_eval)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class MLP(nn.Module):
    def __init__(self, n_in: int, n_classes: int = 5,
                 hidden=(256, 128), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    print(f'DEVICE = {DEVICE}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    from irm_core import SEED as _SEED
    set_seed(_SEED)

    splits = make_splits(mode='causal')
    n_in = splits.X_train.shape[1]
    print(
        f'X_train={splits.X_train.shape}  '
        f'X_val={splits.X_val.shape}  X_test={splits.X_test.shape}  '
        f'n_envs={len(splits.env_ids)}',
        flush=True,
    )
    print(f'envs = {splits.env_ids}', flush=True)

    def ctor():
        return MLP(n_in=n_in)

    runs = []
    for method in ['erm', 'irm', 'groupdro']:
        print(f'\n=== MLP / {method} ===', flush=True)
        cfg = RunConfig(method=method, epochs=120, batch_size=528,
                        lr=1e-3, weight_decay=1e-4,
                        irm_lambda_max=1e2, irm_warmup_epochs=30,
                        dro_eta=1e-2, early_stop_patience=25)
        out = train_and_eval(ctor, splits, cfg, log_prefix='  ')
        print(
            f'  DONE {method}: test_f1={out["test"]["f1_macro"]:.4f}  '
            f'test_acc={out["test"]["acc"]:.4f}  '
            f'per_class={[round(v, 3) for v in out["test"]["f1_per_class"]]}  '
            f'val_f1={out["val"]["f1_macro"]:.4f}  ({out["time_s"]:.0f}s)',
            flush=True,
        )
        runs.append(out)

    suffix = f'_seed{_SEED}' if _SEED != 42 else ''
    with open(os.path.join(RESULTS_DIR, f'irm_mlp_results{suffix}.json'),
              'w') as f:
        json.dump({'seed': _SEED, 'runs': runs}, f, indent=2)

    print('\n=== MLP summary ===')
    print(f'{"method":10s} {"val_f1":>10s} {"test_f1":>10s} '
          f'{"test_acc":>10s}')
    for r in runs:
        print(f'{r["method"]:10s} {r["val"]["f1_macro"]:10.4f} '
              f'{r["test"]["f1_macro"]:10.4f} {r["test"]["acc"]:10.4f}')


if __name__ == '__main__':
    main()
