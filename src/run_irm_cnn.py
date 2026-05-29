"""Phase 2B (b) — 1D-CNN on raw 513-band spectrum + tabular head.

Architecture:
  raw HZ  [B, 1, 513] -> Conv1d stack -> GlobalPool [B, 128]
  tabular [B, 21]     (month + 20 MAX, already standardised)
  head    concat([cnn_feat, tabular]) -> MLP -> 5 logits

The tabular inputs carry the `MAX*` aggregates and month, which Phase 2C
showed are useful for the `normal` class. The conv backbone learns spectral
patterns over the full 513 bands with no band-level preselection.
"""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from irm_core import (DEVICE, RunConfig, make_splits, set_seed, train_and_eval)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


class SpectrumCNN(nn.Module):
    def __init__(self, tab_dim: int, n_classes: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + tab_dim, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_tab, x_raw):
        # x_raw: [B, 1, 513] or [B, 513] — normalize shape
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(1)
        h = self.conv(x_raw).squeeze(-1)           # [B, 128]
        h = torch.cat([h, x_tab], dim=1)
        return self.head(h)


def main():
    print(f'DEVICE = {DEVICE}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    from irm_core import SEED as _SEED
    set_seed(_SEED)

    splits = make_splits(mode='raw')
    # Tabular features: we still use the scaled `X_train` which contains
    # month + 513 HZ + 20 MAX. But the CNN is fed raw_HZ separately; strip
    # HZ columns from tabular to avoid duplication.
    from data import HZ_COLS  # noqa: WPS433
    hz_set = set(HZ_COLS)
    tab_mask = [c not in hz_set for c in splits.feature_cols]
    import numpy as np  # noqa: WPS433
    tab_mask_arr = np.array(tab_mask)
    splits.X_train = splits.X_train[:, tab_mask_arr]
    splits.X_val = splits.X_val[:, tab_mask_arr]
    splits.X_test = splits.X_test[:, tab_mask_arr]
    splits.feature_cols = [c for c, m in zip(splits.feature_cols, tab_mask)
                           if m]
    print(
        f'tab_feat={splits.X_train.shape[1]}  raw_HZ={splits.raw_HZ_train.shape[1]}',
        flush=True,
    )

    tab_dim = splits.X_train.shape[1]

    def ctor():
        return SpectrumCNN(tab_dim=tab_dim)

    runs = []
    for method in ['erm', 'irm', 'groupdro']:
        print(f'\n=== CNN / {method} ===', flush=True)
        cfg = RunConfig(method=method, epochs=60, batch_size=528,
                        lr=1e-3, weight_decay=1e-4,
                        irm_lambda_max=1e2, irm_warmup_epochs=15,
                        dro_eta=1e-2, early_stop_patience=15,
                        use_raw_HZ=True)
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
    with open(os.path.join(RESULTS_DIR, f'irm_cnn_results{suffix}.json'),
              'w') as f:
        json.dump({'seed': _SEED, 'runs': runs}, f, indent=2)

    print('\n=== CNN summary ===')
    print(f'{"method":10s} {"val_f1":>10s} {"test_f1":>10s} '
          f'{"test_acc":>10s}')
    for r in runs:
        print(f'{r["method"]:10s} {r["val"]["f1_macro"]:10.4f} '
              f'{r["test"]["f1_macro"]:10.4f} {r["test"]["acc"]:10.4f}')


if __name__ == '__main__':
    main()
