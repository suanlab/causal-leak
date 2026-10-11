"""SUPERSEDED — retained only to document the retracted protocol.

This script selects nothing: it evaluates a fixed 65-422 Hz subset that was
derived from permutation importance and SHAP measured *on region 43*, and then
reports that subset's score *on region 43*. Selection and evaluation share a
fold, so the reported advantage is optimistically biased (AUDIT.md Blocker 1).

Use src/run_nested_causal_bands.py instead; it selects bands inside the
training regions and emits a per-fold manifest proving the evaluation region
was excluded. Findings: results/FINDINGS_NESTED_BANDS.md.

Original description follows.

Phase 2C follow-up — Causal band subset experiment.

Hypothesis: restricting input to the 65-422 Hz range (the bands whose
permutation actually hurts OOD performance) should match or beat the
full 513-band signal model on the LORO region-43 holdout.

Trains three XGBoost models on the same LORO split:
  - signal_all      : all 513 HZ bands + 20 MAX + month (baseline)
  - causal_lo       : 65-422 Hz only (~36 bands) + 20 MAX + month
  - anti_causal_hi  : 788-5120 Hz only (~364 bands) + 20 MAX + month

If the hypothesis holds, causal_lo >= signal_all > anti_causal_hi on
the OOD test set.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

HOLDOUT_REGION = 43
SEED = 42
XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective='multi:softprob', num_class=5,
    tree_method='hist', n_jobs=-1, random_state=SEED,
    eval_metric='mlogloss', verbosity=0,
)


def bands_in_range(lo, hi):
    """HZ columns whose frequency f satisfies lo <= f < hi."""
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def build(df, cols):
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int32)
    return X, y


def fit_eval(X, y, tr, te):
    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return {
        'acc': float(accuracy_score(y[te], pred)),
        'f1_macro': float(f1_score(y[te], pred, average='macro',
                                   zero_division=0)),
        'f1_per_class': [
            float(f1_score(y[te], pred, labels=[i], average='macro',
                           zero_division=0))
            for i in range(5)
        ],
        'cm': confusion_matrix(y[te], pred, labels=list(range(5))).tolist(),
    }


def main():
    t0 = time.time()
    df = load_all()
    mask_te = df['region_lvl1'].values == HOLDOUT_REGION
    tr = np.where(~mask_te)[0]
    te = np.where(mask_te)[0]
    print(f'LORO region {HOLDOUT_REGION}: train={len(tr)}, test={len(te)}',
          flush=True)

    # Full signal baseline
    FULL_SIG = ['month'] + HZ_COLS + MAX_COLS
    CAUSAL = ['month'] + bands_in_range(65, 422) + MAX_COLS
    ANTI_CAUSAL = ['month'] + bands_in_range(788, 5120 + 1) + MAX_COLS
    LOW_ONLY = ['month'] + bands_in_range(0, 65) + MAX_COLS
    MID_CONFOUNDER = ['month'] + bands_in_range(422, 1470) + MAX_COLS

    configs = [
        ('signal_all', FULL_SIG),
        ('causal_65_422Hz', CAUSAL),
        ('anti_causal_788_5120Hz', ANTI_CAUSAL),
        ('low_only_0_65Hz', LOW_ONLY),
        ('mid_confounder_422_1470Hz', MID_CONFOUNDER),
    ]

    out = []
    for name, cols in configs:
        Xc, y = build(df, cols)
        t = time.time()
        m = fit_eval(Xc, y, tr, te)
        m['config'] = name
        m['n_features'] = len(cols)
        m['time_s'] = float(time.time() - t)
        print(
            f'  {name:>28s}: n_feat={len(cols):4d}  '
            f'acc={m["acc"]:.4f}  f1={m["f1_macro"]:.4f}  '
            f'per_class={[round(v, 3) for v in m["f1_per_class"]]}  '
            f'({m["time_s"]:.0f}s)',
            flush=True,
        )
        out.append(m)

    with open(os.path.join(RESULTS_DIR, 'causal_bands_results.json'),
              'w') as f:
        json.dump(out, f, indent=2)

    print(f'\nTotal: {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
