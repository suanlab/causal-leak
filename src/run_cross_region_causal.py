"""Phase 2E — Cross-region robustness of the causal-band finding.

Question: does restricting the spectrum help on every LORO fold, or only on
the 충북 (region 43) holdout?

For each region we hold it out and train two XGBoost models: one with all 513
HZ bands (+ MAX + month), one with the bands selected for that fold by nested
selection (band_selection.nested_bands_cached), which never sees the held-out
region. The previous version of this script compared against a fixed
65-422 Hz subset that had been chosen from attribution measured on region 43,
so its per-region deltas were only meaningful on that one fold; see
results/FINDINGS_NESTED_BANDS.md.

Skipping regions with only one class in the test set (no macro-F1
defined there in a meaningful way).
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import nested_bands_cached  # noqa: E402
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

SEED = 42
XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective='multi:softprob', num_class=5,
    tree_method='hist', n_jobs=-1, random_state=SEED,
    eval_metric='mlogloss', verbosity=0,
)


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


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
    t_all = time.time()
    df = load_all()
    regions = sorted(df['region_lvl1'].unique().tolist())
    print(f'Regions: {regions}', flush=True)

    signal_all_cols = ['month'] + HZ_COLS + MAX_COLS
    X_all = df[signal_all_cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int32)

    out = []
    for region in regions:
        mask_te = (df['region_lvl1'].values == region)
        tr = np.where(~mask_te)[0]
        te = np.where(mask_te)[0]
        n_classes_te = int(len(np.unique(y[te])))
        if n_classes_te < 2:
            print(f'  skip region {region}: only {n_classes_te} class(es) '
                  f'in test set', flush=True)
            continue

        nested_cols, manifest = nested_bands_cached(
            df, X_all, y, signal_all_cols, region, kind='region')

        t0 = time.time()
        m_all = fit_eval(X_all, y, tr, te)
        t_all_fit = time.time() - t0
        X_causal = df[nested_cols].to_numpy(dtype=np.float32, copy=False)
        t0 = time.time()
        m_caus = fit_eval(X_causal, y, tr, te)
        t_caus_fit = time.time() - t0

        row = {
            'region': int(region),
            'n_test': int(len(te)),
            'n_classes_in_test': n_classes_te,
            'signal_all': m_all,
            'nested_causal': m_caus,
            'selected_groups': manifest['selected_groups'],
            'n_nested_features': len(nested_cols),
            'delta_acc': float(m_caus['acc'] - m_all['acc']),
            'delta_f1': float(m_caus['f1_macro'] - m_all['f1_macro']),
            'time_s': float(t_all_fit + t_caus_fit),
        }
        out.append(row)
        print(
            f'  region {region:3d}  n_te={len(te):5d}  cls_te={n_classes_te}'
            f'  signal_all: acc={m_all["acc"]:.3f} f1={m_all["f1_macro"]:.3f}'
            f'  |  causal: acc={m_caus["acc"]:.3f} f1={m_caus["f1_macro"]:.3f}'
            f'  |  Δf1={row["delta_f1"]:+.4f}  Δacc={row["delta_acc"]:+.4f}'
            f'  ({row["time_s"]:.0f}s)',
            flush=True,
        )

    # aggregate
    d_f1 = np.array([r['delta_f1'] for r in out])
    d_acc = np.array([r['delta_acc'] for r in out])
    weights = np.array([r['n_test'] for r in out], dtype=float)
    agg = {
        'macro_delta_f1': float(d_f1.mean()),
        'macro_delta_f1_std': float(d_f1.std()),
        'weighted_delta_f1': float(np.average(d_f1, weights=weights)),
        'macro_delta_acc': float(d_acc.mean()),
        'weighted_delta_acc': float(np.average(d_acc, weights=weights)),
        'regions_where_causal_wins_f1': int(int((d_f1 > 0).sum())),
        'regions_where_causal_ties_f1': int(int((d_f1 == 0).sum())),
        'regions_where_causal_loses_f1': int(int((d_f1 < 0).sum())),
        'n_regions': int(len(out)),
    }
    print('\n=== AGGREGATE across regions ===', flush=True)
    for k, v in agg.items():
        print(f'  {k}: {v}', flush=True)

    with open(os.path.join(RESULTS_DIR, 'cross_region_results.json'),
              'w') as f:
        json.dump({'per_region': out, 'aggregate': agg}, f, indent=2)
    print(f'\nTotal: {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
