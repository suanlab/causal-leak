"""Phase 1 / E2 — 시군구 (county-level) Leave-One-Sigungu-Out OOD ablation.

Treats each (region_lvl1, region_lvl2) pair as one environment. With 36
unique sigungu in the AI Hub benchmark, this provides ~3x more
environments than the 12-fold 광역시도 LORO used in the original paper.

For every sigungu with at least 4 classes present (16 folds), we
train XGBoost on the remaining sigungu and evaluate the held-out one
under two feature regimes:

  - signal_all : 513 HZ + 20 MAX + month  (534 features)
  - causal     : 65–422 Hz + 20 MAX + month (57 features)

Outputs results/sigungu_ood_results.json with per-fold accuracy /
macro-F1 for each regime, plus an aggregated summary.

Compute: ~5 min total on CPU (16 folds × 2 regimes × ~10 s each).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, LABEL_NAMES, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42
MIN_CLASSES = 4   # only fold sigungu that have at least 4 classes for typing


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def build_features(df, mode):
    if mode == 'signal_all':
        cols = ['month'] + HZ_COLS + MAX_COLS
    elif mode == 'causal':
        cols = ['month'] + bands_in_range(65, 422) + MAX_COLS
    else:
        raise ValueError(mode)
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int32)
    return X, y, cols


def main():
    t0 = time.time()
    df = load_all()
    df['sg'] = df['region_lvl1'].astype(str) + '_' + df['region_lvl2'].astype(str)
    sg_summary = df.groupby('sg').agg(
        n=('label', 'size'), n_cls=('label', 'nunique')
    ).sort_values('n', ascending=False)

    eligible = sg_summary[sg_summary['n_cls'] >= MIN_CLASSES].index.tolist()
    print(f'Sigungu eligible for fold (>= {MIN_CLASSES} classes): '
          f'{len(eligible)} / {len(sg_summary)}', flush=True)
    print(sg_summary.head(20))

    fold_records = []
    for sg_id in eligible:
        te_mask = (df['sg'] == sg_id).values
        tr_mask = ~te_mask

        # both feature regimes
        regime_metrics = {}
        for regime in ['signal_all', 'causal']:
            X, y, cols = build_features(df, regime)
            clf = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                objective='multi:softprob', num_class=5,
                tree_method='hist', n_jobs=-1, random_state=SEED,
                eval_metric='mlogloss', verbosity=0,
            )
            clf.fit(X[tr_mask], y[tr_mask])
            pred = clf.predict(X[te_mask])
            acc = float(accuracy_score(y[te_mask], pred))
            f1 = float(f1_score(y[te_mask], pred, average='macro',
                                zero_division=0))
            per_cls = [float(f1_score(y[te_mask], pred, labels=[i],
                                      average='macro', zero_division=0))
                       for i in range(5)]
            regime_metrics[regime] = {
                'acc': acc, 'f1_macro': f1,
                'f1_per_class': per_cls,
                'n_features': len(cols),
            }

        delta_f1 = (regime_metrics['causal']['f1_macro'] -
                    regime_metrics['signal_all']['f1_macro'])

        rec = {
            'sigungu': sg_id,
            'n_test': int(te_mask.sum()),
            'n_train': int(tr_mask.sum()),
            'n_classes_in_test': int(sg_summary.loc[sg_id, 'n_cls']),
            'signal_all': regime_metrics['signal_all'],
            'causal': regime_metrics['causal'],
            'delta_f1_macro': delta_f1,
        }
        fold_records.append(rec)
        print(
            f'  {sg_id:>10s}  n_te={int(te_mask.sum()):>5d}  '
            f'#cls={int(sg_summary.loc[sg_id, "n_cls"])}  '
            f'sig_F1={regime_metrics["signal_all"]["f1_macro"]:.3f}  '
            f'cau_F1={regime_metrics["causal"]["f1_macro"]:.3f}  '
            f'Δ={delta_f1:+.3f}',
            flush=True,
        )

    # aggregates
    deltas = np.array([r['delta_f1_macro'] for r in fold_records])
    n_test = np.array([r['n_test'] for r in fold_records])
    sig_f1 = np.array([r['signal_all']['f1_macro'] for r in fold_records])
    cau_f1 = np.array([r['causal']['f1_macro'] for r in fold_records])
    weighted_sig = float((sig_f1 * n_test).sum() / n_test.sum())
    weighted_cau = float((cau_f1 * n_test).sum() / n_test.sum())
    weighted_delta = weighted_cau - weighted_sig
    summary = {
        'n_folds': len(fold_records),
        'mean_signal_all_f1': float(sig_f1.mean()),
        'mean_causal_f1': float(cau_f1.mean()),
        'unweighted_mean_delta': float(deltas.mean()),
        'unweighted_std_delta': float(deltas.std(ddof=1)),
        'weighted_mean_signal_all_f1': weighted_sig,
        'weighted_mean_causal_f1': weighted_cau,
        'weighted_mean_delta': weighted_delta,
        'wins_for_causal': int((deltas > 0).sum()),
        'wins_for_signal_all': int((deltas < 0).sum()),
    }
    print('\n=== E2 sigungu LORO summary ===')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    out_path = os.path.join(RESULTS_DIR, 'sigungu_ood_results.json')
    with open(out_path, 'w') as f:
        json.dump({'folds': fold_records, 'summary': summary,
                   'min_classes': MIN_CLASSES, 'seed': SEED},
                  f, indent=2)
    print(f'\nSaved {out_path}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
