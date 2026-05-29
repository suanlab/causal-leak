"""Phase 2A — Leave-One-Region-Out (LORO) evaluation.

For each of the 12 광역시도 regions, hold out all rows from that region as a
test set, train on the remaining 11 regions, and record per-region metrics.
Covers both `full` and `signal` feature sets.

Rationale: Phase 1's GroupKFold-by-site had high variance (std = 0.127) because
some folds held only 7 sites. LORO uses admin-level groups (12 regions) for
larger, more meaningful held-out sets and a tighter summary statistic.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all, get_xy  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=5,
    tree_method='hist',
    n_jobs=-1,
    random_state=SEED,
    eval_metric='mlogloss',
    verbosity=0,
)


def fit_eval(X, y, tr_idx, te_idx):
    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X[tr_idx], y[tr_idx])
    pred = clf.predict(X[te_idx])
    y_te = y[te_idx]
    return {
        'acc': float(accuracy_score(y_te, pred)),
        'f1_macro': float(
            f1_score(y_te, pred, average='macro', zero_division=0)),
        'f1_per_class': [
            float(f1_score(y_te, pred, labels=[i], average='macro',
                           zero_division=0))
            for i in range(5)
        ],
        'n_train': int(len(tr_idx)),
        'n_test': int(len(te_idx)),
        'n_classes_in_test': int(len(np.unique(y_te))),
        'cm': confusion_matrix(y_te, pred, labels=list(range(5))).tolist(),
    }


def main():
    t_all = time.time()
    df = load_all()
    regions = sorted(df['region_lvl1'].unique().tolist())
    print(f'Regions (n={len(regions)}): {regions}', flush=True)
    print('Region sizes:', flush=True)
    for r in regions:
        sub = df[df.region_lvl1 == r]
        cls = sub.leaktype.value_counts().to_dict()
        print(f'  {r}: n={len(sub):5d}  sites={sub.site.nunique():2d}  '
              f'classes={cls}', flush=True)

    all_results = []
    for feat_mode in ['full', 'signal']:
        X, y, _ = get_xy(df, feat_mode)
        print(f'\n=== {feat_mode} | X={X.shape} ===', flush=True)
        per_region = []
        for region in regions:
            mask_te = (df['region_lvl1'].values == region)
            mask_tr = ~mask_te
            tr = np.where(mask_tr)[0]
            te = np.where(mask_te)[0]
            t0 = time.time()
            m = fit_eval(X, y, tr, te)
            m['region'] = int(region)
            m['time_s'] = float(time.time() - t0)
            per_region.append(m)
            print(
                f'  region {region:3d}: n_te={m["n_test"]:5d} '
                f'cls_in_te={m["n_classes_in_test"]}  '
                f'acc={m["acc"]:.4f} f1={m["f1_macro"]:.4f} '
                f'({m["time_s"]:.0f}s)',
                flush=True,
            )
        accs = np.array([r['acc'] for r in per_region])
        f1s = np.array([r['f1_macro'] for r in per_region])
        weights = np.array([r['n_test'] for r in per_region], dtype=float)
        summary = {
            'feat_mode': feat_mode,
            'per_region': per_region,
            'macro_acc': float(accs.mean()),
            'macro_acc_std': float(accs.std()),
            'macro_f1': float(f1s.mean()),
            'macro_f1_std': float(f1s.std()),
            'weighted_acc': float(np.average(accs, weights=weights)),
            'weighted_f1': float(np.average(f1s, weights=weights)),
        }
        all_results.append(summary)
        print(f'  === {feat_mode} macro_acc={summary["macro_acc"]:.4f}'
              f'±{summary["macro_acc_std"]:.4f}  '
              f'weighted_acc={summary["weighted_acc"]:.4f}  '
              f'macro_f1={summary["macro_f1"]:.4f}', flush=True)

    with open(os.path.join(RESULTS_DIR, 'loro_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\n=== LORO HEADLINE ===', flush=True)
    print(f'{"feat":10s}{"macro_acc":>20s}{"weighted_acc":>18s}'
          f'{"macro_f1":>18s}', flush=True)
    for r in all_results:
        print(f'{r["feat_mode"]:10s}'
              f'{r["macro_acc"]:.4f}±{r["macro_acc_std"]:.4f}        '
              f'{r["weighted_acc"]:.4f}            '
              f'{r["macro_f1"]:.4f}', flush=True)

    print(f'\nTotal elapsed: {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
