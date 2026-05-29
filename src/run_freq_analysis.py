"""Phase 2C — Causal frequency band analysis.

Identifies which vibration frequency bands carry generalizable leak signal by
measuring permutation importance under a signal-only LORO setup.

Setup:
  - Feature set: `signal` (513 HZ + 20 MAX + month)
  - Split: LORO holding out region 43 (충북), the only region containing all
           5 classes with substantial counts for every class.
  - Analysis:
      1. Log-spaced band groups (10 groups over 0-5120 Hz) — grouped
         permutation to measure each range's contribution.
      2. XGBoost gain importance aggregated per band group.
      3. Top-20 individual bands by gain.

Output: results/freq_analysis.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, LABEL_NAMES, get_xy, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

HOLDOUT_REGION = 43          # 충북: all 5 classes well represented
N_BAND_GROUPS = 10           # log-spaced
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


def make_band_groups(n_groups=N_BAND_GROUPS):
    """Log-spaced frequency bin groups over 10-5120 Hz (with 0HZ pinned to
    the first group)."""
    edges = np.geomspace(10, 5120, n_groups + 1)
    all_bands = [f'{i}HZ' for i in range(10, 5130, 10)]
    groups = {}
    for i in range(n_groups):
        lo, hi = edges[i], edges[i + 1]
        bands = [b for b in all_bands
                 if lo <= int(b.replace('HZ', '')) < hi]
        name = f'{int(round(lo))}-{int(round(hi))}Hz'
        groups[name] = bands
    # Pin 0HZ to the first group
    first = next(iter(groups))
    groups[first] = ['0HZ'] + groups[first]
    return groups


def main():
    t_all = time.time()
    df = load_all()
    X, y, cols = get_xy(df, 'signal')
    col_idx = {c: i for i, c in enumerate(cols)}

    mask_te = (df['region_lvl1'].values == HOLDOUT_REGION)
    tr = np.where(~mask_te)[0]
    te = np.where(mask_te)[0]
    te_labels, te_counts = np.unique(y[te], return_counts=True)
    print(f'Holdout region={HOLDOUT_REGION}')
    print(f'Train rows={len(tr)}  Test rows={len(te)}')
    print(f'Test label dist: {dict(zip(te_labels.tolist(), te_counts.tolist()))}',
          flush=True)

    t0 = time.time()
    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X[tr], y[tr])
    train_s = time.time() - t0
    print(f'Model trained in {train_s:.0f}s', flush=True)

    base_pred = clf.predict(X[te])
    base_acc = float(accuracy_score(y[te], base_pred))
    base_f1_macro = float(f1_score(y[te], base_pred, average='macro',
                                   zero_division=0))
    base_f1_pc = f1_score(y[te], base_pred, average=None,
                          labels=list(range(5)), zero_division=0)
    base_f1_pc = [float(v) for v in base_f1_pc]
    print(f'Baseline: acc={base_acc:.4f} f1_macro={base_f1_macro:.4f}')
    print(f'  per-class F1: ' + ', '.join(
        f'{n}={v:.3f}' for n, v in zip(LABEL_NAMES, base_f1_pc)),
          flush=True)

    # Grouped permutation importance
    rng = np.random.RandomState(SEED)
    groups = make_band_groups(N_BAND_GROUPS)
    band_perm = []
    X_te = X[te].copy()
    for gname, bands in groups.items():
        idxs = np.array([col_idx[b] for b in bands if b in col_idx])
        X_perm = X_te.copy()
        perm_order = rng.permutation(len(te))
        X_perm[:, idxs] = X_te[perm_order][:, idxs]
        perm_pred = clf.predict(X_perm)
        p_acc = accuracy_score(y[te], perm_pred)
        p_f1 = f1_score(y[te], perm_pred, average='macro', zero_division=0)
        p_f1_pc = f1_score(y[te], perm_pred, average=None,
                           labels=list(range(5)), zero_division=0)
        band_perm.append({
            'group': gname,
            'n_bands': int(len(idxs)),
            'f1_macro_drop': float(base_f1_macro - p_f1),
            'acc_drop': float(base_acc - p_acc),
            'f1_drop_per_class': [
                float(base_f1_pc[i] - p_f1_pc[i]) for i in range(5)
            ],
        })
        print(f'  perm {gname:>12s} (n={len(idxs):3d}): '
              f'f1_drop={base_f1_macro - p_f1:+.4f} '
              f'acc_drop={base_acc - p_acc:+.4f}',
              flush=True)

    # Gain importance per feature
    booster = clf.get_booster()
    imp_raw = booster.get_score(importance_type='gain')
    gain_per_col = {}
    for fkey, val in imp_raw.items():
        idx = int(fkey[1:])
        gain_per_col[cols[idx]] = float(val)

    gain_per_group = {}
    for gname, bands in groups.items():
        gain_per_group[gname] = float(
            sum(gain_per_col.get(b, 0.0) for b in bands))

    top_bands = sorted(gain_per_col.items(), key=lambda kv: -kv[1])[:30]

    out = {
        'holdout_region': HOLDOUT_REGION,
        'train_time_s': train_s,
        'baseline': {
            'acc': base_acc,
            'f1_macro': base_f1_macro,
            'f1_per_class': base_f1_pc,
        },
        'band_groups': {k: v for k, v in groups.items()},
        'band_perm_importance': band_perm,
        'band_gain': gain_per_group,
        'top30_features_by_gain': [
            {'feature': k, 'gain': v} for k, v in top_bands
        ],
    }

    with open(os.path.join(RESULTS_DIR, 'freq_analysis.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved results/freq_analysis.json')
    print(f'Total elapsed: {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
