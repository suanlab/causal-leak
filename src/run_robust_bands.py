"""Phase A (Strengthening 1) — Robust causal frequency-band identification.

Combines three orthogonal evidence sources into one consensus diagnostic:

1. Bootstrap permutation importance (B replicates) → 95% CI on each band group.
2. Multi-region permutation (regions 43, 45, 46, 47 — the 4 folds with at
   least 4 classes and > 5k rows) → consensus bands that win across regions.
3. SHAP values on an XGBoost model → second-source attribution independent
   of permutation.

Outputs:
  results/robust_bands_results.json
  results/robust_bands_summary.md
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42

CANDIDATE_REGIONS = [43, 45, 46, 47]
BOOTSTRAP_B = 100

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective='multi:softprob', num_class=5,
    tree_method='hist', n_jobs=-1, random_state=SEED,
    eval_metric='mlogloss', verbosity=0,
)

BAND_GROUPS = [
    ('10-121', 10, 121),
    ('121-422', 121, 422),
    ('422-1470', 422, 1470),
    ('1470-5120', 1470, 5121),
]


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def fit(X, y, tr, xgb_params=None):
    clf = XGBClassifier(**(xgb_params or XGB_PARAMS))
    clf.fit(X[tr], y[tr])
    return clf


def macro_f1(clf, X, y, idx):
    pred = clf.predict(X[idx])
    return f1_score(y[idx], pred, average='macro', zero_division=0)


def group_to_col_idx(group_name, all_cols):
    _, lo, hi = next(g for g in BAND_GROUPS if g[0] == group_name)
    bands = set(bands_in_range(lo, hi))
    return np.array([i for i, c in enumerate(all_cols) if c in bands])


def bootstrap_permutation_importance(clf, X, y, te, col_idx_map, B=BOOTSTRAP_B,
                                     rng_base_seed=SEED):
    """Bootstrap permutation importance with B resampled permutations.
    Returns dict: group_name -> list of Δf1 values (length B).
    """
    base = macro_f1(clf, X, y, te)
    out = {g[0]: [] for g in BAND_GROUPS}
    X_te = X[te].copy()
    for b in range(B):
        rng = np.random.RandomState(rng_base_seed + b)
        order = rng.permutation(len(te))
        for gname, _, _ in BAND_GROUPS:
            idxs = col_idx_map[gname]
            if len(idxs) == 0:
                out[gname].append(0.0)
                continue
            Xp = X_te.copy()
            Xp[:, idxs] = X_te[order][:, idxs]
            pred = clf.predict(Xp)
            perm_f1 = f1_score(y[te], pred, average='macro',
                               zero_division=0)
            out[gname].append(float(base - perm_f1))
    return base, out


def main():
    t_all = time.time()
    df = load_all()
    cols = ['month'] + HZ_COLS + MAX_COLS
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int32)

    col_idx_map = {g[0]: group_to_col_idx(g[0], cols) for g in BAND_GROUPS}
    results = {'per_region': {}, 'shap': None}

    # ---- 1. Bootstrap permutation × multi-region ----
    for region in CANDIDATE_REGIONS:
        mask = df['region_lvl1'].values == region
        tr = np.where(~mask)[0]
        te = np.where(mask)[0]
        cls = np.unique(y[te])
        if len(cls) < 4:
            print(f'  skip region {region}: only {len(cls)} classes',
                  flush=True)
            continue
        print(f'=== region {region}  train={len(tr)} test={len(te)} '
              f'classes={len(cls)} ===', flush=True)
        t0 = time.time()
        clf = fit(X, y, tr)
        print(f'  trained {time.time() - t0:.0f}s  macro_f1={macro_f1(clf, X, y, te):.4f}',
              flush=True)

        base_f1, perm_samples = bootstrap_permutation_importance(
            clf, X, y, te, col_idx_map, B=BOOTSTRAP_B)

        region_stats = {'base_f1': float(base_f1),
                        'n_test': int(len(te)),
                        'groups': {}}
        for gname, _, _ in BAND_GROUPS:
            arr = np.array(perm_samples[gname])
            region_stats['groups'][gname] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'ci95_low': float(np.percentile(arr, 2.5)),
                'ci95_high': float(np.percentile(arr, 97.5)),
                'samples': [float(v) for v in arr],
            }
        results['per_region'][str(region)] = region_stats
        for gname, _, _ in BAND_GROUPS:
            s = region_stats['groups'][gname]
            print(f'    {gname:>10s}  Δf1 = {s["mean"]:+.4f} '
                  f'(95% CI [{s["ci95_low"]:+.4f}, {s["ci95_high"]:+.4f}])',
                  flush=True)

    # ---- 2. SHAP analysis on canonical (region 43) model ----
    if shap is not None:
        region = 43
        mask = df['region_lvl1'].values == region
        tr = np.where(~mask)[0]
        te = np.where(mask)[0]
        clf = fit(X, y, tr)
        # For speed: sample 500 background + 500 test rows for SHAP
        rng = np.random.RandomState(SEED)
        bg_idx = rng.choice(tr, size=min(500, len(tr)), replace=False)
        te_sample = rng.choice(te, size=min(500, len(te)), replace=False)
        explainer = shap.TreeExplainer(clf)
        t0 = time.time()
        sv = explainer.shap_values(X[te_sample])
        # sv shape: (n_samples, n_features, n_classes) in recent xgboost / shap
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        print(f'  SHAP computed in {time.time() - t0:.0f}s, shape={sv.shape}',
              flush=True)

        # Aggregate |SHAP| over samples and classes → per-feature importance
        mean_abs = np.mean(np.abs(sv), axis=(0, -1))  # (n_features,)
        shap_per_group = {}
        total_hz_abs = 0.0
        for gname, _, _ in BAND_GROUPS:
            idxs = col_idx_map[gname]
            s = float(mean_abs[idxs].sum())
            shap_per_group[gname] = s
            total_hz_abs += s
        shap_share = {g: shap_per_group[g] / total_hz_abs
                      for g in shap_per_group}
        results['shap'] = {'region': region,
                           'n_samples': int(len(te_sample)),
                           'per_group_mean_abs': shap_per_group,
                           'per_group_share': shap_share}
        print('  SHAP band-group shares (normalized over HZ):')
        for g, v in shap_share.items():
            print(f'    {g:>10s}  {v * 100:.1f}%')
    else:
        print('  SHAP not available, skipping.', flush=True)

    # ---- 3. Consensus: which bands win across all regions? ----
    consensus = {}
    for gname, _, _ in BAND_GROUPS:
        per_region_means = [
            results['per_region'][r]['groups'][gname]['mean']
            for r in results['per_region']
        ]
        arr = np.array(per_region_means)
        consensus[gname] = {
            'regions': list(results['per_region'].keys()),
            'deltas': [float(v) for v in arr],
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'wins': int((arr > 0).sum()),
            'losses': int((arr < 0).sum()),
        }
    results['consensus_across_regions'] = consensus

    print('\n=== CONSENSUS ACROSS REGIONS ===')
    for gname, _, _ in BAND_GROUPS:
        c = consensus[gname]
        print(f'  {gname:>10s}  mean Δf1 = {c["mean"]:+.4f} ± {c["std"]:.4f} '
              f'(wins={c["wins"]} / losses={c["losses"]})')

    out_path = os.path.join(RESULTS_DIR, 'robust_bands_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')
    print(f'Total: {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
