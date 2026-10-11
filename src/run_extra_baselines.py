"""Phase 1 / E3 — Additional tabular baselines on the canonical
LORO Chungbuk fold (region_lvl1 == 43).

Models compared (all on 57 causal features = 65–422 Hz + 20 MAX + month):
  - RandomForest (sklearn)
  - LightGBM
  - CatBoost
  - TabNet (pytorch-tabnet)

Comparison anchors (already known from prior phases):
  - XGBoost causal: macro-F1 0.4273
  - MLP GroupDRO  : macro-F1 0.3615

Outputs results/extra_baselines_results.json with one record per model.

Compute: ~10–15 min total on CPU + 1 GPU for TabNet.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import causal_feature_cols
from data import HZ_COLS, MAX_COLS, LABEL_NAMES, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
HOLDOUT_REGION = 43
SEED = 42


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def make_split():
    df = load_all()
    cols = causal_feature_cols(43)
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int64)

    te_mask = df['region_lvl1'].values == HOLDOUT_REGION
    tr_pool = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr_pool))
    n_val = int(0.10 * len(tr_pool))
    va = tr_pool[perm[:n_val]]
    tr = tr_pool[perm[n_val:]]

    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype(np.float32)
    return Xs, y, tr, va, te_idx, cols


def metrics(y_true, y_pred):
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro',
                                   zero_division=0)),
        'f1_per_class': [float(f1_score(y_true, y_pred, labels=[i],
                                        average='macro', zero_division=0))
                         for i in range(5)],
        'cm': confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
    }


def run_random_forest(X, y, tr, va, te):
    print('  fitting RandomForest (n_estimators=400, max_depth=20, '
          '4 threads)...', flush=True)
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=400, max_depth=20,
                                 n_jobs=4, random_state=SEED,
                                 class_weight='balanced')
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return {**metrics(y[te], pred), 'time_s': time.time() - t0}


def run_lightgbm(X, y, tr, va, te):
    import lightgbm as lgb
    print('  fitting LightGBM (n_estimators=200, fixed)...', flush=True)
    t0 = time.time()
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    sample_w = cls_w[y[tr]]
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1,
        num_leaves=63, max_depth=6,
        objective='multiclass', num_class=5,
        random_state=SEED, n_jobs=4, verbose=-1,
    )
    clf.fit(X[tr], y[tr], sample_weight=sample_w)
    pred = clf.predict(X[te])
    return {**metrics(y[te], pred), 'time_s': time.time() - t0,
            'best_iteration': 200}


def run_catboost(X, y, tr, va, te):
    from catboost import CatBoostClassifier
    print('  fitting CatBoost (iterations=200, fixed, 4 threads)...',
          flush=True)
    t0 = time.time()
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    clf = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1,
        loss_function='MultiClass', random_seed=SEED,
        class_weights=cls_w.tolist(),
        thread_count=4, verbose=False,
    )
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te]).reshape(-1).astype(np.int64)
    return {**metrics(y[te], pred), 'time_s': time.time() - t0,
            'best_iteration': 200}


def run_tabnet(X, y, tr, va, te):
    from pytorch_tabnet.tab_model import TabNetClassifier
    print('  fitting TabNet (max_epochs=80)...', flush=True)
    t0 = time.time()
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    sample_w = cls_w[y[tr]]
    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=4,
        gamma=1.5, lambda_sparse=1e-4,
        seed=SEED, verbose=0,
        device_name='cpu',   # avoid GPU contention with multiseed run
    )
    clf.fit(X[tr], y[tr],
            eval_set=[(X[va], y[va])], eval_metric=['accuracy'],
            max_epochs=80, patience=15, batch_size=2048,
            virtual_batch_size=512, num_workers=0,
            weights=sample_w.astype(np.float32))
    pred = clf.predict(X[te])
    return {**metrics(y[te], pred), 'time_s': time.time() - t0,
            'best_epoch': int(clf.best_epoch)}


def main():
    print('Building causal-feature split...', flush=True)
    X, y, tr, va, te, cols = make_split()
    cls_counts = [int((y[te] == c).sum()) for c in range(5)]
    print(f'  X={X.shape}  tr={len(tr)}  va={len(va)}  te={len(te)} '
          f'(per-class={cls_counts})', flush=True)

    runs = {}
    for name, fn in [
        ('random_forest', run_random_forest),
        ('lightgbm', run_lightgbm),
        ('catboost', run_catboost),
        ('tabnet', run_tabnet),
    ]:
        print(f'\n=== {name} ===', flush=True)
        try:
            out = fn(X, y, tr, va, te)
            runs[name] = out
            print(f'  {name}: f1_macro={out["f1_macro"]:.4f} '
                  f'acc={out["acc"]:.4f} ({out["time_s"]:.0f}s)',
                  flush=True)
        except Exception as exc:
            runs[name] = {'error': str(exc)}
            print(f'  {name}: FAILED — {exc}', flush=True)

    print('\n=== EXTRA BASELINES SUMMARY ===')
    print(f'{"model":>16s}  {"f1_macro":>10s}  {"acc":>8s}  {"time_s":>8s}')
    for n, r in runs.items():
        if 'f1_macro' in r:
            print(f'{n:>16s}  {r["f1_macro"]:10.4f}  {r["acc"]:8.4f}  '
                  f'{r["time_s"]:8.1f}')
        else:
            print(f'{n:>16s}  ERROR: {r["error"][:50]}')

    out_path = os.path.join(RESULTS_DIR, 'extra_baselines_results.json')
    with open(out_path, 'w') as f:
        json.dump({'features': cols, 'n_features': len(cols),
                   'holdout_region': HOLDOUT_REGION,
                   'seed': SEED, 'runs': runs}, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
