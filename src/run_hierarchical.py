"""Phase 2D — Hierarchical classifier: normal gate + 4-class abnormal head.

Motivation
----------
Phase 2B showed XGBoost + causal bands is still our strongest OOD model,
largely because it trivially perfect-classifies the `normal` class via MAX
feature energy, while DL models collapse on `normal` (F1 ≈ 0.5). This phase
keeps XGBoost but explicitly decouples the two sub-problems:

  Stage 1  : binary normal vs abnormal classifier (signal energy task).
  Stage 2  : 4-class abnormal classifier trained only on abnormal rows,
             so capacity isn't spent on the trivial normal separation.

Expected effect: Stage 2 gets sharper decision surfaces between in/out/noise/
other because it doesn't have the normal pile-up in training data.

Feature set: causal 65-422 Hz bands + MAX + month (same 57 features as
Phase 2C). Split: LORO region 43.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import MAX_COLS, LABEL_NAMES, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

HOLDOUT_REGION = 43
SEED = 42

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    tree_method='hist', n_jobs=-1, random_state=SEED,
    verbosity=0,
)


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def main():
    t_all = time.time()
    df = load_all()
    cols = ['month'] + bands_in_range(65, 422) + MAX_COLS
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int32)

    mask_te = df['region_lvl1'].values == HOLDOUT_REGION
    tr = np.where(~mask_te)[0]
    te = np.where(mask_te)[0]
    print(f'features={len(cols)}  train={len(tr)}  test={len(te)}',
          flush=True)

    # -------- Stage 1: normal gate (binary) --------
    y_bin = (y == 4).astype(np.int32)   # 1 = normal, 0 = abnormal
    t0 = time.time()
    stage1 = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', **XGB_PARAMS)
    stage1.fit(X[tr], y_bin[tr])
    s1_pred = stage1.predict(X[te])
    s1_proba = stage1.predict_proba(X[te])[:, 1]
    s1_acc = float(accuracy_score(y_bin[te], s1_pred))
    s1_f1 = float(f1_score(y_bin[te], s1_pred, zero_division=0))
    s1_pos = int(s1_pred.sum())
    s1_true_pos = int(y_bin[te].sum())
    print(f'Stage 1 (normal gate): acc={s1_acc:.4f}  f1_pos={s1_f1:.4f}  '
          f'pred_normal={s1_pos}  true_normal={s1_true_pos}  '
          f'({time.time() - t0:.0f}s)', flush=True)

    # -------- Stage 2: 4-class abnormal (abnormal-only training) --------
    abn_tr = tr[y[tr] != 4]
    t0 = time.time()
    stage2 = XGBClassifier(
        objective='multi:softprob', num_class=4,
        eval_metric='mlogloss', **XGB_PARAMS)
    stage2.fit(X[abn_tr], y[abn_tr])   # labels are 0..3 on abnormal rows
    s2_pred_te = stage2.predict(X[te])
    print(f'Stage 2 (4-class abnormal): trained on {len(abn_tr)} rows  '
          f'({time.time() - t0:.0f}s)', flush=True)

    # -------- Compose --------
    final = np.where(s1_pred == 1, 4, s2_pred_te).astype(np.int32)
    acc = float(accuracy_score(y[te], final))
    f1_macro = float(f1_score(y[te], final, average='macro',
                              zero_division=0))
    f1_pc = [float(f1_score(y[te], final, labels=[i], average='macro',
                            zero_division=0)) for i in range(5)]
    cm = confusion_matrix(y[te], final, labels=list(range(5))).tolist()

    print(f'\n=== HIERARCHICAL test ===')
    print(f'  acc={acc:.4f}  f1_macro={f1_macro:.4f}')
    for name, v in zip(LABEL_NAMES, f1_pc):
        print(f'  F1[{name:>6s}] = {v:.4f}')

    # -------- Compare: flat 5-class XGBoost (baseline) --------
    t0 = time.time()
    flat = XGBClassifier(
        objective='multi:softprob', num_class=5,
        eval_metric='mlogloss', **XGB_PARAMS)
    flat.fit(X[tr], y[tr])
    flat_pred = flat.predict(X[te])
    flat_acc = float(accuracy_score(y[te], flat_pred))
    flat_f1 = float(f1_score(y[te], flat_pred, average='macro',
                             zero_division=0))
    flat_pc = [float(f1_score(y[te], flat_pred, labels=[i], average='macro',
                              zero_division=0)) for i in range(5)]
    print(f'\n=== FLAT 5-class (baseline) ===')
    print(f'  acc={flat_acc:.4f}  f1_macro={flat_f1:.4f}  '
          f'({time.time() - t0:.0f}s)')
    for name, v in zip(LABEL_NAMES, flat_pc):
        print(f'  F1[{name:>6s}] = {v:.4f}')

    out = {
        'holdout_region': HOLDOUT_REGION,
        'stage1_binary': {
            'acc': s1_acc, 'f1_pos': s1_f1,
            'pred_normal_count': s1_pos,
            'true_normal_count': s1_true_pos,
        },
        'hierarchical': {
            'acc': acc, 'f1_macro': f1_macro,
            'f1_per_class': f1_pc, 'cm': cm,
        },
        'flat': {
            'acc': flat_acc, 'f1_macro': flat_f1,
            'f1_per_class': flat_pc,
        },
        'delta': {
            'acc': acc - flat_acc,
            'f1_macro': f1_macro - flat_f1,
            'f1_per_class': [f1_pc[i] - flat_pc[i] for i in range(5)],
        },
    }
    with open(os.path.join(RESULTS_DIR, 'hierarchical_results.json'),
              'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nTotal: {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
