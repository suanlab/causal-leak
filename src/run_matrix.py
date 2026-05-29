"""2x2 shortcut-learning experiment: {full, signal} x {random, group_site}.

Also runs an extra 'paper_split' cell that uses the provided train/val files
directly, mirroring the original KCC paper evaluation protocol.

Results saved to ../results/matrix_results.json and ../results/matrix_table.md.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all, get_xy, LABEL_NAMES  # noqa: E402

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    HAS_SGKF = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
N_SPLITS = 5
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
    return {
        'acc': float(accuracy_score(y[te_idx], pred)),
        'f1_macro': float(f1_score(y[te_idx], pred, average='macro')),
        'f1_per_class': [
            float(f1_score(y[te_idx], pred, labels=[i], average='macro'))
            for i in range(5)
        ],
        'n_train': int(len(tr_idx)),
        'n_test': int(len(te_idx)),
        'cm': confusion_matrix(y[te_idx], pred, labels=list(range(5))).tolist(),
    }, clf


def run_kfold(df, feat_mode, split_mode):
    X, y, cols = get_xy(df, feat_mode)
    groups = df['site'].values
    print(f'\n=== {feat_mode} / {split_mode} | X={X.shape} ===', flush=True)

    if split_mode == 'random':
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y))
    elif split_mode == 'group_site':
        if HAS_SGKF:
            cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=SEED)
            splits = list(cv.split(X, y, groups=groups))
        else:
            cv = GroupKFold(n_splits=N_SPLITS)
            splits = list(cv.split(X, y, groups=groups))
    else:
        raise ValueError(split_mode)

    fold_metrics = []
    for fold, (tr, te) in enumerate(splits):
        t0 = time.time()
        m, _ = fit_eval(X, y, tr, te)
        m['fold'] = int(fold)
        m['time_s'] = float(time.time() - t0)
        test_sites = len(np.unique(groups[te]))
        m['n_test_sites'] = int(test_sites)
        fold_metrics.append(m)
        print(
            f'  fold {fold}: acc={m["acc"]:.4f} f1={m["f1_macro"]:.4f} '
            f'n_test_sites={test_sites} ({m["time_s"]:.1f}s)',
            flush=True,
        )

    accs = [m['acc'] for m in fold_metrics]
    f1s = [m['f1_macro'] for m in fold_metrics]
    return {
        'feat_mode': feat_mode,
        'split_mode': split_mode,
        'n_features': X.shape[1],
        'acc_mean': float(np.mean(accs)),
        'acc_std': float(np.std(accs)),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'folds': fold_metrics,
    }


def run_paper_split(df, feat_mode):
    X, y, _ = get_xy(df, feat_mode)
    mask_tr = (df['split'] == 'train').to_numpy()
    mask_te = (df['split'] == 'val').to_numpy()
    tr = np.where(mask_tr)[0]
    te = np.where(mask_te)[0]
    print(f'\n=== paper_split / {feat_mode} | train={len(tr)} val={len(te)} ===',
          flush=True)
    t0 = time.time()
    m, _ = fit_eval(X, y, tr, te)
    m['time_s'] = float(time.time() - t0)
    print(f'  acc={m["acc"]:.4f} f1={m["f1_macro"]:.4f} ({m["time_s"]:.1f}s)',
          flush=True)
    return {
        'feat_mode': feat_mode,
        'split_mode': 'paper_split',
        'n_features': X.shape[1],
        'acc_mean': m['acc'],
        'acc_std': 0.0,
        'f1_mean': m['f1_macro'],
        'f1_std': 0.0,
        'folds': [m],
    }


def save_table(results):
    lines = [
        '# Shortcut Learning 2x2 Experiment Results',
        '',
        '| Feature Set | Split Protocol | # Feat | Accuracy | Macro-F1 |',
        '|-------------|----------------|-------:|---------:|---------:|',
    ]
    for r in results:
        lines.append(
            f'| {r["feat_mode"]} | {r["split_mode"]} | {r["n_features"]} '
            f'| {r["acc_mean"]:.4f} ± {r["acc_std"]:.4f} '
            f'| {r["f1_mean"]:.4f} ± {r["f1_std"]:.4f} |'
        )
    out = '\n'.join(lines) + '\n'
    with open(os.path.join(RESULTS_DIR, 'matrix_table.md'), 'w') as f:
        f.write(out)
    print('\n' + out, flush=True)


def main():
    t_all = time.time()
    df = load_all()
    print(
        f'Loaded {len(df)} rows | sites={df.site.nunique()} '
        f'sensors={df.sid.nunique()} | classes={df.leaktype.value_counts().to_dict()}',
        flush=True,
    )

    results = []
    # Paper-style split (reproduction baseline)
    for feat in ['full', 'signal']:
        results.append(run_paper_split(df, feat))
    # 2x2 matrix with 5-fold CV
    for feat in ['full', 'signal']:
        for split in ['random', 'group_site']:
            results.append(run_kfold(df, feat, split))

    with open(os.path.join(RESULTS_DIR, 'matrix_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    save_table(results)
    print(f'Total elapsed: {time.time() - t_all:.1f}s', flush=True)


if __name__ == '__main__':
    main()
