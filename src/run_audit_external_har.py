"""A1 / EAAI revision — External-dataset partial replication on UCI HAR.

Applies the four-probe identifier shortcut audit and the OOD ladder to
the UCI Human Activity Recognition (HAR) benchmark — 10{,}299 smartphone-
accelerometer windows from 30 subjects performing 6 activities, with
561 hand-crafted time-frequency features per window. HAR is structurally
the opposite of AI Hub: each subject performs all 6 activities, so the
per-subject label purity should be \\emph{low} and the audit should
correctly find \\emph{no} identifier shortcut.

Writes results/audit_external_har_results.json. Compute: <30 s on CPU.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__),
                         '..', 'external_datasets',
                         'har', 'UCI HAR Dataset')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42


def load_har():
    # Use numpy.loadtxt — much faster than pandas regex engine on the
    # whitespace-separated HAR text files.
    X_tr = np.loadtxt(os.path.join(DATA_DIR, 'train', 'X_train.txt'),
                       dtype=np.float32)
    y_tr = np.loadtxt(os.path.join(DATA_DIR, 'train', 'y_train.txt'),
                       dtype=np.int64) - 1
    s_tr = np.loadtxt(os.path.join(DATA_DIR, 'train',
                                    'subject_train.txt'),
                       dtype=np.int64)
    X_te = np.loadtxt(os.path.join(DATA_DIR, 'test', 'X_test.txt'),
                       dtype=np.float32)
    y_te = np.loadtxt(os.path.join(DATA_DIR, 'test', 'y_test.txt'),
                       dtype=np.int64) - 1
    s_te = np.loadtxt(os.path.join(DATA_DIR, 'test',
                                    'subject_test.txt'),
                       dtype=np.int64)

    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    s = np.concatenate([s_tr, s_te], axis=0)
    return X, y, s


def compute_p1(s, y):
    """Per-subject label purity = max-class share within each subject."""
    df = pd.DataFrame({'s': s, 'y': y})
    p = df.groupby('s')['y'].apply(
        lambda v: v.value_counts(normalize=True).iloc[0])
    return {
        'mean': float(p.mean()),
        'median': float(p.median()),
        'frac_purity_1.0': float((p == 1.0).mean()),
        'n_subjects': int(len(p)),
    }


def compute_p2(s_tr, y_tr, s_te, y_te):
    """P2: subject->modal-label lookup."""
    lookup = pd.Series(y_tr, index=s_tr).groupby(level=0).agg(
        lambda v: pd.Series(v).value_counts().index[0]).to_dict()
    pred = np.array([lookup.get(si, -1) for si in s_te])
    covered = pred != -1
    accuracy = float((pred[covered] == y_te[covered]).mean()) \
                if covered.any() else 0.0
    return {
        'coverage': float(covered.mean()),
        'covered_accuracy': accuracy,
    }


def compute_p3(s_tr, s_te):
    overlap = set(s_te) & set(s_tr)
    return {
        'val_subjects_in_train': len(overlap),
        'val_subjects_total': int(len(set(s_te))),
        'overlap_fraction': float(len(overlap) / max(1, len(set(s_te)))),
    }


def train_eval(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    clf = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective='multi:softprob', num_class=6,
        tree_method='hist', n_jobs=4, random_state=SEED,
        eval_metric='mlogloss', verbosity=0,
    )
    clf.fit(X_tr_s, y_tr)
    p = clf.predict(X_te_s)
    return {
        'acc': float(accuracy_score(y_te, p)),
        'f1_macro': float(f1_score(y_te, p, average='macro',
                                    zero_division=0)),
    }


def main():
    print('Loading HAR...', flush=True)
    X, y, s = load_har()
    print(f'  rows={len(X)}  features={X.shape[1]}  '
          f'classes={len(np.unique(y))}  subjects={len(np.unique(s))}',
          flush=True)

    # ====== 4-probe audit (row-level 90/10 split) ======
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    n_val = int(0.1 * len(X))
    va = perm[:n_val]
    tr = perm[n_val:]

    audit = {
        'P1_per_subject_purity': compute_p1(s[tr], y[tr]),
        'P2_subject_lookup':     compute_p2(s[tr], y[tr], s[va], y[va]),
        'P3_subject_overlap':    compute_p3(s[tr], s[va]),
    }
    print('\n=== 4-probe audit on HAR (row-level 90/10 split) ===')
    for k, v in audit.items():
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv}')

    # ====== 3-tier ladder ======
    print('\n=== 3-tier evaluation ladder (5-fold) ===', flush=True)
    ladder = {}

    # Row-CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr_idx, te_idx in skf.split(X, y):
        m = train_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        accs.append(m['acc']); f1s.append(m['f1_macro'])
    ladder['row_cv'] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_macro_mean': float(np.mean(f1s)),
        'f1_macro_std': float(np.std(f1s)),
    }
    print(f'  Row-CV   acc={ladder["row_cv"]["acc_mean"]:.4f}  '
          f'f1={ladder["row_cv"]["f1_macro_mean"]:.4f}', flush=True)

    # GroupSubject-CV
    gkf = GroupKFold(n_splits=5)
    accs, f1s = [], []
    for tr_idx, te_idx in gkf.split(X, y, groups=s):
        m = train_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        accs.append(m['acc']); f1s.append(m['f1_macro'])
    ladder['group_subject_cv'] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_macro_mean': float(np.mean(f1s)),
        'f1_macro_std': float(np.std(f1s)),
    }
    print(f'  GroupSubject-CV  acc={ladder["group_subject_cv"]["acc_mean"]:.4f}  '
          f'f1={ladder["group_subject_cv"]["f1_macro_mean"]:.4f}',
          flush=True)

    ladder['row_to_group_drop'] = (
        ladder['row_cv']['acc_mean'] - ladder['group_subject_cv']['acc_mean'])

    out = {
        'dataset': 'UCI HAR (Human Activity Recognition)',
        'n_rows': int(len(X)),
        'n_features': int(X.shape[1]),
        'n_classes': int(len(np.unique(y))),
        'n_subjects': int(len(np.unique(s))),
        'audit': audit,
        'ladder': ladder,
        'seed': SEED,
    }
    out_path = os.path.join(RESULTS_DIR, 'audit_external_har_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n=== HAR external-replication summary ===')
    print(f'  P1 per-subject purity mean: '
          f'{audit["P1_per_subject_purity"]["mean"]:.4f}  '
          f'(frac=1.0: {audit["P1_per_subject_purity"]["frac_purity_1.0"]:.4f})')
    print(f'  P2 subject-lookup covered acc: '
          f'{audit["P2_subject_lookup"]["covered_accuracy"]:.4f}')
    print(f'  Row-CV acc={ladder["row_cv"]["acc_mean"]:.4f}  '
          f'GroupSubject-CV acc={ladder["group_subject_cv"]["acc_mean"]:.4f}  '
          f'(drop={ladder["row_to_group_drop"]*100:+.1f} pp)')
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'Total: {time.time() - t0:.1f}s')
