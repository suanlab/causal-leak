"""A1 / EAAI revision — External-dataset partial replication on SECOM.

Applies the four-probe identifier shortcut audit and a group-cross-validation
ladder to the UCI SECOM benchmark (1{,}567 semiconductor-process wafers,
591 sensor channels, binary pass/fail). The SECOM dataset has no
explicit identifier column, but timestamps allow construction of
session IDs (per-hour bins) that act as a natural identifier
(a single hour's wafers come from the same fab tool / shift state
and therefore form a coherent group).

The audit asks: does a shortcut exist where session ID alone predicts
pass/fail on the row-level split? And how large is the row-CV
$\\rightarrow$ session-CV $\\rightarrow$ day-CV accuracy drop?

Writes results/audit_external_secom_results.json. Compute: <30 s on CPU.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'external_datasets')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42


def load_secom():
    """Load SECOM into a DataFrame with `label`, `timestamp`,
    `session_id` (per-hour bin), `day` (per-day bin), and 590 feature columns."""
    X = pd.read_csv(os.path.join(DATA_DIR, 'secom.data'),
                    header=None, sep=r'\s+',
                    na_values='NaN', engine='python')
    labels_df = pd.read_csv(os.path.join(DATA_DIR, 'secom_labels.data'),
                            header=None, sep=r'(?<=\d)\s+(?=")',
                            engine='python')
    labels_df.columns = ['label', 'timestamp']
    labels_df['timestamp'] = labels_df['timestamp'].str.strip('"')
    labels_df['ts'] = pd.to_datetime(labels_df['timestamp'],
                                      format='%d/%m/%Y %H:%M:%S')
    df = pd.concat([X.reset_index(drop=True), labels_df.reset_index(drop=True)],
                   axis=1)
    df['label'] = (df['label'] == 1).astype(int)   # 1 = fail (positive)
    df['session_id'] = df['ts'].dt.strftime('%Y%m%d_%H')
    df['day_id'] = df['ts'].dt.strftime('%Y%m%d')

    # Impute NaN with column means
    feat_cols = list(range(X.shape[1]))
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].mean())
    return df, feat_cols


def compute_p1(df, sid_col='session_id'):
    """Per-session label purity."""
    p = df.groupby(sid_col)['label'].apply(
        lambda s: s.value_counts(normalize=True).iloc[0])
    return {
        'mean': float(p.mean()),
        'median': float(p.median()),
        'frac_purity_1.0': float((p == 1.0).mean()),
        'n_groups': int(len(p)),
    }


def compute_p2(df_tr, df_te, sid_col='session_id'):
    """P2: session_id-modal-label lookup."""
    lookup = df_tr.groupby(sid_col)['label'].agg(
        lambda s: s.value_counts().index[0]).to_dict()
    pred = df_te[sid_col].map(lookup)
    covered = pred.notna()
    accuracy = float((pred[covered] == df_te.loc[covered, 'label']).mean()) \
                if covered.any() else 0.0
    return {
        'coverage': float(covered.mean()),
        'covered_accuracy': accuracy,
    }


def compute_p3(df_tr, df_te, sid_col='session_id'):
    """Train/test overlap of session IDs."""
    overlap = set(df_te[sid_col]) & set(df_tr[sid_col])
    return {
        'val_groups_in_train': len(overlap),
        'val_groups_total': int(df_te[sid_col].nunique()),
        'overlap_fraction': float(len(overlap)
                                   / max(1, df_te[sid_col].nunique())),
    }


def train_eval_xgb(X_tr, y_tr, X_te, y_te, seed=SEED):
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    # n_jobs limited to 4 to avoid CPU thrash on the small SECOM split.
    # n_estimators reduced to 100 since SECOM has only ~1.5k rows and
    # overfitting is the main risk.
    clf = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective='binary:logistic',
        tree_method='hist', n_jobs=4, random_state=seed,
        eval_metric='logloss', verbosity=0,
    )
    clf.fit(X_tr_s, y_tr)
    p = clf.predict(X_te_s)
    return {
        'acc': float(accuracy_score(y_te, p)),
        'f1_macro': float(f1_score(y_te, p, average='macro',
                                    zero_division=0)),
        'f1_pos': float(f1_score(y_te, p, average='binary',
                                  pos_label=1, zero_division=0)),
    }


def main():
    print('Loading SECOM...', flush=True)
    df, feat_cols = load_secom()
    print(f'  rows={len(df)}  features={len(feat_cols)}  '
          f'positive (fail) rate={df["label"].mean():.4f}', flush=True)
    print(f'  sessions (per-hour) = {df["session_id"].nunique()}  '
          f'days = {df["day_id"].nunique()}', flush=True)

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df['label'].to_numpy(dtype=np.int64)

    # ====== 4-probe audit (mimic the AI Hub paper-split) ======
    # Use a stratified 90/10 row-level split as the "PaperSplit" analogue.
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_val = int(0.1 * len(df))
    va_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    df_tr = df.iloc[tr_idx]
    df_va = df.iloc[va_idx]

    audit = {
        'P1_per_session_purity': compute_p1(df_tr, 'session_id'),
        'P1_per_day_purity':     compute_p1(df_tr, 'day_id'),
        'P2_session_lookup':     compute_p2(df_tr, df_va, 'session_id'),
        'P2_day_lookup':         compute_p2(df_tr, df_va, 'day_id'),
        'P3_session_overlap':    compute_p3(df_tr, df_va, 'session_id'),
        'P3_day_overlap':        compute_p3(df_tr, df_va, 'day_id'),
    }
    print('\n=== 4-probe audit on SECOM (row-level 90/10 split) ===')
    for k, v in audit.items():
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv}')

    # ====== 3-tier evaluation ladder ======
    print('\n=== 3-tier evaluation ladder (5-fold) ===', flush=True)
    ladder = {}

    # Row-CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        m = train_eval_xgb(X[tr], y[tr], X[te], y[te])
        accs.append(m['acc']); f1s.append(m['f1_macro'])
    ladder['row_cv'] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_macro_mean': float(np.mean(f1s)),
        'f1_macro_std': float(np.std(f1s)),
    }
    print(f'  Row-CV   acc={ladder["row_cv"]["acc_mean"]:.4f}  '
          f'f1={ladder["row_cv"]["f1_macro_mean"]:.4f}', flush=True)

    # GroupSession-CV
    gkf = GroupKFold(n_splits=5)
    accs, f1s = [], []
    for tr, te in gkf.split(X, y, groups=df['session_id'].values):
        m = train_eval_xgb(X[tr], y[tr], X[te], y[te])
        accs.append(m['acc']); f1s.append(m['f1_macro'])
    ladder['group_session_cv'] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_macro_mean': float(np.mean(f1s)),
        'f1_macro_std': float(np.std(f1s)),
    }
    print(f'  GroupSession-CV  acc={ladder["group_session_cv"]["acc_mean"]:.4f}  '
          f'f1={ladder["group_session_cv"]["f1_macro_mean"]:.4f}', flush=True)

    # GroupDay-CV
    accs, f1s = [], []
    for tr, te in gkf.split(X, y, groups=df['day_id'].values):
        m = train_eval_xgb(X[tr], y[tr], X[te], y[te])
        accs.append(m['acc']); f1s.append(m['f1_macro'])
    ladder['group_day_cv'] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_macro_mean': float(np.mean(f1s)),
        'f1_macro_std': float(np.std(f1s)),
    }
    print(f'  GroupDay-CV  acc={ladder["group_day_cv"]["acc_mean"]:.4f}  '
          f'f1={ladder["group_day_cv"]["f1_macro_mean"]:.4f}', flush=True)

    # Aggregate (row-CV minus group-CV drop)
    ladder['row_to_group_session_drop'] = (
        ladder['row_cv']['acc_mean'] - ladder['group_session_cv']['acc_mean'])
    ladder['row_to_group_day_drop'] = (
        ladder['row_cv']['acc_mean'] - ladder['group_day_cv']['acc_mean'])

    out = {
        'dataset': 'SECOM (UCI)',
        'n_rows': len(df),
        'n_features': len(feat_cols),
        'positive_rate': float(y.mean()),
        'n_sessions_per_hour': int(df['session_id'].nunique()),
        'n_days': int(df['day_id'].nunique()),
        'audit': audit,
        'ladder': ladder,
        'seed': SEED,
    }

    out_path = os.path.join(RESULTS_DIR, 'audit_external_secom_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n=== SECOM external-replication summary ===')
    print(f'  P2 session-lookup covered accuracy: '
          f'{audit["P2_session_lookup"]["covered_accuracy"]:.4f}  '
          f'(coverage={audit["P2_session_lookup"]["coverage"]:.3f})')
    print(f'  Row-CV acc={ladder["row_cv"]["acc_mean"]:.4f}, '
          f'GroupSession-CV acc={ladder["group_session_cv"]["acc_mean"]:.4f} '
          f'(drop={ladder["row_to_group_session_drop"]*100:+.1f} pp)')
    print(f'  Row-CV acc={ladder["row_cv"]["acc_mean"]:.4f}, '
          f'GroupDay-CV acc={ladder["group_day_cv"]["acc_mean"]:.4f} '
          f'(drop={ladder["row_to_group_day_drop"]*100:+.1f} pp)')
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'Total: {time.time() - t0:.1f}s')
