"""A1 / EAAI revision — External-dataset partial replication on
MIMII-DG (DCASE 2022 task 2, dev set: bearing).

Applies the four-probe identifier shortcut audit and the OOD ladder
to the MIMII-DG bearing subset. MIMII-DG is purpose-built for
domain generalisation: each machine has a `section` (sub-type),
and recordings split into `source` (training distribution) and
`target` (shifted distribution) domains. The natural audit
correspondence is:

  - identifier (P1, P2): `section` ID (0, 1, 2)
  - group hierarchy (P3, OOD ladder): `section` and `domain`
  - physical feature group: log-mel-spectrogram channels (64 mel bins,
    mean+std aggregated over time per recording)

Writes results/audit_external_mimii_results.json. Compute: ~5 min
on CPU (feature extraction dominates).
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__),
                         '..', 'external_datasets', 'mimii_dg')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42
SR = 16000
N_MELS = 64
WIN_S = 10.0   # MIMII-DG clips are 10 s


def parse_filename(path):
    """MIMII-DG filename format:
       section_<sect>_<src/tgt>_<train/test>_<normal/anomaly>_<id>_*.wav
    """
    name = os.path.basename(path)
    m = re.match(r'section_(\d+)_(source|target)_(train|test)_'
                 r'(normal|anomaly)_(\d+)', name)
    if not m:
        return None
    return {
        'path': path,
        'section': int(m.group(1)),
        'domain': m.group(2),
        'split': m.group(3),
        'label': 1 if m.group(4) == 'anomaly' else 0,
        'id': int(m.group(5)),
    }


def extract_features(path):
    """Aggregate log-mel spectrogram into [mean_64 + std_64] = 128 features."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=512)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])


def build_dataset(machine_dir):
    """Walk all .wav under machine_dir, return DataFrame + feature matrix."""
    files = glob.glob(os.path.join(machine_dir, '**', '*.wav'),
                       recursive=True)
    records = []
    for f in files:
        info = parse_filename(f)
        if info is not None:
            records.append(info)
    print(f'  found {len(records)} WAV files', flush=True)
    df = pd.DataFrame(records)

    feats = []
    t0 = time.time()
    for i, p in enumerate(df['path']):
        feats.append(extract_features(p))
        if (i + 1) % 200 == 0:
            print(f'    extracted {i+1}/{len(df)}  '
                  f'({(time.time()-t0):.0f}s)', flush=True)
    X = np.stack(feats).astype(np.float32)
    return df, X


def compute_p1(df, group_col='section'):
    p = df.groupby(group_col)['label'].apply(
        lambda v: v.value_counts(normalize=True).iloc[0])
    return {
        'mean': float(p.mean()),
        'median': float(p.median()),
        'frac_purity_1.0': float((p == 1.0).mean()),
        'n_groups': int(len(p)),
    }


def compute_p2(df_tr, df_te, group_col='section'):
    lookup = df_tr.groupby(group_col)['label'].agg(
        lambda v: v.value_counts().index[0]).to_dict()
    pred = df_te[group_col].map(lookup)
    covered = pred.notna()
    accuracy = float((pred[covered] == df_te.loc[covered, 'label']).mean()) \
                if covered.any() else 0.0
    return {
        'coverage': float(covered.mean()),
        'covered_accuracy': accuracy,
    }


def compute_p3(df_tr, df_te, group_col='section'):
    overlap = set(df_te[group_col]) & set(df_tr[group_col])
    return {
        'val_groups_in_train': len(overlap),
        'val_groups_total': int(df_te[group_col].nunique()),
        'overlap_fraction': float(len(overlap)
                                    / max(1, df_te[group_col].nunique())),
    }


def train_eval(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='binary:logistic',
        tree_method='hist', n_jobs=4, random_state=SEED,
        eval_metric='logloss', verbosity=0,
    )
    clf.fit(X_tr_s, y_tr)
    p = clf.predict(X_te_s)
    return {
        'acc': float(accuracy_score(y_te, p)),
        'f1_macro': float(f1_score(y_te, p, average='macro',
                                    zero_division=0)),
    }


def main(machine='bearing'):
    machine_dir = os.path.join(DATA_DIR, machine)
    print(f'Loading MIMII-DG {machine} from {machine_dir} ...', flush=True)
    df, X = build_dataset(machine_dir)
    y = df['label'].to_numpy()

    print(f'  rows={len(df)}  features={X.shape[1]}  '
          f'sections={sorted(df["section"].unique().tolist())}  '
          f'domains={sorted(df["domain"].unique().tolist())}  '
          f'positive (anomaly) rate={y.mean():.4f}', flush=True)

    # ====== 4-probe audit (90/10 row split for P2/P3) ======
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_val = int(0.1 * len(df))
    va = perm[:n_val]
    tr = perm[n_val:]

    audit = {
        'P1_per_section_purity': compute_p1(df, 'section'),
        'P1_per_section_domain_purity': compute_p1(
            df.assign(sd=df['section'].astype(str) + '_' + df['domain']),
            'sd'),
        'P2_section_lookup': compute_p2(df.iloc[tr], df.iloc[va], 'section'),
        'P3_section_overlap': compute_p3(df.iloc[tr], df.iloc[va], 'section'),
    }
    print('\n=== 4-probe audit on MIMII-DG ===')
    for k, v in audit.items():
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv}')

    # ====== 3-tier OOD ladder ======
    print('\n=== 3-tier evaluation ladder ===', flush=True)
    ladder = {}

    # Row-CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr_i, te_i in skf.split(X, y):
        m = train_eval(X[tr_i], y[tr_i], X[te_i], y[te_i])
        accs.append(m['acc']); f1s.append(m['f1_macro'])
    ladder['row_cv'] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_macro_mean': float(np.mean(f1s)),
        'f1_macro_std': float(np.std(f1s)),
    }
    print(f'  Row-CV   acc={ladder["row_cv"]["acc_mean"]:.4f}  '
          f'f1={ladder["row_cv"]["f1_macro_mean"]:.4f}', flush=True)

    # GroupSection-CV
    accs, f1s = [], []
    sections = df['section'].values
    gkf = GroupKFold(n_splits=3)
    for tr_i, te_i in gkf.split(X, y, groups=sections):
        m = train_eval(X[tr_i], y[tr_i], X[te_i], y[te_i])
        accs.append(m['acc']); f1s.append(m['f1_macro'])
    ladder['group_section_cv'] = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_macro_mean': float(np.mean(f1s)),
        'f1_macro_std': float(np.std(f1s)),
    }
    print(f'  GroupSection-CV  acc={ladder["group_section_cv"]["acc_mean"]:.4f}  '
          f'f1={ladder["group_section_cv"]["f1_macro_mean"]:.4f}',
          flush=True)

    # GroupDomain-CV: train on source, eval on target
    src_idx = np.where(df['domain'].values == 'source')[0]
    tgt_idx = np.where(df['domain'].values == 'target')[0]
    if len(src_idx) > 0 and len(tgt_idx) > 0:
        m = train_eval(X[src_idx], y[src_idx], X[tgt_idx], y[tgt_idx])
        ladder['source_to_target'] = m
        print(f'  Source->Target  acc={m["acc"]:.4f}  f1={m["f1_macro"]:.4f}',
              flush=True)

    ladder['row_to_group_section_drop'] = (
        ladder['row_cv']['acc_mean'] - ladder['group_section_cv']['acc_mean'])

    out = {
        'dataset': f'MIMII-DG ({machine})',
        'n_rows': int(len(df)),
        'n_features': int(X.shape[1]),
        'sections': sorted(df['section'].unique().tolist()),
        'domains': sorted(df['domain'].unique().tolist()),
        'positive_rate': float(y.mean()),
        'audit': audit,
        'ladder': ladder,
        'seed': SEED,
    }
    out_path = os.path.join(RESULTS_DIR,
                             f'audit_external_mimii_{machine}_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n=== MIMII-DG external-replication summary ===')
    print(f'  P1 per-section purity mean: '
          f'{audit["P1_per_section_purity"]["mean"]:.4f}')
    print(f'  P2 section-lookup covered acc: '
          f'{audit["P2_section_lookup"]["covered_accuracy"]:.4f}')
    print(f'  Row-CV acc={ladder["row_cv"]["acc_mean"]:.4f}  '
          f'GroupSection-CV acc={ladder["group_section_cv"]["acc_mean"]:.4f}  '
          f'(drop={ladder["row_to_group_section_drop"]*100:+.1f} pp)')
    if 'source_to_target' in ladder:
        print(f'  Source->Target acc={ladder["source_to_target"]["acc"]:.4f}')
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    machine = sys.argv[1] if len(sys.argv) > 1 else 'bearing'
    t0 = time.time()
    main(machine)
    print(f'Total: {time.time() - t0:.1f}s')
