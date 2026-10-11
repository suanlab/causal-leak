"""Is binary normal-vs-leak detection reading leak physics or a site signature?

Background. No sensor in the benchmark carries both labels (0 of 2,710), and
normal recordings exist at only 8 of 49 installations. A binary detector could
therefore separate normal from leak by recognising the kind of installation a
recording came from rather than anything about leak noise. The metadata field
`lrate` is 90 for every leak recording and 30-70 for every normal one; earlier
analysis (DATASET_REPORT.md) found it is assigned after labelling, so it is not
a model input, but whether it tracks anything spectral is checked here too.

Three tests, all with the full 534-column physical input (the most capable
feature set, so the conservative choice for a confound test):

A. Within-installation separation. For each installation holding at least
   MIN_PER_CLASS normal and leak recordings, train on every region except the
   installation's own and score its recordings. Inside one installation a
   site signature is constant, so it cannot separate the classes; an AUC well
   above 0.5 there is signal beyond the site. Reported per recording and per
   sensor (mean score), with the miss rate on that site's leak recordings.

B. Campaign signature. Leak recordings only. Can the spectrum tell leak
   recordings at the 8 normal-bearing installations from leak recordings at
   the 41 leak-only ones? Grouped by installation. A high AUC means those 8
   installations share a signature a detector could exploit.

C. lrate dose-response. Among held-out normal recordings from test A, does
   the detector's leak score rise with lrate? Spearman correlation.

Output: results/lrate_confound_results.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42
NORMAL = 4
MIN_PER_CLASS = 30

XGB_BIN = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
               objective='binary:logistic', eval_metric='logloss',
               tree_method='hist', n_jobs=-1, random_state=SEED,
               verbosity=0)


def auc_or_none(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def main():
    t_all = time.time()
    df = load_all()
    cols = ['month'] + HZ_COLS + MAX_COLS
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    is_leak = (df['label'].values != NORMAL).astype(np.int32)
    site = df['site'].astype(str).values
    sid = df['sid'].astype(str).values
    region = df['region_lvl1'].values
    lrate = df['lrate'].values

    # ---------------- A. within-installation separation ----------------
    comp = {}
    for s in np.unique(site):
        m = site == s
        comp[s] = (int((is_leak[m] == 0).sum()), int(is_leak[m].sum()),
                   int(region[m][0]))
    mixed = sorted([s for s, (n, l, _) in comp.items()
                    if n >= MIN_PER_CLASS and l >= MIN_PER_CLASS])
    regions_needed = sorted({comp[s][2] for s in mixed})
    print(f'A. mixed installations with >= {MIN_PER_CLASS} per class: '
          f'{len(mixed)}  (regions {regions_needed})', flush=True)

    test_a, normal_scores = [], []
    for r in regions_needed:
        tr = np.where(region != r)[0]
        te = np.where(region == r)[0]
        t0 = time.time()
        clf = XGBClassifier(**XGB_BIN)
        clf.fit(X[tr], is_leak[tr])
        score = np.zeros(len(df), dtype=np.float64)
        score[te] = clf.predict_proba(X[te])[:, 1]
        region_auc = auc_or_none(is_leak[te], score[te])
        print(f'  region {r}: trained {time.time() - t0:.0f}s, '
              f'whole-region AUC={region_auc:.4f}', flush=True)

        for s in [x for x in mixed if comp[x][2] == r]:
            idx = np.where(site == s)[0]
            y_s, sc_s = is_leak[idx], score[idx]
            # sensor-level: one score per sensor (mean), one label per sensor
            sens = {}
            for i in idx:
                sens.setdefault(sid[i], []).append(score[i])
            s_ids = sorted(sens)
            s_y = np.array([is_leak[np.where(sid == k)[0][0]] for k in s_ids])
            s_sc = np.array([np.mean(sens[k]) for k in s_ids])
            rec = {
                'installation': s, 'region': int(r),
                'n_normal': comp[s][0], 'n_leak': comp[s][1],
                'n_normal_sensors': int((s_y == 0).sum()),
                'n_leak_sensors': int((s_y == 1).sum()),
                'auc_recording': auc_or_none(y_s, sc_s),
                'auc_sensor': auc_or_none(s_y, s_sc),
                'miss_rate_at_0.5': float((sc_s[y_s == 1] < 0.5).mean()),
                'false_alarm_rate_at_0.5': float((sc_s[y_s == 0] >= 0.5).mean()),
                'whole_region_auc': region_auc,
            }
            test_a.append(rec)
            print(f'    {s}: n={rec["n_normal"]}/{rec["n_leak"]} '
                  f'sensors={rec["n_normal_sensors"]}/{rec["n_leak_sensors"]}  '
                  f'AUC rec={rec["auc_recording"]:.4f} '
                  f'sensor={rec["auc_sensor"]:.4f}  '
                  f'miss={rec["miss_rate_at_0.5"]:.4f} '
                  f'FA={rec["false_alarm_rate_at_0.5"]:.4f}', flush=True)

        nrm = te[is_leak[te] == 0]
        normal_scores.extend(zip(lrate[nrm].tolist(), score[nrm].tolist(),
                                 [r] * len(nrm)))

    # ---------------- B. campaign signature among leak recordings ----------
    normal_bearing = {s for s, (n, _, _) in comp.items() if n > 0}
    leak_idx = np.where(is_leak == 1)[0]
    yb = np.array([1 if site[i] in normal_bearing else 0 for i in leak_idx])
    gb = site[leak_idx]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(leak_idx))
    fold_aucs = []
    print(f'\nB. campaign signature: {int(yb.sum())} leak recordings at '
          f'{len(normal_bearing)} normal-bearing installations vs '
          f'{int((yb == 0).sum())} at leak-only ones', flush=True)
    for k, (a, b) in enumerate(cv.split(X[leak_idx], yb, gb)):
        clf = XGBClassifier(**XGB_BIN)
        clf.fit(X[leak_idx][a], yb[a])
        oof[b] = clf.predict_proba(X[leak_idx][b])[:, 1]
        fold_aucs.append(auc_or_none(yb[b], oof[b]))
        print(f'  fold {k}: AUC={fold_aucs[-1]}', flush=True)
    test_b = {
        'n_positive_recordings': int(yb.sum()),
        'n_negative_recordings': int((yb == 0).sum()),
        'oof_auc': auc_or_none(yb, oof),
        'fold_aucs': fold_aucs,
    }
    print(f'  pooled out-of-fold AUC = {test_b["oof_auc"]:.4f}', flush=True)

    # ---------------- C. lrate dose-response among normals ----------------
    lr = np.array([a for a, _, _ in normal_scores])
    sc = np.array([b for _, b, _ in normal_scores])
    rho, p = spearmanr(lr, sc)
    by_level = {int(v): {'n': int((lr == v).sum()),
                         'mean_leak_score': float(sc[lr == v].mean()),
                         'frac_score_ge_0.5': float((sc[lr == v] >= 0.5).mean())}
                for v in sorted(np.unique(lr))}
    test_c = {'spearman_rho': float(rho), 'p_value': float(p),
              'n_normal_recordings': int(len(lr)), 'by_lrate': by_level}
    print(f'\nC. lrate vs leak score among held-out normals: '
          f'Spearman rho={rho:.4f} (p={p:.3g}, n={len(lr)})', flush=True)
    for v, d in by_level.items():
        print(f'  lrate={v}: n={d["n"]:>6}  mean score={d["mean_leak_score"]:.5f}'
              f'  frac>=0.5={d["frac_score_ge_0.5"]:.5f}', flush=True)

    out = {'seed': SEED, 'feature_set': 'signal_all (534)',
           'min_per_class': MIN_PER_CLASS,
           'A_within_installation': test_a,
           'B_campaign_signature': test_b,
           'C_lrate_dose_response': test_c,
           'time_s': float(time.time() - t_all)}
    path = os.path.join(RESULTS_DIR, 'lrate_confound_results.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'\nSaved {path}  ({time.time() - t_all:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
