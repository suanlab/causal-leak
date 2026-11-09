"""Operating characteristics of audit steps A1 and A2 on planted shortcuts.

The water benchmark is one release; this script asks whether the protocol's
tests behave as intended where the truth is known. It uses the public CWRU
bearing data (12 kHz drive end; normal baseline and nine seeded faults, each
bearing recorded at motor loads 0-3 hp; fetch with scripts/fetch_cwru.sh into
$CWRU_DIR) and pure simulation.

Each recording is cut into 2048-sample windows. A window's features mimic the
water release: a 128-bin log-magnitude spectrum and the peak absolute
amplitude of each of ten consecutive sub-windows (the "checks"), in integer
device units (amplitude x 1000).

A1, real units. Unit = bearing (one label for its whole life): a window-level
random split should give lift near 1. Unit = motor load (every load holds
every class): lift near 0 -- the within-benchmark analogue of grouping on a
field that carries the class prior.

A1, simulation. G units of m recordings, K = 5 classes with the water class
shares; each recording takes its unit's home label with probability lambda,
otherwise a label drawn from the shares. lambda = 0 is the null. Rejection
rate of the permutation test at 5 % over repetitions.

A2, planted device rule. A device statistic g = #(sub-window peaks >= T) is
logged as lrate = 10 g for g <= 8 and as the code 90 for g >= 9.
  S1  selection-defined: the release keeps g in {3..7} U {9, 10}; windows with
      g <= 7 are relabelled 'quiet', the rest keep their fault label.
  S2  rule-defined, not selected: every window kept, windows below the
      device code (g <= 8) relabelled 'quiet'.
  S3  easy class, not rule-defined: original labels, target = normal baseline,
      no logged field.
T1 (identity), T2 (sufficiency: LightGBM on the spectrum, peaks and count
against LightGBM on the count alone, target vs rest, under load-grouped
4-fold; both learners see the same folds, and bearing grouping would leave the
single normal bearing out of training whenever it is tested) and T3 (cluster
-bootstrap ratio at the boundary bin, fixed at 8 by the code, and at the
inclusion floor; clusters are recording files) are applied to each.
T3 power: S1 with a fraction epsilon of the boundary-bin windows put back.
T3 size: unselected releases at six thresholds, every interior bin, each
replicate release resampled by recording file.
A3 on CWRU: fault-type typing (inner race / ball / outer race) with each
fault size held out, against 1/K.

The planted 'quiet' class correlates with the real normal baseline, whose
amplitudes are low; the water release has the same structure.

Output: results/audit_semisynthetic_results.json
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import scipy.io as sio
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from audit_tests import (counts_at, depletion_test, identify_k,  # noqa: E402
                         identify_threshold, lift_test)

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
CWRU_DIR = os.environ.get('CWRU_DIR', os.path.join(os.path.dirname(__file__),
                                                   '..', '..', 'external', 'cwru'))
SEED = 42
WIN, NSUB, NBIN = 2048, 10, 128
CODE = 90
BEARINGS = {'normal': (97, 98, 99, 100),
            'IR007': (105, 106, 107, 108), 'B007': (118, 119, 120, 121),
            'OR007': (130, 131, 132, 133), 'IR014': (169, 170, 171, 172),
            'B014': (185, 186, 187, 188), 'OR014': (197, 198, 199, 200),
            'IR021': (209, 210, 211, 212), 'B021': (222, 223, 224, 225),
            'OR021': (234, 235, 236, 237)}
WATER_SHARES = np.array([14932, 19731, 5658, 7897, 22166]) / 70384


def load_cwru():
    feats, peaks, label, bearing, load, rec = [], [], [], [], [], []
    for b, ids in BEARINGS.items():
        for hp, fid in enumerate(ids):
            path = os.path.join(CWRU_DIR, f'{fid}.mat')
            if not os.path.exists(path):
                raise SystemExit(f'{path} missing: run scripts/fetch_cwru.sh '
                                 '(or set CWRU_DIR)')
            x = sio.loadmat(path)[f'X{fid:03d}_DE_time'].ravel()
            n = len(x) // WIN
            w = x[:n * WIN].reshape(n, WIN)
            mag = np.abs(np.fft.rfft(w, axis=1))[:, 1:]            # 1024 bins
            feats.append(np.log1p(mag.reshape(n, NBIN, -1).mean(2)))
            sub = w[:, :WIN - WIN % NSUB].reshape(n, NSUB, -1)
            peaks.append(np.rint(np.abs(sub).max(2) * 1000).astype(int))
            label += ['normal' if b == 'normal' else b[:-3]] * n
            bearing += [b] * n
            load += [hp] * n
            rec += [fid] * n
    return (np.vstack(feats), np.vstack(peaks), np.array(label),
            np.array(bearing), np.array(load), np.array(rec))


def a1_real(label, bearing, load, rng):
    idx = np.arange(len(label))
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=label)
    out = {}
    for name, unit in (('bearing', bearing), ('load', load)):
        out[name] = lift_test(unit[tr], label[tr], unit[te], label[te],
                              n_perm=1000, rng=rng)
    # Bearing-held-out: the three 0.021-inch bearings are unseen, so the lookup
    # cannot inform (overlap 0).
    te = np.isin(bearing, ['IR021', 'B021', 'OR021'])
    tr, te = idx[~te], idx[te]
    out['bearing_held_out'] = lift_test(bearing[tr], label[tr], bearing[te],
                                        label[te], n_perm=1000, rng=rng)
    return out


def a1_simulation(rng, lambdas=(0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
                  sizes=(50, 500, 2700), m=18, reps=200, n_perm=200):
    K = len(WATER_SHARES)
    out = {}
    for G in sizes:
        rows = []
        for lam in lambdas:
            rej, rej_eff, lifts = 0, 0, []
            for _ in range(reps):
                home = rng.choice(K, G, p=WATER_SHARES)
                unit = np.repeat(np.arange(G), m)
                y = np.where(rng.random(G * m) < lam, home[unit],
                             rng.choice(K, G * m, p=WATER_SHARES))
                te = rng.random(G * m) < 0.3
                r = lift_test(unit[~te], y[~te], unit[te], y[te], n_perm=n_perm, rng=rng)
                rej += r['p_value'] <= 0.05
                rej_eff += r['reject']
                lifts.append(r['lift'])
            rows.append({'lambda': lam, 'reject_rate_permutation': rej / reps,
                         'reject_rate': rej_eff / reps,
                         'mean_lift': float(np.mean(lifts))})
            print(f'  A1 sim G={G} lambda={lam}: reject {rej_eff / reps:.3f} '
                  f'(permutation only {rej / reps:.3f}) '
                  f'lift {np.mean(lifts):+.3f}', flush=True)
        out[str(G)] = rows
    return out


def choose_T(peaks, label):
    """Device threshold: the median sub-window peak over faulty windows, so
    counts spread over 0-10 for faulty bearings."""
    return int(np.median(peaks[label != 'normal']))


def release(peaks, label, T, scenario, eps=0.0, rng=None):
    c = counts_at(peaks, T)
    if scenario == 'S3':
        keep = np.ones(len(c), bool)
        y = label.copy()
        logged = None
    else:
        keep = np.ones(len(c), bool)
        if scenario == 'S1':
            keep = ((c >= 3) & (c <= 7)) | (c >= 9)
            if eps > 0:
                b8 = np.flatnonzero(c == 8)
                keep[rng.choice(b8, int(round(eps * len(b8))), replace=False)] = True
        y = np.where(c <= (7 if scenario == 'S1' else 8), 'quiet', label)
        logged = np.where(c >= 9, CODE, 10 * c)
    return keep, y, logged, c


def _lgbm():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                              random_state=SEED, verbose=-1, n_jobs=8)


def sufficiency(X, c, target, groups):
    """T2: LightGBM on (x, g) against LightGBM on g alone, target vs rest,
    under the same load-grouped folds."""
    errs = {}
    for name, XX in (('model', X), ('count', c.reshape(-1, 1))):
        pred = np.zeros(len(target), bool)
        for tr, te in GroupKFold(n_splits=len(np.unique(groups))).split(XX, target, groups):
            if len(np.unique(target[tr])) < 2:
                pred[te] = target[tr][0]
                continue
            pred[te] = _lgbm().fit(XX[tr], target[tr]).predict(XX[te]).astype(bool)
        errs[name] = int((pred != target).sum())
    return {'model_errors': errs['model'], 'count_errors': errs['count'],
            'n': int(len(target)), 'pass': errs['model'] >= errs['count']}


def best_count_rule(c, target):
    best = (-1, None)
    for k in range(1, 11):
        flag = c < k                                         # target = low count
        ba = (flag[target].mean() + (~flag[~target]).mean()) / 2
        if ba > best[0]:
            best = (ba, k)
    return best[1]


def a2(feats, peaks, label, load, rec_id, T, rng):
    X = np.hstack([feats, peaks])
    out = {}
    for sc in ('S1', 'S2', 'S3'):
        keep, y, logged, c = release(peaks, label, T, sc)
        Xk, ck, yk, gk, rk = X[keep], c[keep], y[keep], load[keep], rec_id[keep]
        hist = np.bincount(ck, minlength=11).tolist()
        rec = {'n': int(keep.sum()), 'count_histogram': hist}
        if logged is None:
            rec['T1'] = {'solutions': [], 'pass': False, 'note': 'no logged field'}
            target = yk == 'normal'
        else:
            lk = logged[keep]
            sols = identify_threshold(peaks[keep], lk, CODE, scale=10,
                                      grid=np.arange(1, int(peaks.max()) + 2))
            rec['T1'] = {'solutions': sols[:5], 'n_solutions': len(sols),
                         'pass': len(sols) == 1 and sols[0] == T}
            k_lo, k_hi = identify_k(peaks[keep], lk, CODE, T)
            rec['k_range'] = [k_lo, k_hi]
            target = yk == 'quiet'
        rec['T2'] = sufficiency(Xk, ck, target, gk)
        # The boundary bin is fixed at 8 by the code (counts of 9 and 10 read
        # 90) in every release, S3 included.
        floor = 3 if sc == 'S1' else None
        rec['T3'] = depletion_test(ck, rk, boundary=8, floor=floor, rng=rng)
        rec['T3']['pass'] = rec['T3']['boundary']['pass']
        out[sc] = rec
        print(f'  {sc}: n {rec["n"]}  T1 {rec["T1"]["pass"]}  T2 model '
              f'{rec["T2"]["model_errors"]} vs count {rec["T2"]["count_errors"]}  '
              f'T3 ratio {rec["T3"]["boundary"]["ratio"]} upper '
              f'{rec["T3"]["boundary"]["ratio_upper95"]:.2f}', flush=True)
    return out


def t3_power(peaks, label, rec_id, T, rng, eps_list=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0),
             reps=200):
    rows = []
    for eps in eps_list:
        rej = 0
        ratios = []
        for _ in range(reps):
            keep, _, _, c = release(peaks, label, T, 'S1', eps=eps, rng=rng)
            t = depletion_test(c[keep], rec_id[keep], boundary=8, n_boot=500,
                               rng=rng)['boundary']
            rej += t['pass']
            ratios.append(t['ratio'])
        rows.append({'epsilon': eps, 'reject_rate': rej / reps,
                     'mean_ratio': float(np.mean(ratios))})
        print(f'  T3 power eps={eps}: reject {rej / reps:.3f}', flush=True)
    return rows


def t3_size(peaks, label, rec_id, rng, qs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            reps=200):
    """Unselected releases: every interior bin at six thresholds. Each
    replicate resamples recording files with replacement, then applies T3
    (500 inner resamples). Also counts how often the Poisson p would reject."""
    faulty = peaks[label != 'normal']
    files = np.unique(rec_id)
    idx = {f: np.flatnonzero(rec_id == f) for f in files}
    rows = []
    for q in qs:
        T = int(np.quantile(faulty, q))
        c = counts_at(peaks, T)
        for b in range(1, 10):
            rej = rej_p = 0
            for _ in range(reps):
                pick = rng.choice(files, len(files))
                ii = np.concatenate([idx[f] for f in pick])
                rid = np.concatenate([[f'{k}'] * len(idx[f]) for k, f in enumerate(pick)])
                t = depletion_test(c[ii], rid, boundary=b, n_boot=500, rng=rng)['boundary']
                rej += t['pass']
                rej_p += (t['p_value'] is not None and t['p_value'] < 0.05)
            rows.append({'quantile': q, 'T': T, 'bin': b, 'reject_rate': rej / reps,
                         'poisson_reject_rate': rej_p / reps})
        worst = max(r['reject_rate'] for r in rows if r['quantile'] == q)
        print(f'  T3 size q={q} T={T}: worst-bin reject {worst:.3f}', flush=True)
    return rows


def a3_cwru(feats, peaks, label, bearing):
    """Fault-type typing with each fault size held out (step A3)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    X = np.hstack([feats, peaks])
    f = label != 'normal'
    size = np.array([b[-3:] for b in bearing])
    out = {}
    idx = np.flatnonzero(f)
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=label[idx])
    p = _lgbm().fit(X[tr], label[tr]).predict(X[te])
    out['random_windows'] = {'macro_f1': float(f1_score(label[te], p, average='macro')),
                             'balanced_acc': float(balanced_accuracy_score(label[te], p)),
                             'reference': 1 / 3}
    for s_ in ('007', '014', '021'):
        te = idx[size[idx] == s_]
        tr = idx[size[idx] != s_]
        p = _lgbm().fit(X[tr], label[tr]).predict(X[te])
        out[f'held_out_{s_}'] = {
            'macro_f1': float(f1_score(label[te], p, average='macro')),
            'balanced_acc': float(balanced_accuracy_score(label[te], p)),
            'reference': 1 / 3}
        print(f'  A3 CWRU held-out size {s_}: macro-F1 {out[f"held_out_{s_}"]["macro_f1"]:.3f}',
              flush=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    feats, peaks, label, bearing, load, rec_id = load_cwru()
    T = choose_T(peaks, label)
    c = counts_at(peaks, T)
    out = {'n_windows': int(len(label)), 'n_bearings': int(len(np.unique(bearing))),
           'class_counts': {k: int(v) for k, v in zip(*np.unique(label, return_counts=True))},
           'T': T, 'count_histogram_all': np.bincount(c, minlength=11).tolist()}
    print(f'{len(label)} windows, T={T}, counts {out["count_histogram_all"]}', flush=True)
    out['a1_real'] = a1_real(label, bearing, load, rng)
    print('A1 real:', {k: round(v['lift'], 3) for k, v in out['a1_real'].items()}, flush=True)
    out['a2'] = a2(feats, peaks, label, load, rec_id, T, rng)
    out['t3_power'] = t3_power(peaks, label, rec_id, T, rng)
    out['t3_size'] = t3_size(peaks, label, rec_id, rng)
    out['a3_cwru'] = a3_cwru(feats, peaks, label, bearing)
    out['a1_simulation'] = a1_simulation(rng)
    with open(os.path.join(RES, 'audit_semisynthetic_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
