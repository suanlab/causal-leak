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
T1 (identity), T2 (sufficiency: LightGBM vs the count rule for target vs rest
under load-grouped 4-fold; bearing grouping would leave the single normal
bearing out of training whenever it is tested) and T3 (depletion at the boundary bin and the
inclusion floor) are applied to each. T3 power: S1 with a fraction epsilon of
the boundary-bin windows put back.

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
    feats, peaks, label, bearing, load = [], [], [], [], []
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
    return (np.vstack(feats), np.vstack(peaks), np.array(label),
            np.array(bearing), np.array(load))


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
            rej, lifts = 0, []
            for _ in range(reps):
                home = rng.choice(K, G, p=WATER_SHARES)
                unit = np.repeat(np.arange(G), m)
                y = np.where(rng.random(G * m) < lam, home[unit],
                             rng.choice(K, G * m, p=WATER_SHARES))
                te = rng.random(G * m) < 0.3
                r = lift_test(unit[~te], y[~te], unit[te], y[te], n_perm=n_perm, rng=rng)
                rej += r['p_value'] <= 0.05
                lifts.append(r['lift'])
            rows.append({'lambda': lam, 'reject_rate': rej / reps,
                         'mean_lift': float(np.mean(lifts))})
            print(f'  A1 sim G={G} lambda={lam}: reject {rej / reps:.3f} '
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


def sufficiency(X, c, target, groups, rule_k=None):
    """LightGBM target-vs-rest under load-grouped folds, against a count rule
    (target iff count < k). With rule_k None, k is chosen on each training
    fold by balanced accuracy."""
    pred = np.zeros(len(target), bool)
    rule = np.zeros(len(target), bool)
    for tr, te in GroupKFold(n_splits=len(np.unique(groups))).split(X, target, groups):
        k = rule_k if rule_k is not None else best_count_rule(c[tr], target[tr])
        rule[te] = c[te] < k
        if len(np.unique(target[tr])) < 2:
            pred[te] = target[tr][0]
            continue
        clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                 random_state=SEED, verbose=-1, n_jobs=8)
        clf.fit(X[tr], target[tr])
        pred[te] = clf.predict(X[te]).astype(bool)
    return {'model_errors': int((pred != target).sum()),
            'rule_errors': int((rule != target).sum()),
            'n': int(len(target))}


def best_count_rule(c, target):
    best = (-1, None)
    for k in range(1, 11):
        flag = c < k                                         # target = low count
        ba = (flag[target].mean() + (~flag[~target]).mean()) / 2
        if ba > best[0]:
            best = (ba, k)
    return best[1]


def a2(feats, peaks, label, load, T, rng):
    X = np.hstack([feats, peaks])
    out = {}
    for sc in ('S1', 'S2', 'S3'):
        keep, y, logged, c = release(peaks, label, T, sc)
        Xk, ck, yk, gk = X[keep], c[keep], y[keep], load[keep]
        hist = np.bincount(ck, minlength=11).tolist()
        rec = {'n': int(keep.sum()), 'count_histogram': hist}
        if logged is None:
            rec['T1'] = {'solutions': [], 'pass': False, 'note': 'no logged field'}
            target = yk == 'normal'
            k = best_count_rule(ck, target)
            rec['rule_k'] = k
            rule_k = None
        else:
            lk = logged[keep]
            sols = identify_threshold(peaks[keep], lk, CODE, scale=10,
                                      grid=np.arange(1, int(peaks.max()) + 2))
            rec['T1'] = {'solutions': sols[:5], 'n_solutions': len(sols),
                         'pass': len(sols) == 1 and sols[0] == T}
            k_lo, k_hi = identify_k(peaks[keep], lk, CODE, T)
            rec['k_range'] = [k_lo, k_hi]
            k = k_lo
            target = yk == 'quiet'
            rule_k = k
        rec['T2'] = sufficiency(Xk, ck, target, gk, rule_k)
        rec['T2']['pass'] = rec['T2']['model_errors'] >= rec['T2']['rule_errors']
        floor = 3 if sc == 'S1' else None
        bnd = 8 if sc != 'S3' else min(max(k, 1), 9)
        rec['T3'] = depletion_test(hist, boundary=bnd, floor=floor)
        rec['T3']['pass'] = bool(rec['T3']['boundary']['p_value'] is not None
                                 and rec['T3']['boundary']['p_value'] < 0.05)
        out[sc] = rec
        print(f'  {sc}: n {rec["n"]}  T1 {rec["T1"]["pass"]}  T2 model '
              f'{rec["T2"]["model_errors"]} vs rule {rec["T2"]["rule_errors"]}  '
              f'T3 p {rec["T3"]["boundary"]["p_value"]}', flush=True)
    return out


def t3_power(peaks, label, T, rng, eps_list=(0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
             reps=200):
    rows = []
    for eps in eps_list:
        rej = 0
        ratios = []
        for _ in range(reps):
            keep, _, _, c = release(peaks, label, T, 'S1', eps=eps, rng=rng)
            hist = np.bincount(c[keep], minlength=11)
            t = depletion_test(hist, boundary=8)['boundary']
            rej += t['p_value'] < 0.05
            ratios.append(t['ratio'])
        rows.append({'epsilon': eps, 'reject_rate': rej / reps,
                     'mean_ratio': float(np.mean(ratios))})
        print(f'  T3 power eps={eps}: reject {rej / reps:.3f}', flush=True)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    feats, peaks, label, bearing, load = load_cwru()
    T = choose_T(peaks, label)
    c = counts_at(peaks, T)
    out = {'n_windows': int(len(label)), 'n_bearings': int(len(np.unique(bearing))),
           'class_counts': {k: int(v) for k, v in zip(*np.unique(label, return_counts=True))},
           'T': T, 'count_histogram_all': np.bincount(c, minlength=11).tolist()}
    print(f'{len(label)} windows, T={T}, counts {out["count_histogram_all"]}', flush=True)
    out['a1_real'] = a1_real(label, bearing, load, rng)
    print('A1 real:', {k: round(v['lift'], 3) for k, v in out['a1_real'].items()}, flush=True)
    out['a2'] = a2(feats, peaks, label, load, T, rng)
    out['t3_power'] = t3_power(peaks, label, T, rng)
    out['a1_simulation'] = a1_simulation(rng)
    with open(os.path.join(RES, 'audit_semisynthetic_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
