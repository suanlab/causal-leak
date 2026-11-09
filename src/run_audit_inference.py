"""Inference and checks requested in the DMKD round-1 review.

Every part reads stored results or refits small tree models; nothing here
retrains a network.

1. region_level   Typing at the level of the held-out region: per-region
                  alarm-class macro-F1 minus 1/K and balanced accuracy minus
                  1/K (loro_results.json), sign counts, an interval that
                  resamples regions, and the minimum detectable excess at 80 %
                  power. Counties likewise (leak_f1_results.json).
2. canonical_ci   Sensor-resampled intervals for the four tree models on the
                  canonical region (43), from the per-recording scores in
                  results/dispatch_op/, with the number of sites stated.
3. calibration_dispersion
                  For step A4: with the threshold calibrated on the pooled
                  training OOF scores, the miss rate each training region
                  achieves on its own OOF scores. This spread is visible
                  before the test region is touched.
4. sufficiency    For step A2 (test T2): does a learned normal-vs-rest model
                  add anything to the device count? LightGBM on the 534
                  signal columns, with and without the count as a feature,
                  under leave-one-region-out, against the count rule.
5. variance       Site against class in the alarm recordings' log spectrum:
                  share of variance of the top principal components explained
                  by site, by class, and by class within site.
6. margin         Typing by amplitude margin: held-out alarm recordings binned
                  by the median of their ten check amplitudes (quartiles).
7. lift           For step A1: lift of the sensor lookup over the largest
                  class share on the benchmark's own validation split, against
                  a null that permutes the training sensors' labels.
8. site_tests     Paired false-dispatch comparisons of Appendix A tested by
                  normal-bearing site (exact sign test) with Holm adjustment
                  over the same 18 comparisons, from the stored dispatch
                  results.

Usage: run_audit_inference.py [section ...]  (default: all; named sections
are recomputed and merged into the existing result file).

Output: results/audit_inference_results.json
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from analyze_dispatch import (LEAK, MODELS, NORMAL, NUISANCE,  # noqa: E402
                              REGIONS as DISPATCH_REGIONS, calibrate,
                              dispatch_score, load)
from audit_tests import lift_test  # noqa: E402
from data import HZ_COLS, get_xy, load_all  # noqa: E402
from summarize_leak_f1 import leak_f1, per_class_from_cm, present_from_cm  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
ALARM = (0, 1, 2, 3)
SEED = 42
N_BOOT = 10000
AMP_COLS = [f'MAX{i}' for i in range(1, 20, 2)]
DEVICE_T = 125


def L(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def balanced_acc_from_cm(cm):
    """Mean recall over the alarm classes present (a prediction of normal
    counts as an error); chance is 1/K for any input-blind guesser."""
    cm = np.asarray(cm, dtype=float)
    pres = present_from_cm(cm)
    rec = [cm[c, c] / cm[c].sum() for c in pres]
    return float(np.mean(rec)) if rec else None


def mde(sd, n, power=0.8, alpha=0.05):
    if n < 2:
        return None
    t_a = stats.t.ppf(1 - alpha / 2, n - 1)
    t_b = stats.t.ppf(power, n - 1)
    return float((t_a + t_b) * sd / np.sqrt(n))


def summarise(excess, rng):
    e = np.asarray(excess, dtype=float)
    boots = [rng.choice(e, len(e)).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    t = stats.ttest_1samp(e, 0.0) if len(e) > 1 else None
    n = len(e)
    t_ci = (None, None)
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * e.std(ddof=1) / np.sqrt(n)
        t_ci = (float(e.mean() - h), float(e.mean() + h))
    # Exact sign-flip test of mean zero (all 2^n sign patterns; n <= 16 here).
    flip_p = None
    if 1 < n <= 16:
        signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
        means = np.abs((signs * np.abs(e)).mean(1))
        flip_p = float((means >= abs(e.mean()) - 1e-12).mean())
    return {'n': int(len(e)), 'mean': float(e.mean()),
            't_ci': list(t_ci), 'sign_flip_p': flip_p,
            'sd': float(e.std(ddof=1)) if len(e) > 1 else None,
            'n_above': int((e > 0).sum()), 'n_below': int((e < 0).sum()),
            'resample_units_ci': [float(lo), float(hi)],
            'sign_test_p': float(stats.binomtest(int((e > 0).sum()), len(e)).pvalue),
            't_test_p': float(t.pvalue) if t is not None else None,
            'mde80': mde(e.std(ddof=1), len(e)) if len(e) > 1 else None}


def guess_f1(p, q):
    """Expected macro-F1 (large-sample) of an input-blind guess drawing class
    k with probability q_k, on test shares p, over classes with p_k > 0."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float(np.mean(2 * p[m] * q[m] / (p[m] + q[m])))


def region_level(rng):
    out = {}
    for run in L('loro_results.json'):
        rows = []
        tot = sum(np.asarray(f['cm'])[:4].sum(1) for f in run['per_region'])
        for f in run['per_region']:
            pres = present_from_cm(f['cm'])
            if len(pres) < 2:
                continue
            K = len(pres)
            a = leak_f1(per_class_from_cm(f['cm']), pres)
            b = balanced_acc_from_cm(f['cm'])
            te_n = np.asarray(f['cm'])[:4].sum(1)
            p_te = te_n / te_n.sum()
            tr_n = tot - te_n
            p_tr = tr_n / tr_n.sum()
            rows.append({'region': f['region'], 'K': K,
                         'oracle_reference': 1 / K,
                         'uniform_guess': guess_f1(p_te, np.where(p_te > 0, 1 / K, 0)),
                         'training_prior_guess': guess_f1(p_te, p_tr),
                         'alarm_rows': int(np.asarray(f['cm'])[:4].sum()),
                         'alarm_f1': a, 'alarm_f1_excess': a - 1 / K,
                         'balanced_acc': b, 'balanced_acc_excess': b - 1 / K})
        res = {'per_region': rows}
        k4 = [r for r in rows if r['K'] == 4]
        res['k4_references'] = {k: float(np.mean([r[k] for r in k4])) for k in
                                ('oracle_reference', 'uniform_guess', 'training_prior_guess',
                                 'alarm_f1')}
        for name, sel in (('k2', lambda r: True), ('k4', lambda r: r['K'] == 4)):
            rs = [r for r in rows if sel(r)]
            res[name] = {m: summarise([r[f'{m}_excess'] for r in rs], rng)
                         for m in ('alarm_f1', 'balanced_acc')}
        out[run['feat_mode']] = res
    county = [f for f in L('leak_f1_results.json')['county']['folds']
              if f.get('f1_macro_leak') is not None]
    ex = [f['f1_macro_leak'] - f['chance'] for f in county]
    out['county'] = {'n_counties': len(county),
                     'mean_alarm_f1': float(np.mean([f['f1_macro_leak'] for f in county])),
                     'mean_chance': float(np.mean([f['chance'] for f in county])),
                     'alarm_f1': summarise(ex, rng)}
    return out


def canonical_ci(rng):
    d = load(43)
    y, sid, site = d['te_y'], d['te_sid'], d['te_site']
    nn = y != NORMAL
    groups = {g: np.where(sid == g)[0] for g in np.unique(sid)}
    keys = list(groups)
    out = {'n_sensors': len(keys), 'n_sites': int(len(np.unique(site))),
           'alarm_sensors_at_largest_site': None, 'models': {}}
    alarm_sid = np.unique(sid[nn])
    s_site = {s: site[sid == s][0] for s in alarm_sid}
    vals, counts = np.unique(list(s_site.values()), return_counts=True)
    out['alarm_sensors'] = int(len(alarm_sid))
    out['alarm_sensors_at_largest_site'] = int(counts.max())

    def stat(idx, pred):
        yy, pp = y[idx], pred[idx]
        pres = [c for c in ALARM if (yy == c).any()]
        f = f1_score(yy, pp, labels=pres, average='macro', zero_division=0)
        return f, 1 / len(pres)

    for m in MODELS:
        pred = d[f'te_{m}'].argmax(1)
        f, ch = stat(np.arange(len(y)), pred)
        boots = []
        for _ in range(2000):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([groups[keys[i]] for i in pick])
            boots.append(stat(idx, pred)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out['models'][m] = {'alarm_f1': float(f), 'chance': ch,
                            'sensor_ci': [float(lo), float(hi)]}
    return out


def calibration_dispersion():
    out = {}
    for r in DISPATCH_REGIONS:
        d = load(r)
        rec = {}
        for m in ('tree_only_soft_avg', 'lightgbm'):
            oof = d[f'oof_{m}']
            ok = ~np.isnan(oof[:, 0])
            rec[m] = {}
            for t in (0.05, 0.10, 0.20):
                thr = calibrate(oof, d['tr_y'], t)
                s = dispatch_score(oof)
                per = {}
                for g in np.unique(d['tr_region']):
                    mk = ok & (d['tr_region'] == g) & np.isin(d['tr_y'], LEAK)
                    if mk.sum() >= 50:
                        per[int(g)] = float((s[mk] < thr).mean())
                te_leak = np.isin(d['te_y'], LEAK)
                te_miss = float((dispatch_score(d[f'te_{m}'])[te_leak] < thr).mean())
                v = list(per.values())
                rec[m][f'miss{int(t * 100):02d}'] = {
                    'per_training_region': per, 'min': min(v), 'max': max(v),
                    'test_region_miss': te_miss}
        out[str(r)] = rec
    return out


def sufficiency(df):
    X, y, cols = get_xy(df, 'signal')
    amp = df[AMP_COLS].to_numpy()
    count = (amp >= DEVICE_T).sum(1)
    is_norm = (y == NORMAL).astype(int)
    rule_flag = count >= 8                      # non-normal per the device rule
    reg = df['region_lvl1'].to_numpy()
    Xc = np.column_stack([X, count])
    params = dict(n_estimators=300, num_leaves=63, learning_rate=0.05,
                  random_state=SEED, verbose=-1, n_jobs=-1)
    res = {}
    for name, XX in (('signal', X), ('signal_plus_count', Xc),
                     ('count_only', count.reshape(-1, 1))):
        pred = np.zeros(len(y), dtype=int)
        score = np.zeros(len(y))
        for r in np.unique(reg):
            tr, te = reg != r, reg == r
            clf = lgb.LGBMClassifier(**params).fit(XX[tr], is_norm[tr])
            score[te] = clf.predict_proba(XX[te])[:, 1]
            pred[te] = (score[te] >= 0.5).astype(int)
        err = pred != is_norm
        both = np.unique(reg[is_norm == 1])
        m2 = np.isin(reg, both)
        res[name] = {
            'errors': int(err.sum()),
            'false_normal': int(((pred == 1) & (is_norm == 0)).sum()),
            'false_alarm': int(((pred == 0) & (is_norm == 1)).sum()),
            'errors_on_rule_misses': int(err[~rule_flag & (is_norm == 0)].sum()),
            'auc_regions_with_both': float(roc_auc_score(is_norm[m2], score[m2]))}
    rule_err = rule_flag == (is_norm == 1)
    res['count_rule'] = {
        'errors': int(rule_err.sum()),
        'auc_regions_with_both': float(roc_auc_score(
            is_norm[np.isin(reg, np.unique(reg[is_norm == 1]))],
            -count[np.isin(reg, np.unique(reg[is_norm == 1]))]))}
    return res


def variance(df):
    a = (df['label'] != NORMAL).to_numpy()
    X = np.log1p(df.loc[a, HZ_COLS].to_numpy(dtype=float))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pca = PCA(n_components=10, random_state=SEED).fit(X)
    Z = pca.transform(X)
    site = df.loc[a, 'site'].astype(str).to_numpy()
    sensor = df.loc[a, 'sid'].astype(str).to_numpy()
    cls = df.loc[a, 'label'].to_numpy()
    cell = np.char.add(np.char.add(site, '|'), cls.astype(str))

    def eta2(z, g):
        mu = z.mean()
        ss_t = ((z - mu) ** 2).sum()
        ss_b = sum(len(z[g == k]) * (z[g == k].mean() - mu) ** 2 for k in np.unique(g))
        return ss_b / ss_t

    comps = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        e_site, e_cls, e_cell = eta2(z, site), eta2(z, cls), eta2(z, cell)
        e_sens = eta2(z, sensor)
        comps.append({'explained_variance_ratio': float(pca.explained_variance_ratio_[j]),
                      'site': float(e_site), 'class': float(e_cls),
                      'class_within_site': float(e_cell - e_site),
                      'sensor': float(e_sens),
                      'sensor_beyond_site_class': float(e_sens - e_cell)})
    w = np.array([c['explained_variance_ratio'] for c in comps])
    wavg = {k: float(np.average([c[k] for c in comps], weights=w))
            for k in ('site', 'class', 'class_within_site', 'sensor',
                      'sensor_beyond_site_class')}
    return {'n_rows': int(a.sum()), 'n_sites': int(len(np.unique(site))),
            'components': comps, 'weighted': wavg,
            'total_explained': float(w.sum())}


def margin(df):
    amp = df[AMP_COLS].to_numpy()
    reg = df['region_lvl1'].to_numpy()
    med_all, y_all, p_all = [], [], {m: [] for m in ('tree_only_soft_avg', 'lightgbm')}
    for r in DISPATCH_REGIONS:
        d = load(r)
        rows = np.where(reg == r)[0]
        assert np.array_equal(df['label'].to_numpy()[rows], d['te_y'])
        nn = d['te_y'] != NORMAL
        med_all.append(np.median(amp[rows][nn], 1))
        y_all.append(d['te_y'][nn])
        for m in p_all:
            p_all[m].append(d[f'te_{m}'][nn])
    med = np.concatenate(med_all)
    y = np.concatenate(y_all)
    edges = np.quantile(med, [0, 0.25, 0.5, 0.75, 1.0])
    q = np.clip(np.searchsorted(edges, med, side='right') - 1, 0, 3)
    out = {'edges': edges.tolist(), 'quartiles': []}
    for k in range(4):
        mk = q == k
        rec = {'n': int(mk.sum())}
        for m, ps in p_all.items():
            p = np.concatenate(ps)[mk]
            pres = [c for c in ALARM if (y[mk] == c).any()]
            rec[m] = {
                'alarm_f1': float(f1_score(y[mk], p.argmax(1), labels=pres,
                                           average='macro', zero_division=0)),
                'leak_vs_nuisance_auc': float(roc_auc_score(
                    np.isin(y[mk], LEAK), dispatch_score(p)))}
        out['quartiles'].append(rec)
    return out


def lift(df, rng, n_perm=1000):
    tr, va = df[df['split'] == 'train'], df[df['split'] == 'val']
    return lift_test(tr['sid'], tr['label'], va['sid'], va['label'],
                     n_perm=n_perm, rng=rng)


def site_tests():
    import importlib
    holm = importlib.import_module('analyze_dispatch').holm
    d = L('dispatch_operating_points_results.json')
    rows = []
    for op, lst in d['comparisons'].items():
        for c in lst:
            rows.append({'op': op, 'a': c['a'], 'b': c['b'],
                         'sensors_a': c['false_dispatch_sensors_a'],
                         'sensors_b': c['false_dispatch_sensors_b'],
                         **c['install_sign_test']})
    adj = holm([r['p_value'] for r in rows])
    for r, a in zip(rows, adj):
        r['p_holm'] = float(a)
    return {'n_tests': len(rows), 'n_sites': len(d['comparisons']['argmax'][0]['install_counts']),
            'min_p': min(r['p_value'] for r in rows),
            'min_p_holm': min(r['p_holm'] for r in rows), 'tests': rows}


SECTIONS = ('region_level', 'canonical_ci', 'calibration_dispersion', 'variance',
            'margin', 'sufficiency', 'lift', 'site_tests')


def main():
    rng = np.random.default_rng(SEED)
    df = load_all()
    want = sys.argv[1:] or list(SECTIONS)
    path = os.path.join(RES, 'audit_inference_results.json')
    out = {}
    if sys.argv[1:] and os.path.exists(path):
        out = L('audit_inference_results.json')
    calls = {'region_level': lambda: region_level(rng),
             'canonical_ci': lambda: canonical_ci(rng),
             'calibration_dispersion': calibration_dispersion,
             'variance': lambda: variance(df), 'margin': lambda: margin(df),
             'sufficiency': lambda: sufficiency(df), 'lift': lambda: lift(df, rng),
             'site_tests': site_tests}
    for sec in SECTIONS:
        if sec in want:
            out[sec] = calls[sec]()
            print(f'{sec} done', flush=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    if sys.argv[1:]:
        for sec in want:
            print(sec, json.dumps(out[sec])[:600])
        return

    rl = out['region_level']
    for feat in ('signal', 'full'):
        for k in ('k4', 'k2'):
            a, b = rl[feat][k]['alarm_f1'], rl[feat][k]['balanced_acc']
            print(f'{feat} {k}: alarm-F1 excess {a["mean"]:+.3f} '
                  f'[{a["resample_units_ci"][0]:+.3f}, {a["resample_units_ci"][1]:+.3f}] '
                  f'{a["n_above"]}/{a["n"]} above, MDE {a["mde80"]:.3f}; '
                  f'BA excess {b["mean"]:+.3f} {b["n_above"]}/{b["n"]} above')
    c = rl['county']
    print(f'county: {c["mean_alarm_f1"]:.3f} vs {c["mean_chance"]:.3f}, '
          f'{c["alarm_f1"]["n_above"]}/{c["n_counties"]} above')
    for m, v in out['canonical_ci']['models'].items():
        print(f'region 43 {m}: {v["alarm_f1"]:.3f} sensor CI '
              f'[{v["sensor_ci"][0]:.3f}, {v["sensor_ci"][1]:.3f}]')
    print('variance (weighted over top-10 PCs):', out['variance']['weighted'])
    for k, qq in enumerate(out['margin']['quartiles']):
        print(f'margin Q{k + 1}: n {qq["n"]}  soft avg F1 '
              f'{qq["tree_only_soft_avg"]["alarm_f1"]:.3f} AUC '
              f'{qq["tree_only_soft_avg"]["leak_vs_nuisance_auc"]:.3f}')
    print('sufficiency:', json.dumps(out['sufficiency']))


if __name__ == '__main__':
    main()
