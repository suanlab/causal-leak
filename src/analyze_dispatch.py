"""Score the deployment candidates under an explicit dispatch rule.

Reads the per-recording probabilities written by run_dispatch_operating_points
(results/dispatch_op/region<R>.npz) and reports, per held-out region and pooled:

  typing      accuracy, macro-F1 over the classes present in the region
              (and over all five, for comparison with earlier tables),
              per-class F1, the confusion matrix, and four-class typing among
              the true non-normal recordings only
  dispatch    at the argmax decision and at thresholds calibrated on the
              training pool's region-grouped out-of-fold scores:
                missed leaks          true in/out, not dispatched
                false dispatches      true normal, dispatched
                nuisance dispatches   true noise/other, dispatched
  sensors     normal sensors with >= 1 false dispatch, and in/out sensors
              never dispatched, compared pairwise by exact McNemar and a
              cluster bootstrap (sensor_stats)
  installs    per-installation false-dispatch counts, compared by sign test
  cost        the miss-to-dispatch cost ratio above which one model is cheaper
  diagnostic  normal-vs-rest AUC and false dispatches at matched pooled miss
              counts, with the threshold read off the TEST regions. Those
              thresholds are fitted on the evaluation data, so they show only
              whether a ranking depends on the operating point; they are not a
              protocol-valid comparison.

Output: results/dispatch_operating_points_results.json
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from data import LABEL_NAMES  # noqa: E402
from sensor_stats import cluster_bootstrap, mcnemar_exact, sign_test  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
# DISPATCH_FEATURES=all reads the all-signal run (results/dispatch_op_all/) and
# writes dispatch_operating_points_all_results.json.
FEATURES = os.environ.get('DISPATCH_FEATURES', 'nested')
OP_DIR = os.path.join(RES, 'dispatch_op' if FEATURES == 'nested' else 'dispatch_op_all')
OUT_NAME = ('dispatch_operating_points_results.json' if FEATURES == 'nested'
            else 'dispatch_operating_points_all_results.json')

MODELS = ['xgb_flat', 'xgb_hier', 'tree_only_soft_avg', 'lightgbm']
REGIONS = [27, 29, 43, 44, 45, 46, 47, 48]
FAR_REGIONS = [27, 29, 44, 45]          # >= 100 normal recordings
IN, OUT, NOISE, OTHER, NORMAL = range(5)
LEAK = (IN, OUT)
NUISANCE = (NOISE, OTHER)
MISS_TARGETS = [0.01, 0.05, 0.10, 0.20]
PAIRS = [('tree_only_soft_avg', 'lightgbm'), ('xgb_hier', 'lightgbm'),
         ('xgb_flat', 'lightgbm')]


def load(r):
    # sid is saved as an object array, so the archive needs pickle to load.
    z = np.load(os.path.join(OP_DIR, f'region{r}.npz'), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d['te_sid'] = d['te_sid'].astype(str)
    return d


def dispatch_score(p):
    return p[:, IN] + p[:, OUT]


def typing(y, p):
    pred = p.argmax(1)
    present = sorted(set(y.tolist()))
    cm = confusion_matrix(y, pred, labels=list(range(5)))
    f1c = f1_score(y, pred, labels=list(range(5)), average=None,
                   zero_division=0)
    return {
        'acc': float((pred == y).mean()),
        'f1_macro_present': float(f1_score(y, pred, labels=present,
                                           average='macro', zero_division=0)),
        'f1_macro_all5': float(f1_score(y, pred, average='macro',
                                        zero_division=0)),
        'classes_present': [LABEL_NAMES[c] for c in present],
        'f1_per_class': {LABEL_NAMES[c]: float(f1c[c]) for c in present},
        'cm': cm.tolist(),
        **typing4(y, pred),
    }


def typing4(y, pred):
    """Leak-class macro-F1: mean five-class F1 over the leak classes present.

    The normal/non-normal boundary is reproduced by the device's own trigger
    count (run_device_rule.py), so the normal class's F1 pads a five-class
    average with something no model needs to learn. Averaging the per-class
    F1 of in/out/noise/other only (from the same five-class confusion) keeps a
    normal predicted as a leak type, or a leak predicted as normal, as an
    error, and is computable from any stored per-class F1.
    """
    present = [c for c in (IN, OUT, NOISE, OTHER) if (y == c).any()]
    if not present:
        return {}
    f1c = f1_score(y, pred, labels=present, average=None, zero_division=0)
    m = y != NORMAL
    return {
        'f1_macro_leak': float(f1c.mean()),
        'acc_leak': float((pred[m] == y[m]).mean()),
        'n_leak_rows': int(m.sum()),
    }


def dispatch_counts(y, d, sid, site):
    """Counts and unit-level error sets for one dispatch decision vector."""
    leak, norm, nuis = np.isin(y, LEAK), y == NORMAL, np.isin(y, NUISANCE)
    miss = leak & ~d
    fd = norm & d
    leak_sens = set(sid[leak].tolist())
    hit_sens = set(sid[leak & d].tolist())
    return {
        'n_leak': int(leak.sum()), 'missed': int(miss.sum()),
        'n_normal': int(norm.sum()), 'false_dispatch': int(fd.sum()),
        'n_nuisance': int(nuis.sum()), 'nuisance_dispatch': int((nuis & d).sum()),
        'leak_sensors': len(leak_sens),
        'leak_sensors_never_dispatched': sorted(leak_sens - hit_sens),
        'normal_sensors': sorted(set(sid[norm].tolist())),
        'fd_sensors': Counter(sid[fd].tolist()),
        'fd_by_install': Counter(site[fd].tolist()),
        'normal_installs': sorted(set(site[norm].tolist())),
        'sensor_dispatch': sensor_dispatch(y, d, sid),
        'precision': float(np.isin(y[d], LEAK).mean()) if d.any() else None,
    }


def sensor_dispatch(y, d, sid, ks=(1, 2, 3)):
    """Per-sensor view: crews go to locations, and a persistent leak flags on
    many days. A sensor is dispatched under a k-day rule when at least k of
    its recordings are dispatched. Every sensor carries a single label, so
    leak and nuisance sensors are disjoint.
    """
    out = {}
    for name, cls in (('leak', LEAK), ('nuisance', NUISANCE), ('normal', (NORMAL,))):
        m = np.isin(y, cls)
        if not m.any():
            continue
        sids, inv = np.unique(sid[m], return_inverse=True)
        hits = np.bincount(inv, weights=d[m].astype(float), minlength=len(sids))
        out[name] = {'sensors': int(len(sids)),
                     **{f'k{k}': int((hits >= k).sum()) for k in ks}}
    return out


def calibrate(oof_p, tr_y, target):
    """Threshold on s hitting `target` miss rate among training in/out OOF."""
    s = dispatch_score(oof_p)[np.isin(tr_y, LEAK)]
    s = s[~np.isnan(s)]
    return float(np.quantile(s, target))


def strip(c):
    """JSON-friendly copy of dispatch_counts output."""
    out = {k: v for k, v in c.items()
           if k not in ('fd_sensors', 'fd_by_install')}
    out['fd_sensors'] = dict(c['fd_sensors'])
    out['fd_by_install'] = {str(k): v for k, v in c['fd_by_install'].items()}
    out['leak_sensors_never_dispatched'] = len(c['leak_sensors_never_dispatched'])
    out['normal_sensors'] = len(c['normal_sensors'])
    out['normal_installs'] = [int(i) for i in c['normal_installs']]
    return out


def pool(per_region, regions, key):
    tot = Counter()
    for r in regions:
        for k in ('n_leak', 'missed', 'n_normal', 'false_dispatch',
                  'n_nuisance', 'nuisance_dispatch', 'leak_sensors'):
            tot[k] += per_region[r][key][k]
    return dict(tot)


def pool_sensors(per_region, regions, key):
    tot = {}
    for r in regions:
        for name, v in per_region[r][key]['sensor_dispatch'].items():
            t = tot.setdefault(name, {})
            for k, n in v.items():
                t[k] = t.get(k, 0) + n
    return tot


def compare(per_region, op, a, b):
    """Sensor- and installation-level paired tests of model a against b."""
    fa, fb, normal_units, la, lb = Counter(), Counter(), [], set(), set()
    ia, ib, installs = Counter(), Counter(), set()
    for r in FAR_REGIONS:
        ca, cb = per_region[r][op][a], per_region[r][op][b]
        fa.update({(r, s): n for s, n in ca['fd_sensors'].items()})
        fb.update({(r, s): n for s, n in cb['fd_sensors'].items()})
        normal_units += [(r, s) for s in ca['normal_sensors']]
        ia.update({(r, i): n for i, n in ca['fd_by_install'].items()})
        ib.update({(r, i): n for i, n in cb['fd_by_install'].items()})
        installs |= {(r, i) for i in ca['normal_installs']}
    for r in REGIONS:
        ca, cb = per_region[r][op][a], per_region[r][op][b]
        la |= {(r, s) for s in ca['leak_sensors_never_dispatched']}
        lb |= {(r, s) for s in cb['leak_sensors_never_dispatched']}
    inst = sorted(installs)
    return {
        'a': a, 'b': b,
        'false_dispatch_sensors_a': len(fa), 'false_dispatch_sensors_b': len(fb),
        'mcnemar_false_dispatch': mcnemar_exact(fa, fb),
        'bootstrap_false_dispatch': cluster_bootstrap(fa, fb, normal_units),
        'install_sign_test': sign_test([(ia.get(u, 0), ib.get(u, 0))
                                        for u in inst]),
        'install_counts': [[f'{u[0]}:{u[1]}', ia.get(u, 0), ib.get(u, 0)]
                           for u in inst],
        'never_dispatched_leak_sensors_a': len(la),
        'never_dispatched_leak_sensors_b': len(lb),
        'mcnemar_never_dispatched': mcnemar_exact(la, lb),
    }


def break_even(pa, pb):
    """Cost ratio c_miss / c_dispatch at which a and b cost the same.

    cost = ratio * missed + (false + nuisance dispatches). Returns the ratio
    and which model is cheaper above it, or None when one dominates.
    """
    dm = pa['missed'] - pb['missed']
    dd = ((pa['false_dispatch'] + pa['nuisance_dispatch'])
          - (pb['false_dispatch'] + pb['nuisance_dispatch']))
    if dm == 0 and dd == 0:
        return {'ratio': None, 'dominates': 'tie', 'd_missed': 0,
                'd_dispatches': 0}
    if dm == 0 or dd == 0 or np.sign(dm) == np.sign(dd):
        better = 'a' if (dm <= 0 and dd <= 0) else 'b'
        return {'ratio': None, 'dominates': better, 'd_missed': int(dm),
                'd_dispatches': int(dd)}
    ratio = -dd / dm
    return {'ratio': float(ratio),
            'cheaper_above': 'a' if dm < 0 else 'b',
            'd_missed': int(dm), 'd_dispatches': int(dd)}


def matched_test_diagnostic(data, ks=(0, 3, 10, 30, 100)):
    """False dispatches at pooled test miss counts -- test-fitted, diagnostic."""
    out = {}
    for m in MODELS:
        s_leak, s_norm = [], []
        for r in FAR_REGIONS:
            d = data[r]
            s = dispatch_score(d[f'te_{m}'])
            s_leak.append(s[np.isin(d['te_y'], LEAK)])
            s_norm.append(s[d['te_y'] == NORMAL])
        s_leak = np.sort(np.concatenate(s_leak))
        s_norm = np.concatenate(s_norm)
        row = {}
        for k in ks:
            t = s_leak[k] if k < len(s_leak) else np.inf   # k leaks below t
            row[str(k)] = int((s_norm >= t).sum())
        out[m] = row
    return out


def curves(data, n=200):
    """Miss / false / nuisance dispatch as the threshold on s sweeps, pooled.

    Plotted as the trade-off each model offers; points read off it are fitted
    on the test regions and are not used to compare models.
    """
    thr = np.concatenate([[0.0], np.logspace(-6, 0, n)])
    out = {}
    for scope, regs in (('far_regions', FAR_REGIONS), ('all_regions', REGIONS)):
        out[scope] = {}
        for m in MODELS:
            ys = np.concatenate([data[r]['te_y'] for r in regs])
            ss = np.concatenate([dispatch_score(data[r][f'te_{m}']) for r in regs])
            leak, norm = np.isin(ys, LEAK), ys == NORMAL
            nuis = np.isin(ys, NUISANCE)
            d = ss[None, :] >= thr[:, None]
            out[scope][m] = {
                'threshold': thr.tolist(),
                'missed': (leak[None, :] & ~d).sum(1).tolist(),
                'false_dispatch': (norm[None, :] & d).sum(1).tolist(),
                'nuisance_dispatch': (nuis[None, :] & d).sum(1).tolist(),
                'n_leak': int(leak.sum()), 'n_normal': int(norm.sum()),
                'n_nuisance': int(nuis.sum())}
    return out


def sensor_level_auc(data):
    """Leak-vs-nuisance and in-vs-out AUC with each sensor's scores averaged
    over its recordings, as a utility acting per location would. Sensors are
    keyed by (region, sid): two identifiers occur in two regions."""
    out = {}
    for m in MODELS:
        keys, ys, s_disp, s_in = [], [], [], []
        for r in REGIONS:
            d = data[r]
            nn = d['te_y'] != NORMAL
            p = d[f'te_{m}'][nn]
            for sid in np.unique(d['te_sid'][nn]):
                k = d['te_sid'][nn] == sid
                keys.append((r, sid))
                ys.append(int(d['te_y'][nn][k][0]))
                s_disp.append(float(dispatch_score(p[k]).mean()))
                s_in.append(float((p[k, IN] / (p[k, IN] + p[k, OUT] + 1e-12)).mean()))
        ys, s_disp, s_in = map(np.asarray, (ys, s_disp, s_in))
        lk = np.isin(ys, LEAK)
        out[m] = {'n_sensors': int(len(ys)), 'n_leak_sensors': int(lk.sum()),
                  'auc_leak_vs_nuisance': float(roc_auc_score(lk, s_disp)),
                  'auc_in_vs_out': float(roc_auc_score(ys[lk] == IN, s_in[lk]))}
    return out


def typing_excess(data, regions, unit, n_boot=2000, seed=42):
    """Leak-row-weighted mean of (leak-class macro-F1 - 1/K) over `regions`,
    with a percentile interval from resampling `unit` clusters ('sensor' or
    'site') within each region."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in ('tree_only_soft_avg', 'lightgbm'):
        per = []
        for r in regions:
            d = data[r]
            nn = d['te_y'] != NORMAL
            y, pred = d['te_y'], d[f'te_{m}'].argmax(1)
            u = d['te_sid'] if unit == 'sensor' else d['te_site'].astype(str)
            per.append((y, pred, u, nn))

        def stat(idx_list):
            num = den = 0.0
            for (y, pred, u, nn), idx in zip(per, idx_list):
                yy, pp = y[idx], pred[idx]
                pres = [c for c in (IN, OUT, NOISE, OTHER) if (yy == c).any()]
                f = f1_score(yy, pp, labels=pres, average='macro', zero_division=0)
                w = int(np.isin(yy, LEAK + NUISANCE).sum())
                num += w * (f - 1 / len(pres))
                den += w
            return num / den

        obs = stat([np.arange(len(y)) for y, _, _, _ in per])
        groups = [{g: np.where(u == g)[0] for g in np.unique(u)} for _, _, u, _ in per]
        boots = []
        for _ in range(n_boot):
            idx_list = []
            for g in groups:
                keys = list(g)
                pick = rng.integers(0, len(keys), len(keys))
                idx_list.append(np.concatenate([g[keys[i]] for i in pick]))
            boots.append(stat(idx_list))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[m] = {'excess': float(obs), 'lo': float(lo), 'hi': float(hi),
                  'units_per_region': [len(g) for g in groups]}
    return out


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p in the input order."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    data = {r: load(r) for r in REGIONS}
    per_region, thresholds = {}, {}
    for r in REGIONS:
        d = data[r]
        y, sid, site = d['te_y'], d['te_sid'], d['te_site']
        rec = {'typing': {}, 'auc_normal_vs_rest': {}, 'auc_leak_vs_nuisance': {},
               'auc_in_vs_out': {}}
        thresholds[r] = {}
        # LightGBM's argmax miss rate on the training OOF fixes one more
        # target, so the others can be compared at LightGBM's own operating
        # point without looking at the test region.
        lg_oof = d['oof_lightgbm']
        lk = np.isin(d['tr_y'], LEAK) & ~np.isnan(lg_oof[:, 0])
        lg_miss = float((~np.isin(lg_oof[lk].argmax(1), LEAK)).mean())
        targets = {f'miss{int(t * 100):02d}': t for t in MISS_TARGETS}
        targets['match_lgbm_argmax'] = lg_miss
        ops = ['argmax'] + list(targets)
        for op in ops:
            rec[op] = {}
        for m in MODELS:
            p = d[f'te_{m}']
            rec['typing'][m] = typing(y, p)
            if (y == NORMAL).any() and (y != NORMAL).any():
                rec['auc_normal_vs_rest'][m] = float(
                    roc_auc_score(y == NORMAL, p[:, NORMAL]))
            # Once the device has triggered, the dispatch question is in/out
            # against noise/other; this AUC needs no threshold at all.
            nn = y != NORMAL
            if np.isin(y[nn], LEAK).any() and np.isin(y[nn], NUISANCE).any():
                rec['auc_leak_vs_nuisance'][m] = float(roc_auc_score(
                    np.isin(y[nn], LEAK), dispatch_score(p[nn])))
            # Indoor against outdoor among true leaks: what crew preparation
            # would need. Score p(in) / (p(in) + p(out)), threshold-free.
            lk = np.isin(y, LEAK)
            if (y == IN).any() and (y == OUT).any():
                rec['auc_in_vs_out'][m] = float(roc_auc_score(
                    y[lk] == IN, p[lk, IN] / (p[lk, IN] + p[lk, OUT] + 1e-12)))
            rec['argmax'][m] = dispatch_counts(
                y, np.isin(p.argmax(1), LEAK), sid, site)
            s = dispatch_score(p)
            thresholds[r][m] = {}
            for name, t in targets.items():
                thr = calibrate(d[f'oof_{m}'], d['tr_y'], t)
                thresholds[r][m][name] = thr
                rec[name][m] = dispatch_counts(y, s >= thr, sid, site)
        rec['lgbm_oof_argmax_miss'] = lg_miss
        per_region[r] = rec

    ops = ['argmax'] + [f'miss{int(t * 100):02d}' for t in MISS_TARGETS] \
        + ['match_lgbm_argmax']
    pooled, comparisons, costs = {}, {}, {}
    for op in ops:
        pooled[op] = {m: {'far_regions': pool({r: per_region[r][op] for r in REGIONS},
                                              FAR_REGIONS, m),
                          'all_regions': pool({r: per_region[r][op] for r in REGIONS},
                                              REGIONS, m),
                          'sensors_all_regions': pool_sensors(
                              {r: per_region[r][op] for r in REGIONS}, REGIONS, m)}
                      for m in MODELS}
        comparisons[op] = [compare(per_region, op, a, b) for a, b in PAIRS]
        costs[op] = {f'{a}_vs_{b}': {
            sc: break_even(pooled[op][a][sc], pooled[op][b][sc])
            for sc in ('far_regions', 'all_regions')} for a, b in PAIRS}

    pooled_auc, pooled_io = {}, {}
    for m in MODELS:
        ys = np.concatenate([data[r]['te_y'] for r in REGIONS])
        ps = np.concatenate([data[r][f'te_{m}'] for r in REGIONS])
        lk = np.isin(ys, LEAK)
        pooled_io[m] = float(roc_auc_score(
            ys[lk] == IN, ps[lk, IN] / (ps[lk, IN] + ps[lk, OUT] + 1e-12)))
        ys = np.concatenate([data[r]['te_y'] for r in REGIONS])
        ss = np.concatenate([dispatch_score(data[r][f'te_{m}']) for r in REGIONS])
        nn = ys != NORMAL
        pooled_auc[m] = float(roc_auc_score(np.isin(ys[nn], LEAK), ss[nn]))

    # Every normal false-dispatch McNemar test, Holm-adjusted together.
    tests = [(op, c['a'], c['b'], c['mcnemar_false_dispatch']['p_value'])
             for op, cs in comparisons.items() for c in cs]
    adj = holm([t[3] for t in tests])
    holm_table = [{'op': t[0], 'a': t[1], 'b': t[2], 'p': t[3], 'p_holm': a_}
                  for t, a_ in zip(tests, adj)]

    four_class = [r for r in REGIONS
                  if len(set(data[r]['te_y'][data[r]['te_y'] != NORMAL].tolist())) == 4]
    excess = {f'{scope}_{unit}': typing_excess(data, regs, unit)
              for scope, regs in (('four_class', four_class), ('above3', [43, 46, 47]))
              for unit in ('sensor', 'site')}

    out = {
        'regions': REGIONS, 'far_regions': FAR_REGIONS,
        'four_class_regions': four_class,
        'sensor_level_auc': sensor_level_auc(data),
        'typing_excess_over_reference': excess,
        'mcnemar_false_dispatch_holm': holm_table,
        'auc_leak_vs_nuisance_pooled': pooled_auc,
        'auc_in_vs_out_pooled': pooled_io,
        'dispatch_rule': 'dispatch iff in/out; s = p(in) + p(out)',
        'miss_targets': MISS_TARGETS,
        'thresholds': {str(r): v for r, v in thresholds.items()},
        'per_region': {str(r): {
            'typing': v['typing'],
            'auc_normal_vs_rest': v['auc_normal_vs_rest'],
            'auc_leak_vs_nuisance': v['auc_leak_vs_nuisance'],
            'auc_in_vs_out': v['auc_in_vs_out'],
            'lgbm_oof_argmax_miss': v['lgbm_oof_argmax_miss'],
            **{op: {m: strip(v[op][m]) for m in MODELS} for op in ops}}
            for r, v in per_region.items()},
        'pooled': pooled, 'comparisons': comparisons, 'cost_break_even': costs,
        'diagnostic_test_fitted_matched_miss': matched_test_diagnostic(data),
        'curves_test_fitted': curves(data),
    }
    path = os.path.join(RES, OUT_NAME)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)

    for op in ops:
        print(f'=== {op} ===')
        print(f"  {'model':<20}{'miss(4)':>12}{'FD(4)':>8}{'nuis(4)':>12}"
              f"{'miss(8)':>14}{'nuis(8)':>14}")
        for m in MODELS:
            a, b = pooled[op][m]['far_regions'], pooled[op][m]['all_regions']
            print(f"  {m:<20}{a['missed']:>6}/{a['n_leak']:<5}{a['false_dispatch']:>8}"
                  f"{a['nuisance_dispatch']:>6}/{a['n_nuisance']:<5}"
                  f"{b['missed']:>7}/{b['n_leak']:<6}{b['nuisance_dispatch']:>7}/{b['n_nuisance']:<6}")
        for c in comparisons[op]:
            print(f"  {c['a']} vs {c['b']}: FD sensors {c['false_dispatch_sensors_a']}"
                  f" v {c['false_dispatch_sensors_b']} p={c['mcnemar_false_dispatch']['p_value']:.3f};"
                  f" never-dispatched leak sensors {c['never_dispatched_leak_sensors_a']}"
                  f" v {c['never_dispatched_leak_sensors_b']}"
                  f" p={c['mcnemar_never_dispatched']['p_value']:.3g}")
    print('\nAUC in/out vs noise/other among non-normal recordings:')
    for m in MODELS:
        per = [per_region[r]['auc_leak_vs_nuisance'].get(m) for r in REGIONS]
        print(f"  {m:<20} pooled {pooled_auc[m]:.3f}  per region "
              + ' '.join('--' if v is None else f'{v:.2f}' for v in per))
    print('AUC in vs out among true leaks:')
    for m in MODELS:
        per = [per_region[r]['auc_in_vs_out'].get(m) for r in REGIONS]
        print(f"  {m:<20} pooled {pooled_io[m]:.3f}  per region "
              + ' '.join('--' if v is None else f'{v:.2f}' for v in per))
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
