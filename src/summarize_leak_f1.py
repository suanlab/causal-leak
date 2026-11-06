"""Leak-class macro-F1 for every typing result already on disk.

Five-class macro-F1 mixes two things. The normal class is separated by the
device's own trigger count (run_device_rule.py), so its F1 is 1.0 wherever
normals are present; and where they are absent, scikit-learn's default
averages in a 0.0 for it. Either way the five-class average moves for reasons
unrelated to leak typing, and averages over folds with different class sets
are not comparable.

This script recomputes, from stored per-class F1 or confusion matrices, the
mean F1 over the leak classes (in/out/noise/other) actually present in each
evaluation set. It fits nothing. The reference is 1/K for K leak classes
present: the expected leak-class macro-F1 of a classifier that guesses in
proportion to the class shares (each class's F1 is then its share).

  ladder     row-CV and group-CV (matrix_results), leave-one-region-out
             (loro_results; region-size weighted and unweighted)
  canonical  every model family on region 43 (seed means where run)
  county     the 16 county holdouts (sigungu_ood_results)
  temporal   the post-cutoff holdout (temporal_split_results)

Output: results/leak_f1_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import LABEL_NAMES, load_all  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
LEAK = [0, 1, 2, 3]
L = lambda n: json.load(open(os.path.join(RES, n)))  # noqa: E731


def leak_f1(per_class, present):
    vals = [per_class[c] for c in present]
    return float(np.mean(vals)) if vals else None


def present_from_cm(cm):
    return [c for c in LEAK if sum(cm[c]) > 0]


def per_class_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(5), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(5), where=cm.sum(1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros(5),
                     where=(prec + rec) > 0).tolist()


def ladder():
    out = {}
    for m in L('matrix_results.json'):
        if m['split_mode'] not in ('random', 'group_site'):
            continue
        vals = [leak_f1(f['f1_per_class'], present_from_cm(f['cm']))
                for f in m['folds']]
        out[f"{m['feat_mode']}_{m['split_mode']}"] = {
            'f1_macro_leak': float(np.mean(vals)), 'f1_macro5': m['f1_mean'],
            'acc': m['acc_mean']}
    for m in L('loro_results.json'):
        vals, w, ch = [], [], []
        for f in m['per_region']:
            pres = present_from_cm(f['cm'])
            v = leak_f1(f['f1_per_class'], pres)
            if v is not None:
                vals.append(v)
                w.append(f['n_test'])
                ch.append(1 / len(pres))
        # Leak-F1 is a property of the leak recordings, so the primary
        # weighting is by leak (non-normal) recordings over regions with at
        # least two leak classes; region 36 (one class) types nothing.
        variants = {}
        for kmin in (2, 3):
            lv, lw, lc = [], [], []
            for f in m['per_region']:
                cm_ = np.asarray(f['cm'])
                pres = present_from_cm(f['cm'])
                if len(pres) < kmin:
                    continue
                lv.append(leak_f1(f['f1_per_class'], pres))
                lw.append(cm_[:4].sum())
                lc.append(1 / len(pres))
            variants[f'k{kmin}'] = {
                'n_regions': len(lv),
                'leak_weighted': float(np.average(lv, weights=lw)),
                'leak_weighted_chance': float(np.average(lc, weights=lw)),
                'unweighted': float(np.mean(lv)),
                'unweighted_chance': float(np.mean(lc))}
        out[f"{m['feat_mode']}_loro"] = {
            'leak_rows_weighted': variants,
            'f1_macro_leak': float(np.mean(vals)),
            'f1_macro_leak_weighted': float(np.average(vals, weights=w)),
            'chance_mean': float(np.mean(ch)),
            'chance_weighted': float(np.average(ch, weights=w)),
            'f1_macro5': m['macro_f1'], 'acc_weighted': m['weighted_acc'],
            'n_regions': len(vals)}
    return out


def canonical():
    rows = {}
    op = L('operational_folds_results.json')
    f43 = next(f for f in op['folds'] if f['region'] == 43)['models']
    for k in ('lightgbm', 'tree_only_soft_avg', 'xgb_flat', 'xgb_hier'):
        cm = f43[k]['cm']
        rows[k] = {'f1_macro_leak': leak_f1(per_class_from_cm(cm),
                                            present_from_cm(cm)),
                   'f1_macro5': f43[k]['f1_macro'], 'n_seeds': 1}
    cm43 = f43['lightgbm']['cm']
    present = present_from_cm(cm43)

    ebm = L('extra_baselines_multiseed.json')['summary']
    for k in ('catboost', 'random_forest', 'tabnet_causal'):
        v = [leak_f1(s['f1_per_class'], present) for s in ebm[k]['per_seed']]
        rows[k] = {'f1_macro_leak': float(np.mean(v)),
                   'f1_macro_leak_std': float(np.std(v)),
                   'f1_macro5': ebm[k]['f1_macro_mean'], 'n_seeds': len(v)}

    for net in ('cnn', 'mlp'):
        per = {}
        for sfx in ('', '_seed0', '_seed1', '_seed2', '_seed3'):
            d = L(f'irm_{net}_results{sfx}.json')
            runs = d['runs'] if isinstance(d, dict) else d
            for r in runs:
                per.setdefault(r['method'], []).append(
                    leak_f1(r['test']['f1_per_class'], present))
        for meth, v in per.items():
            rows[f'{net}_{meth}'] = {'f1_macro_leak': float(np.mean(v)),
                                     'f1_macro_leak_std': float(np.std(v)),
                                     'n_seeds': len(v)}

    # Networks early-stopped on whole held-out training regions
    # (CAUSAL_VAL_MODE=region), when those runs exist.
    for net in ('cnn', 'mlp'):
        per = {}
        for sfx in ('', '_seed0', '_seed1', '_seed2', '_seed3'):
            f = os.path.join(RES, f'irm_{net}_results_regionval{sfx}.json')
            if not os.path.exists(f):
                continue
            for r in json.load(open(f))['runs']:
                per.setdefault(r['method'], []).append(
                    (leak_f1(r['test']['f1_per_class'], present),
                     r['test']['f1_macro']))
        for meth, v in per.items():
            lf, f5 = np.array(v).T
            rows[f'{net}_{meth}_regionval'] = {
                'f1_macro_leak': float(lf.mean()), 'f1_macro_leak_std': float(lf.std()),
                'f1_macro5': float(f5.mean()), 'f1_macro5_std': float(f5.std()),
                'n_seeds': len(v)}
    f = os.path.join(RES, 'domain_alignment_results_regionval.json')
    if os.path.exists(f):
        runs_rv = json.load(open(f))['runs']
        for meth in ('coral', 'mmd', 'dann'):
            # Under region validation the weight is chosen on validation F1.
            best = max((r for r in runs_rv if r['method'] == meth),
                       key=lambda r: r['best_val_f1'])
            rows[f'{meth}_regionval'] = {
                'f1_macro_leak': leak_f1(best['test']['f1_per_class'], present),
                'f1_macro5': best['test']['f1_macro'], 'n_seeds': 1,
                'lambda': best.get('align_lambda')}

    # Alignment methods: the lambda with the best TEST macro-F1, as in the
    # published table. That choice favours these methods.
    for meth in ('coral', 'mmd', 'dann'):
        runs = [r for r in L('domain_alignment_results.json')['runs']
                if r['method'] == meth]
        best = max(runs, key=lambda r: r['test']['f1_macro'])
        rows[meth] = {'f1_macro_leak': leak_f1(best['test']['f1_per_class'],
                                               present),
                      'f1_macro5': best['test']['f1_macro'], 'n_seeds': 1,
                      'lambda': best.get('align_lambda')}
    return rows


def county_temporal():
    df = load_all()
    df['sg'] = df['region_lvl1'].astype(str) + '_' + df['region_lvl2'].astype(str)
    sg = L('sigungu_ood_results.json')
    county = []
    for f in sg['folds']:
        y = df.loc[df['sg'] == f['sigungu'], 'label']
        present = [c for c in LEAK if (y == c).any()]
        county.append({
            'sigungu': f['sigungu'], 'n_test': f['n_test'],
            'leak_classes_present': [LABEL_NAMES[c] for c in present],
            'f1_macro5': f['nested_causal']['f1_macro'],
            'f1_macro_leak': leak_f1(f['nested_causal']['f1_per_class'], present),
            'chance': 1 / len(present),
        })
    w = np.array([c['n_test'] for c in county], dtype=float)
    f4 = np.array([c['f1_macro_leak'] for c in county])
    ch = np.array([c['chance'] for c in county])

    tp = L('temporal_split_results.json')
    dist = tp['split_info']['test_class_dist']
    present_t = [c for c in LEAK if dist.get(LABEL_NAMES[c], 0) > 0]
    temporal = {k: {'f1_macro5': m['f1_macro'],
                    'f1_macro_leak': leak_f1(
                        [m['f1_per_class'][LABEL_NAMES[c]] for c in range(5)],
                        present_t)}
                for k, m in tp['models'].items()}
    return ({'folds': county, 'n_folds': len(county),
             'mean_f1_macro_leak': float(f4.mean()),
             'weighted_mean_f1_macro_leak': float(np.average(f4, weights=w)),
             'min': float(f4.min()), 'max': float(f4.max()),
             'chance_mean': float(ch.mean()),
             'folds_above_chance': int((f4 > ch).sum())},
            {'cutoff_date': tp['split_info']['cutoff_date'],
             'chance': 1 / len(present_t), 'models': temporal})


def main():
    county, temporal = county_temporal()
    out = {'definition': 'mean five-class F1 over in/out/noise/other present',
           'chance_definition': '1/K for K leak classes present',
           'ladder': ladder(),
           'canonical_region43': {'chance': 0.25, 'models': canonical()},
           'county': county, 'temporal': temporal}
    with open(os.path.join(RES, 'leak_f1_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    for k, v in out['ladder'].items():
        print(f'ladder {k:<18} leak-F1 {v["f1_macro_leak"]:.4f}'
              + (f' (weighted {v["f1_macro_leak_weighted"]:.4f}; chance '
                 f'{v["chance_mean"]:.3f} / weighted {v["chance_weighted"]:.3f})'
                 if 'f1_macro_leak_weighted' in v else ''))
    for k, v in sorted(out['canonical_region43']['models'].items(),
                       key=lambda kv: -kv[1]['f1_macro_leak']):
        sd = f' ± {v["f1_macro_leak_std"]:.3f}' if 'f1_macro_leak_std' in v else ''
        print(f'region 43 {k:<20} leak-F1 {v["f1_macro_leak"]:.4f}{sd}')
    print(f"county: {county['n_folds']} folds, mean {county['mean_f1_macro_leak']:.4f}"
          f" (weighted {county['weighted_mean_f1_macro_leak']:.4f}, "
          f"range {county['min']:.3f}-{county['max']:.3f}; chance "
          f"{county['chance_mean']:.3f}; above chance in "
          f"{county['folds_above_chance']})")
    for k, v in temporal['models'].items():
        print(f"temporal {k:<22} leak-F1 {v['f1_macro_leak']:.4f}")


if __name__ == '__main__':
    main()
