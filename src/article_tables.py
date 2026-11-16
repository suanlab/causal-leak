"""Rows of the DMKD article's data tables, generated from the result files.

Each function returns the LaTeX rows of one table (model families, per-region
typing, triage outcomes, per-region transfer, evaluation ladder, region by
class), so no table cell is typed by hand. src/cited_numbers.py prints them
with --tables.
"""
import json, os
R = os.path.join(os.path.dirname(__file__), '..', 'results')
L = lambda n: json.load(open(os.path.join(R, n)))
# The DMKD article reports the dispatch comparison on all 534 signal columns
# (no band selection, so calibration is fully nested); the nested-band run is
# a sensitivity analysis.
DISPATCH = 'dispatch_operating_points_all_results.json'


def f3(x): return f'{x:.3f}'


def rng2(v):
    lo, hi = f'{min(v):.2f}', f'{max(v):.2f}'
    return lo if lo == hi else f'{lo}--{hi}'


def models_rows():
    c = L('leak_f1_results.json')['canonical_region43']['models']
    ms = L('multiseed_summary.json')

    def five(k):
        if k.startswith(('cnn_', 'mlp_')):
            net, m = k.split('_', 1)
            return f"{ms[net][m]['test_f1_mean']:.3f}"
        return f3(c[k]['f1_macro5'])

    def leak(k):
        if k not in c:
            return '--'
        v = c[k]
        s = f3(v['f1_macro_leak'])
        return f'${s} \\pm {v["f1_macro_leak_std"]:.3f}$' if 'f1_macro_leak_std' in v else s

    nets = ('cnn_', 'mlp_', 'coral', 'mmd', 'dann')
    fam = [('Trees', [('lightgbm', 'LightGBM'), ('tree_only_soft_avg', 'Tree-only soft average'),
                      ('xgb_flat', 'Gradient-boosted, flat'), ('xgb_hier', 'Gradient-boosted, hierarchical'),
                      ('catboost', 'CatBoost'), ('random_forest', 'Random forest')]),
           ('Tabular DL', [('tabnet_causal', 'TabNet')]),
           ('Invariance', [('cnn_irm', 'CNN, IRM'), ('cnn_erm', 'CNN, ERM'), ('cnn_groupdro', 'CNN, GroupDRO'),
                           ('mlp_irm', 'MLP, IRM'), ('mlp_groupdro', 'MLP, GroupDRO'), ('mlp_erm', 'MLP, ERM')]),
           ('Alignment', [('coral', 'DeepCORAL'), ('mmd', 'MMD-DG'), ('dann', 'DANN')])]
    out = []
    for i, (f, rows) in enumerate(fam):
        if i:
            out.append('\\midrule')
        for j, (k, lab) in enumerate(rows):
            rv = leak(f'{k}_regionval') if k.startswith(nets) else '--'
            out.append(f"{f if j == 0 else ''} & {lab} & {leak(k)} & {rv} & {five(k)} \\\\")
    return '\n'.join(out)


def typing_rows():
    d = L(DISPATCH)['per_region']
    M = ['xgb_flat', 'xgb_hier', 'tree_only_soft_avg', 'lightgbm']
    out = []
    for r in ['43', '46', '47', '45', '44', '48', '27', '29']:
        t = d[r]['typing']
        k = len([c for c in t['lightgbm']['classes_present'] if c != 'normal'])
        io = [d[r]['auc_in_vs_out'][m] for m in M]
        out.append(f"{r} & {k} & {1/k:.3f} & " + ' & '.join(f3(t[m]['f1_macro_leak']) for m in M)
                   + f" & {rng2(io)} \\\\")
    return '\n'.join(out)


def dispatch_rows():
    d = L(DISPATCH)
    names = {'tree_only_soft_avg': 'Tree-only soft avg.', 'xgb_hier': 'GBT hierarchical',
             'xgb_flat': 'GBT flat', 'lightgbm': 'LightGBM'}
    ops = [('argmax', 'Argmax'), ('miss01', 'Calibrated, 1\\,\\%'), ('miss05', 'Calibrated, 5\\,\\%'),
           ('miss10', 'Calibrated, 10\\,\\%'), ('miss20', 'Calibrated, 20\\,\\%')]
    out = []
    for i, (op, lab) in enumerate(ops):
        if i:
            out.append('\\midrule')
        comp = {c['a']: c for c in d['comparisons'][op]}
        for j, m in enumerate(['tree_only_soft_avg', 'xgb_hier', 'xgb_flat', 'lightgbm']):
            a = d['pooled'][op][m]['all_regions']; f = d['pooled'][op][m]['far_regions']
            sens = comp[m]['false_dispatch_sensors_a'] if m in comp else comp['tree_only_soft_avg']['false_dispatch_sensors_b']
            out.append(f"{lab if j == 0 else ''} & {names[m]} & {100*a['missed']/a['n_leak']:.1f} & "
                       f"{100*a['nuisance_dispatch']/a['n_nuisance']:.1f} & {f['false_dispatch']} ({sens}) \\\\")
    return '\n'.join(out)


def transfer_rows(op='miss10'):
    d = L(DISPATCH)['per_region']
    out = []
    for r in ['27', '29', '43', '44', '45', '46', '47', '48']:
        e, g = d[r][op]['tree_only_soft_avg'], d[r][op]['lightgbm']
        nuis = (lambda x: f"{100 * x['nuisance_dispatch'] / x['n_nuisance']:.1f}"
                if x['n_nuisance'] else '--')
        out.append(f"{r} & {e['n_leak']:,} & {e['leak_sensors']} & {100 * e['missed'] / e['n_leak']:.1f} & "
                   f"{100 * g['missed'] / g['n_leak']:.1f} & {nuis(e)} & {nuis(g)} \\\\"
                   .replace(',', '{,}'))
    return '\n'.join(out)


def ladder_rows():
    m = {(r['feat_mode'], r['split_mode']): r for r in L('matrix_results.json')}
    lad = L('leak_f1_results.json')['ladder']
    sh = L('sensor_holdout_results.json')['ladder_group_sensor']
    lo = {r['feat_mode']: r for r in L('loro_results.json')}

    def f5(v):
        return sum(x['f1_macro'] for x in v['folds']) / len(v['folds'])
    rows = [('Row-CV (published protocol)',
             lambda f: (m[(f, 'random')]['acc_mean'], m[(f, 'random')]['f1_mean'],
                        lad[f'{f}_random']['f1_macro_leak'])),
            ('Group-CV by sensor',
             lambda f: (sh[f]['acc_mean'], f5(sh[f]), sh[f]['alarm_f1_mean'])),
            ('Group-CV by site',
             lambda f: (m[(f, 'group_site')]['acc_mean'], m[(f, 'group_site')]['f1_mean'],
                        lad[f'{f}_group_site']['f1_macro_leak'])),
            ('LORO (weighted)',
             lambda f: (lo[f]['weighted_acc'], lo[f]['weighted_f1'],
                        lad[f'{f}_loro']['leak_rows_weighted']['k2']['leak_weighted']))]
    return '\n'.join(f"{lab} & " + ' & '.join(f3(v) for f in ('full', 'signal') for v in fn(f))
                     + ' \\\\' for lab, fn in rows)


def regions_rows():
    lo = {r['feat_mode']: r for r in L('loro_results.json')}['signal']
    out, tot = [], [0] * 6
    for f in sorted(lo['per_region'], key=lambda x: x['region']):
        n = [sum(row) for row in f['cm']]
        vals = n + [sum(n)]
        tot = [a + b for a, b in zip(tot, vals)]
        out.append(f"{f['region']} & " + ' & '.join(f'{v:,}'.replace(',', '{,}') for v in vals) + ' \\\\')
    out.append('All & ' + ' & '.join(f'{v:,}'.replace(',', '{,}') for v in tot) + ' \\\\')
    return '\n'.join(out)


if __name__ == '__main__':
    print('%% MODELS'); print(models_rows())
    print('%% TYPING'); print(typing_rows())
    print('%% DISPATCH'); print(dispatch_rows())
    print('%% TRANSFER'); print(transfer_rows())
    print('%% LADDER'); print(ladder_rows())
    print('%% REGIONS'); print(regions_rows())
