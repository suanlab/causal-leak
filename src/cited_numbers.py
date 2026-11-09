"""Print every number the DMKD article cites, from the released result files.

Each line is  <section>  <quantity>  <value>, with values formatted as the
article prints them. Only result files under results/ are read, so the
script runs without the AI Hub data. Recompute the result files with
scripts/reproduce_all.sh, then run this script to see what the article
should say.

Usage: python src/cited_numbers.py [--json]
"""
import json
import os
import sys

import numpy as np

R = os.path.join(os.path.dirname(__file__), '..', 'results')
ROWS = []


def L(name):
    with open(os.path.join(R, name)) as fh:
        return json.load(fh)


def num(v, fmt='{:.3f}'):
    """Article formatting: thousands separators for integers >= 1,000."""
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    return fmt.format(v)


def put(section, quantity, value, fmt='{:.3f}'):
    ROWS.append((section, quantity, num(value, fmt) if not isinstance(value, str) else value))


def main():
    ia = L('identifier_audit_results.json')
    put('5.1 A1', 'sensors', ia['n_sensors'])
    put('5.1 A1', 'training sensors', ia['train_sensors'])
    put('5.1 A1', 'validation sensors', ia['val_sensors'])
    put('5.1 A1', 'validation sensors also in training', ia['val_sensors_in_train'])
    put('5.1 A1', 'share of validation sensors in training (%)',
        100 * ia['val_sensors_in_train'] / ia['val_sensors'], '{:.2f}')
    put('5.1 A1', 'lookup coverage (%)', 100 * ia['lookup_coverage'], '{:.1f}')
    put('5.1 A1', 'uncovered validation rows', ia['uncovered_rows'])
    put('5.1 A1', 'uncovered rows with lrate < 90', ia['uncovered_lrate_below_90'])
    put('5.1 A1', 'lookup + lrate rule, conservative (%)',
        100 * ia['lookup_plus_lrate_accuracy_conservative'], '{:.2f}')
    put('5.1 A1', 'lookup + lrate rule, filled (%)',
        100 * ia['lookup_plus_lrate_accuracy_filled'], '{:.2f}')

    ai = L('audit_inference_results.json')
    lf = ai['lift']
    put('5.1 A1', 'lookup alone, uncovered as errors (%)', 100 * lf['lookup_accuracy'], '{:.2f}')
    put('5.1 A1', 'lift L', lf['lift'])
    put('5.1 A1', 'largest class share (%)', 100 * lf['p_max'], '{:.1f}')
    put('5.1 A1', 'permutation null: mean L', lf['null_mean'])
    put('5.1 A1', 'permutation null: 95th percentile of L', lf['null_q95'])

    dr = L('device_rule_results.json')
    put('5.2 A2', 'identified threshold T (every fold)',
        str(sorted({f['T'] for f in dr['folds']})))
    put('5.2 A2', 'k range', '{}--{}'.format(*dr['pooled_fit']['k_range']))
    p, f4 = dr['pooled_loro'], dr['far_regions_loro']
    put('5.2 A2', 'normal recordings flagged / total', f"{p['false_flags']} / {num(p['n_normal'])}")
    put('5.2 A2', 'non-normal recordings missed / total', f"{p['missed']} / {num(p['n_rest'])}")
    put('5.2 A2', 'regions 27/29/44/45: normals flagged / total',
        f"{f4['false_flags']} / {num(f4['n_normal'])} ({num(f4['normal_sensors'])} sensors)")
    put('5.2 A2', 'regions 27/29/44/45: non-normal missed / total',
        f"{f4['missed']} / {num(f4['n_rest'])}")
    h = dr['count_histogram_at_pooled_T']
    put('5.2 A2', 'non-normal recordings at count 9', h['non_normal'][9])
    put('5.2 A2', 'non-normal recordings at count 10', h['non_normal'][10])
    put('5.2 A2', 'recordings at count 8', h['normal'][8] + h['non_normal'][8])
    dp = dr['depletion_at_pooled_T']
    put('5.2 A2', 'T3 boundary bin: observed / reference',
        f"{dp['boundary']['observed']} / {num(int(dp['boundary']['expected']))}")
    put('5.2 A2', 'T3 floor bin: observed / reference',
        f"{dp['floor']['observed']} / {num(int(dp['floor']['expected']))}")
    put('5.2 A2', 'superseded step-10 grid: misses',
        dr['grid_search']['grid10']['pooled_loro']['missed'])
    su = ai['sufficiency']
    put('5.2 A2', 'T2 errors: count only / signal / both',
        f"{su['count_only']['errors']} / {su['signal']['errors']} / "
        f"{su['signal_plus_count']['errors']}")

    m = {(r['feat_mode'], r['split_mode']): r for r in L('matrix_results.json')}
    lad = L('leak_f1_results.json')['ladder']
    sh = L('sensor_holdout_results.json')
    lo = {r['feat_mode']: r for r in L('loro_results.json')}
    for feat in ('full', 'signal'):
        g = sh['ladder_group_sensor'][feat]
        k2 = lad[f'{feat}_loro']['leak_rows_weighted']['k2']
        put('5.3 ladder', f'{feat}: accuracy row-CV / sensor / site / LORO',
            ' / '.join(num(v) for v in (m[(feat, 'random')]['acc_mean'], g['acc_mean'],
                                         m[(feat, 'group_site')]['acc_mean'],
                                         lo[feat]['weighted_acc'])))
        put('5.3 ladder', f'{feat}: alarm-F1 row-CV / sensor / site / LORO',
            ' / '.join(num(v) for v in (lad[f'{feat}_random']['f1_macro_leak'],
                                         g['alarm_f1_mean'],
                                         lad[f'{feat}_group_site']['f1_macro_leak'],
                                         k2['leak_weighted'])))
    put('5.3 ladder', 'LORO class-share reference',
        lad['signal_loro']['leak_rows_weighted']['k2']['leak_weighted_chance'])
    vw = ai['variance']['weighted']
    put('5.3 ladder', 'variance share (%): sensor / site / class / class within site',
        ' / '.join(f'{100 * vw[k]:.1f}' for k in ('sensor', 'site', 'class', 'class_within_site')))
    wr = sh['within_region']
    put('5.3 ladder', 'within region 46 / 47: new sensor',
        f"{num(wr['46']['sensor']['alarm_f1_mean'])} / {num(wr['47']['sensor']['alarm_f1_mean'])}")
    put('5.3 ladder', 'within region 46 / 47: new site',
        f"{num(wr['46']['site']['alarm_f1_mean'])} / {num(wr['47']['site']['alarm_f1_mean'])}")

    rl = ai['region_level']
    per = {r['region']: r for r in rl['signal']['per_region']}
    put('5.4 A3', 'excess over 1/K, regions 43-48',
        ', '.join(f"{per[g]['alarm_f1_excess']:+.3f}" for g in (43, 44, 45, 46, 47, 48)))
    for met, lab in (('alarm_f1', 'macro-F1'), ('balanced_acc', 'balanced accuracy')):
        x = rl['signal']['k4'][met]
        put('5.4 A3', f'{lab}: mean excess, t-interval, sign-flip p, detectable excess',
            f"{x['mean']:+.3f}, [{x['t_ci'][0]:+.3f}, {x['t_ci'][1]:+.3f}], "
            f"{x['sign_flip_p']:.3f}, {x['mde80']:.3f}")
    c = rl['county']
    put('5.4 A3', 'counties: mean alarm-F1 / reference / above',
        f"{num(c['mean_alarm_f1'])} / {num(c['mean_chance'])} / {c['alarm_f1']['n_above']} of {c['n_counties']}")
    put('5.4 A3', 'counties: t-interval of the mean excess',
        f"[{c['alarm_f1']['t_ci'][0]:+.3f}, {c['alarm_f1']['t_ci'][1]:+.3f}]")
    cc = ai['canonical_ci']
    for mname, v in cc['models'].items():
        put('5.5 families', f'region 43 {mname}: alarm-F1 [sensor interval]',
            f"{num(v['alarm_f1'])} [{v['sensor_ci'][0]:.2f}, {v['sensor_ci'][1]:.2f}]")
    put('5.5 families', 'region 43 alarm sensors / at largest site',
        f"{cc['alarm_sensors']} / {cc['alarm_sensors_at_largest_site']}")

    d = L('dispatch_operating_points_results.json')
    au = d['auc_leak_vs_nuisance_pooled']
    put('5.6 A4', 'leak-vs-noise AUC, pooled (range over models)',
        f"{min(au.values()):.3f}--{max(au.values()):.3f}")
    for op in ('miss01', 'miss05', 'miss10', 'miss20'):
        vals = [100 * d['pooled'][op][mm]['all_regions']['missed'] /
                d['pooled'][op][mm]['all_regions']['n_leak'] for mm in d['pooled'][op]]
        put('5.6 A4', f'pooled missed leaks (%) at {op}', f'{min(vals):.1f}--{max(vals):.1f}')
    tr = [100 * d['per_region'][str(r)]['miss10'][mm]['missed'] / d['per_region'][str(r)]['miss10'][mm]['n_leak']
          for r in d['regions'] for mm in ('tree_only_soft_avg', 'lightgbm')
          if d['per_region'][str(r)]['miss10'][mm]['n_leak']]
    put('5.6 A4', 'per-region missed leaks at miss10 (%)', f'{min(tr):.1f}--{max(tr):.1f}')
    cd = ai['calibration_dispersion']
    mx = [v['tree_only_soft_avg']['miss10']['max'] for v in cd.values()]
    put('5.6 A4', 'training regions own miss at the 10 % threshold: max over folds (%)',
        f'{100 * min(mx):.0f}--{100 * max(mx):.0f}')
    st = ai['site_tests']
    put('App. A', 'site-level sign tests: smallest p / smallest Holm p',
        f"{st['min_p']:.3f} / {st['min_p_holm']:.2f}")

    ss = L('audit_semisynthetic_results.json')
    put('3.3 planted', 'CWRU windows', ss['n_windows'])
    for sc in ('S1', 'S2', 'S3'):
        r = ss['a2'][sc]
        put('3.3 planted', f'{sc}: T1 / T2 (signal / count) / T3 (n_b / ref; bound)',
            f"{'pass' if r['T1']['pass'] else 'fail'} / {r['T2']['model_errors']} / "
            f"{r['T2']['count_errors']} / {r['T3']['boundary']['observed']} / "
            f"{r['T3']['boundary']['expected']:.0f}; {r['T3']['boundary']['ratio_upper95']:.2f}")
    put('3.3 planted', 'A1 lift: bearing / load (p)',
        f"{ss['a1_real']['bearing']['lift']:.3f} ({ss['a1_real']['bearing']['p_value']:.3f}) / "
        f"{ss['a1_real']['load']['lift']:.3f} ({ss['a1_real']['load']['p_value']:.2f})")
    rr = np.array([x['reject_rate'] for x in ss['t3_size']])
    put('3.3 planted', 'T3 size: mean false rejection / worst case',
        f'{rr.mean():.3f} / {rr.max():.3f}')
    put('3.3 planted', 'T3 power by share restored',
        ', '.join(f"{x['epsilon']:g}: {x['reject_rate']:.1f}" for x in ss['t3_power']))
    a3 = ss['a3_cwru']
    put('3.3 planted', 'A3 CWRU macro-F1, fault size held out (007 / 014 / 021)',
        ' / '.join(num(a3[f'held_out_{k}']['macro_f1']) for k in ('007', '014', '021')))

    if '--json' in sys.argv:
        print(json.dumps([{'section': s, 'quantity': q, 'value': v} for s, q, v in ROWS],
                         indent=1, ensure_ascii=False))
        return
    w = max(len(q) for _, q, _ in ROWS)
    for s, q, v in ROWS:
        print(f'{s:<14} {q:<{w}}  {v}')


if __name__ == '__main__':
    main()
