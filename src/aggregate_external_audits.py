"""Aggregate the 7 MIMII-DG machine audits + SECOM + HAR into a single
comparison table for the paper. Writes results/external_audit_summary.json
and prints a markdown table.
"""
import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    rows = []

    # SECOM
    p = os.path.join(RESULTS_DIR, 'audit_external_secom_results.json')
    if os.path.exists(p):
        d = load(p)
        rows.append({
            'dataset': 'SECOM (UCI)',
            'domain': 'semiconductor',
            'n_rows': d['n_rows'],
            'p1_mean': d['audit']['P1_per_session_purity']['mean'],
            'p2_acc': d['audit']['P2_session_lookup']['covered_accuracy'],
            'p3_overlap': d['audit']['P3_session_overlap']['overlap_fraction'],
            'row_cv_acc': d['ladder']['row_cv']['acc_mean'],
            'group_cv_acc': d['ladder']['group_day_cv']['acc_mean'],
            'drop_pp': (d['ladder']['row_cv']['acc_mean']
                         - d['ladder']['group_day_cv']['acc_mean']) * 100,
            'source_to_target_acc': None,
        })

    # HAR
    p = os.path.join(RESULTS_DIR, 'audit_external_har_results.json')
    if os.path.exists(p):
        d = load(p)
        rows.append({
            'dataset': 'UCI HAR',
            'domain': 'wearable accel.',
            'n_rows': d['n_rows'],
            'p1_mean': d['audit']['P1_per_subject_purity']['mean'],
            'p2_acc': d['audit']['P2_subject_lookup']['covered_accuracy'],
            'p3_overlap': d['audit']['P3_subject_overlap']['overlap_fraction'],
            'row_cv_acc': d['ladder']['row_cv']['acc_mean'],
            'group_cv_acc': d['ladder']['group_subject_cv']['acc_mean'],
            'drop_pp': (d['ladder']['row_cv']['acc_mean']
                         - d['ladder']['group_subject_cv']['acc_mean']) * 100,
            'source_to_target_acc': None,
        })

    # MIMII-DG (7 machines)
    for machine in ['bearing', 'fan', 'gearbox', 'slider', 'valve',
                     'ToyCar', 'ToyTrain']:
        p = os.path.join(RESULTS_DIR,
                          f'audit_external_mimii_{machine}_results.json')
        if not os.path.exists(p):
            continue
        d = load(p)
        rows.append({
            'dataset': f'MIMII-DG {machine}',
            'domain': 'audio anomaly',
            'n_rows': d['n_rows'],
            'p1_mean': d['audit']['P1_per_section_purity']['mean'],
            'p2_acc': d['audit']['P2_section_lookup']['covered_accuracy'],
            'p3_overlap': d['audit']['P3_section_overlap']['overlap_fraction'],
            'row_cv_acc': d['ladder']['row_cv']['acc_mean'],
            'group_cv_acc': d['ladder']['group_section_cv']['acc_mean'],
            'drop_pp': (d['ladder']['row_cv']['acc_mean']
                         - d['ladder']['group_section_cv']['acc_mean']) * 100,
            'source_to_target_acc':
                d['ladder'].get('source_to_target', {}).get('acc'),
        })

    print(f'{"Dataset":<22s}  {"Domain":<18s}  {"n":>6s}  '
          f'{"P1":>5s}  {"P2":>5s}  {"P3":>5s}  '
          f'{"row":>5s}  {"grp":>5s}  {"drop_pp":>7s}  {"S->T":>5s}')
    for r in rows:
        s2t = f'{r["source_to_target_acc"]:.3f}' \
               if r['source_to_target_acc'] is not None else '   -'
        print(f'{r["dataset"]:<22s}  {r["domain"]:<18s}  '
              f'{r["n_rows"]:>6d}  '
              f'{r["p1_mean"]:>5.3f}  '
              f'{r["p2_acc"]:>5.3f}  '
              f'{r["p3_overlap"]:>5.2f}  '
              f'{r["row_cv_acc"]:>5.3f}  '
              f'{r["group_cv_acc"]:>5.3f}  '
              f'{r["drop_pp"]:>+7.2f}  '
              f'{s2t:>5s}')

    out = {'datasets': rows}
    with open(os.path.join(RESULTS_DIR, 'external_audit_summary.json'),
               'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
