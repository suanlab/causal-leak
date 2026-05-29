"""Aggregate multi-seed results for domain-alignment baselines.

Reads:
  results/domain_alignment_results.json            (seed 42)
  results/domain_alignment_results_seed{0,1,2,3}.json

Writes:
  results/domain_alignment_multiseed.json
"""
import json
import os
import sys

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEEDS = [42, 0, 1, 2, 3]


def load_seed(seed):
    if seed == 42:
        path = os.path.join(RESULTS_DIR, 'domain_alignment_results.json')
    else:
        path = os.path.join(RESULTS_DIR,
                            f'domain_alignment_results_seed{seed}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def paired_wilcoxon(a, b):
    if len(a) != len(b) or len(a) < 3:
        return float('nan')
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(res.pvalue)
    except Exception:
        return float('nan')


def main():
    per_method = {'coral': {'f1': [], 'acc': [], 'seeds': []},
                  'mmd': {'f1': [], 'acc': [], 'seeds': []},
                  'dann': {'f1': [], 'acc': [], 'seeds': []}}

    for s in SEEDS:
        data = load_seed(s)
        if data is None:
            print(f'  seed {s}: NOT YET AVAILABLE', flush=True)
            continue
        for r in data['runs']:
            m = r['method']
            if m not in per_method:
                continue
            per_method[m]['f1'].append(r['test']['f1_macro'])
            per_method[m]['acc'].append(r['test']['acc'])
            per_method[m]['seeds'].append(s)

    summary = {}
    for method, d in per_method.items():
        if not d['f1']:
            summary[method] = {'error': 'no seeds available'}
            continue
        summary[method] = {
            'n_seeds': len(d['f1']),
            'seeds': d['seeds'],
            'f1_macro_mean': round(float(np.mean(d['f1'])), 4),
            'f1_macro_std': round(float(np.std(d['f1'], ddof=1))
                                  if len(d['f1']) > 1 else 0.0, 4),
            'f1_macro_per_seed': dict(zip(d['seeds'], [round(x, 4) for x in d['f1']])),
            'acc_mean': round(float(np.mean(d['acc'])), 4),
            'acc_std': round(float(np.std(d['acc'], ddof=1))
                             if len(d['acc']) > 1 else 0.0, 4),
        }

    # Paired Wilcoxon against XGBoost causal reference (0.4273, deterministic)
    xgb_ref = 0.4273
    tests = []
    for method, s in summary.items():
        if 'f1_macro_per_seed' not in s or s['n_seeds'] < 5:
            continue
        vals = [s['f1_macro_per_seed'][seed] for seed in SEEDS
                if seed in s['f1_macro_per_seed']]
        p = paired_wilcoxon(vals, [xgb_ref] * len(vals))
        tests.append({
            'comparison': f'{method} vs XGBoost_causal',
            'mean_diff': round(s['f1_macro_mean'] - xgb_ref, 4),
            'wilcoxon_p': round(p, 4) if not np.isnan(p) else 'n/a',
        })

    out = {
        'description': 'Multi-seed domain alignment on Chungbuk LORO fold',
        'seeds': SEEDS,
        'summary': summary,
        'paired_tests': tests,
    }

    out_path = os.path.join(RESULTS_DIR, 'domain_alignment_multiseed.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved {out_path}')

    print('\n=== DOMAIN ALIGNMENT MULTI-SEED SUMMARY ===')
    print(f'{"Method":>10s}  {"n":>3s}  {"F1 mean":>10s}  {"F1 std":>8s}  '
          f'{"Acc mean":>10s}')
    for m, s in summary.items():
        if 'f1_macro_mean' in s:
            print(f'{m:>10s}  {s["n_seeds"]:>3d}  '
                  f'{s["f1_macro_mean"]:10.4f}  {s["f1_macro_std"]:8.4f}  '
                  f'{s["acc_mean"]:10.4f}')

    print('\n=== Paired vs XGBoost causal (0.4273) ===')
    for t in tests:
        print(f'  {t["comparison"]}: diff={t["mean_diff"]:+.4f}  '
              f'p={t["wilcoxon_p"]}')


if __name__ == '__main__':
    main()
