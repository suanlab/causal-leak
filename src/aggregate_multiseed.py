"""Aggregate multi-seed results across {0, 1, 2, 3, 42} seeds.

Reads:
  results/irm_mlp_results.json            (seed 42)
  results/irm_mlp_results_seed{0,1,2,3}.json
  results/irm_cnn_results.json            (seed 42)
  results/irm_cnn_results_seed{0,1,2,3}.json
  results/augmentation_results.json       (seed 42)
  results/augmentation_results_seed{0,1,2,3}.json
  results/pga_results.json                (seed 42)
  results/pga_results_seed{0,1,2,3}.json

Writes:
  results/multiseed_summary.json
  results/multiseed_summary.md

Reports mean ± std (sample, ddof=1) of test_f1_macro and test_acc per
configuration, plus a paired Wilcoxon signed-rank p-value where two
methods can be paired across seeds.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEEDS = [42, 0, 1, 2, 3]


def load_seed_results(base: str) -> Dict[int, dict]:
    out = {}
    for s in SEEDS:
        if s == 42:
            path = os.path.join(RESULTS_DIR, f'{base}.json')
        else:
            path = os.path.join(RESULTS_DIR, f'{base}_seed{s}.json')
        if os.path.exists(path):
            with open(path) as f:
                out[s] = json.load(f)
    return out


def aggregate_irm(base: str):
    """Aggregate runs from irm_*_results files (list of method dicts)."""
    seed_data = load_seed_results(base)
    method_metrics: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {'test_f1': [], 'test_acc': [], 'val_f1': [], 'seeds': []})
    for s, data in seed_data.items():
        runs = data.get('runs', data) if isinstance(data, dict) else data
        for r in runs:
            method = r.get('method', 'unknown')
            method_metrics[method]['test_f1'].append(r['test']['f1_macro'])
            method_metrics[method]['test_acc'].append(r['test']['acc'])
            method_metrics[method]['val_f1'].append(r['val']['f1_macro'])
            method_metrics[method]['seeds'].append(s)
    summary = {}
    for m, d in method_metrics.items():
        if not d['test_f1']:
            continue
        summary[m] = {
            'n_seeds': len(d['seeds']),
            'seeds': d['seeds'],
            'test_f1_mean': float(np.mean(d['test_f1'])),
            'test_f1_std': float(np.std(d['test_f1'], ddof=1)
                                  if len(d['test_f1']) > 1 else 0.0),
            'test_f1_seeds': d['test_f1'],
            'test_acc_mean': float(np.mean(d['test_acc'])),
            'test_acc_std': float(np.std(d['test_acc'], ddof=1)
                                   if len(d['test_acc']) > 1 else 0.0),
            'val_f1_mean': float(np.mean(d['val_f1'])),
        }
    return summary


def aggregate_aug(base: str = 'augmentation_results'):
    seed_data = load_seed_results(base)
    rec_metrics: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {'test_f1': [], 'test_acc': [], 'val_f1': [], 'seeds': []})
    for s, data in seed_data.items():
        runs = data.get('runs', data) if isinstance(data, dict) else data
        for r in runs:
            name = r.get('name', 'unknown')
            rec_metrics[name]['test_f1'].append(r['test_f1'])
            rec_metrics[name]['test_acc'].append(r['test_acc'])
            rec_metrics[name]['val_f1'].append(r.get('val_f1', float('nan')))
            rec_metrics[name]['seeds'].append(s)
    summary = {}
    for n, d in rec_metrics.items():
        if not d['test_f1']:
            continue
        summary[n] = {
            'n_seeds': len(d['seeds']),
            'seeds': d['seeds'],
            'test_f1_mean': float(np.mean(d['test_f1'])),
            'test_f1_std': float(np.std(d['test_f1'], ddof=1)
                                  if len(d['test_f1']) > 1 else 0.0),
            'test_f1_seeds': d['test_f1'],
            'test_acc_mean': float(np.mean(d['test_acc'])),
            'test_acc_std': float(np.std(d['test_acc'], ddof=1)
                                   if len(d['test_acc']) > 1 else 0.0),
        }
    return summary


def aggregate_pga(base: str = 'pga_results'):
    seed_data = load_seed_results(base)
    cfg_metrics: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {'test_f1': [], 'test_acc': [], 'val_f1': [], 'seeds': []})
    for s, data in seed_data.items():
        runs = data.get('runs', [])
        for r in runs:
            tag = r.get('tag', 'unknown')
            cfg_metrics[tag]['test_f1'].append(r['test_f1'])
            cfg_metrics[tag]['test_acc'].append(r['test_acc'])
            cfg_metrics[tag]['val_f1'].append(r.get('val_f1', float('nan')))
            cfg_metrics[tag]['seeds'].append(s)
    summary = {}
    for c, d in cfg_metrics.items():
        if not d['test_f1']:
            continue
        summary[c] = {
            'n_seeds': len(d['seeds']),
            'seeds': d['seeds'],
            'test_f1_mean': float(np.mean(d['test_f1'])),
            'test_f1_std': float(np.std(d['test_f1'], ddof=1)
                                  if len(d['test_f1']) > 1 else 0.0),
            'test_acc_mean': float(np.mean(d['test_acc'])),
            'test_acc_std': float(np.std(d['test_acc'], ddof=1)
                                   if len(d['test_acc']) > 1 else 0.0),
        }
    return summary


def paired_wilcoxon(a: List[float], b: List[float]) -> float:
    """Wilcoxon signed-rank p-value for paired samples."""
    if len(a) != len(b) or len(a) < 3:
        return float('nan')
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(res.pvalue)
    except Exception:
        return float('nan')


def main():
    out = {
        'mlp': aggregate_irm('irm_mlp_results'),
        'cnn': aggregate_irm('irm_cnn_results'),
        'aug': aggregate_aug('augmentation_results'),
        'pga': aggregate_pga('pga_results'),
    }

    # Pairwise tests on common seed set
    pairs = []
    if 'mlp' in out:
        m = out['mlp']
        if 'erm' in m and 'groupdro' in m and \
           m['erm']['seeds'] == m['groupdro']['seeds']:
            p = paired_wilcoxon(m['groupdro']['test_f1_seeds'],
                                m['erm']['test_f1_seeds'])
            pairs.append({
                'comparison': 'MLP GroupDRO vs MLP ERM (test_f1)',
                'wilcoxon_p': p,
                'mean_diff': m['groupdro']['test_f1_mean']
                              - m['erm']['test_f1_mean'],
                'n_seeds': len(m['groupdro']['seeds']),
            })
        if 'irm' in m and 'erm' in m and \
           m['irm']['seeds'] == m['erm']['seeds']:
            p = paired_wilcoxon(m['irm']['test_f1_seeds'],
                                m['erm']['test_f1_seeds'])
            pairs.append({
                'comparison': 'MLP IRM vs MLP ERM (test_f1)',
                'wilcoxon_p': p,
                'mean_diff': m['irm']['test_f1_mean']
                              - m['erm']['test_f1_mean'],
                'n_seeds': len(m['irm']['seeds']),
            })

    out['paired_tests'] = pairs

    summary_path = os.path.join(RESULTS_DIR, 'multiseed_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved {summary_path}')

    # Markdown summary
    md = ['# Multi-seed summary (E1)\n\n']
    md.append(f'Seeds attempted: {SEEDS}\n\n')
    for family, data in out.items():
        if family == 'paired_tests':
            continue
        md.append(f'## {family.upper()}\n\n')
        md.append('| Config | N seeds | test_f1 (mean) | test_f1 (std) '
                  '| test_acc (mean) | test_acc (std) |\n')
        md.append('|--------|--------:|---------------:|--------------:'
                  '|----------------:|---------------:|\n')
        for cfg, m in data.items():
            md.append(f'| {cfg} | {m["n_seeds"]} | '
                      f'{m["test_f1_mean"]:.4f} | '
                      f'{m["test_f1_std"]:.4f} | '
                      f'{m["test_acc_mean"]:.4f} | '
                      f'{m["test_acc_std"]:.4f} |\n')
        md.append('\n')

    if out['paired_tests']:
        md.append('## Paired Wilcoxon tests\n\n')
        md.append('| Comparison | mean Δ | Wilcoxon p | n |\n')
        md.append('|------------|-------:|----------:|--:|\n')
        for p in out['paired_tests']:
            md.append(f'| {p["comparison"]} | {p["mean_diff"]:+.4f} '
                      f'| {p["wilcoxon_p"]:.4f} | {p["n_seeds"]} |\n')

    with open(os.path.join(RESULTS_DIR, 'multiseed_summary.md'), 'w') as f:
        f.write(''.join(md))
    print(f'Saved {os.path.join(RESULTS_DIR, "multiseed_summary.md")}')


if __name__ == '__main__':
    main()
