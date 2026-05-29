"""Figure 14 — Cost-vs-Accuracy Pareto scatter for the canonical
Chungbuk LORO fold.

X-axis: inference time per 1k rows (ms, log scale).
Y-axis: OOD macro-F1.
Size: peak RAM (MB).
Color: family (tree-ensemble / tabular-DL / domain-DL / DA-baseline).

Reads results/compute_cost_results.json + results/multiseed_summary.json
+ results/extra_baselines_results.json + results/domain_alignment_results.json
+ results/ensemble_results.json. Writes
results/figs/fig14_pareto.pdf (vector).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})


def safe_load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    cost = safe_load(os.path.join(RESULTS_DIR, 'compute_cost_results.json'))
    if cost is None:
        print('no compute_cost_results.json — run run_compute_cost.py')
        return
    bench = {b['model']: b for b in cost['benchmarks']}

    # Collect (label, infer_ms, f1, ram_mb, family)
    points = []
    for name, b in bench.items():
        if name == 'tree_ensemble':
            family = 'ensemble'
        elif name.startswith(('xgb', 'lightgbm', 'catboost', 'random_forest')):
            family = 'tree'
        else:
            family = 'deep'
        points.append({
            'name': name.replace('_', ' '),
            'infer': b['inference_ms_per_1k_rows'],
            'f1': b['test_f1_macro'],
            'ram': b['peak_ram_mb'],
            'family': family,
        })

    # Re-tag the directly-measured tree_ensemble row (deployment recipe)
    # so it gets the dedicated colour. No synthetic proxy is used: the
    # measurement comes from `run_compute_cost.py:train_tree_ensemble`
    # which trains both stages and times the full averaged-probability
    # inference path end-to-end.
    for p in points:
        if p['name'].strip() in ('tree ensemble', 'tree_ensemble',
                                  'tree-only ensemble'):
            p['family'] = 'ensemble'
            p['name'] = 'tree ensemble'
            break

    # Pull DA / domain-alignment numbers (no inference time available — use
    # MLP ERM proxy)
    da = safe_load(os.path.join(RESULTS_DIR, 'domain_alignment_results.json'))
    if da is not None:
        mlp_infer = bench.get('mlp_erm', {}).get(
            'inference_ms_per_1k_rows', 0.5)
        mlp_ram = bench.get('mlp_erm', {}).get('peak_ram_mb', 65)
        for r in da.get('runs', []):
            points.append({
                'name': r['method'].upper(),
                'infer': mlp_infer,
                'f1': r['test']['f1_macro'],
                'ram': mlp_ram,
                'family': 'domain-align',
            })

    fam_color = {'tree': '#2E86C1', 'deep': '#E67E22',
                 'ensemble': '#C0392B', 'domain-align': '#8E44AD'}

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for fam in ['tree', 'deep', 'ensemble', 'domain-align']:
        sel = [p for p in points if p['family'] == fam]
        if not sel:
            continue
        xs = [p['infer'] for p in sel]
        ys = [p['f1'] for p in sel]
        ss = [max(40, min(700, p['ram'] * 4)) for p in sel]
        ax.scatter(xs, ys, s=ss, alpha=0.65, color=fam_color[fam],
                   edgecolors='black', linewidth=0.5,
                   label=fam.replace('-', ' '))

    ax.set_xscale('log')
    ax.set_xlabel('Inference time per 1{,}000 rows (ms, log scale)')
    ax.set_ylabel('OOD macro-F1 on Chungbuk LORO fold')
    ax.set_title('Cost vs.\\ Accuracy Pareto frontier '
                 '(circle area $\\propto$ peak RAM)')
    ax.axhline(0.4326, color='red', linestyle='--', linewidth=0.7,
               alpha=0.6)
    ax.text(0.99, 0.4326, 'deployment candidate (0.4326)', fontsize=11,
            color='red', va='bottom', ha='right',
            transform=ax.get_yaxis_transform())
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              borderaxespad=0., frameon=True)

    offsets = {
        'xgboost flat':   (-80, -12),
        'xgboost hier':   (-80, -38),
        'lightgbm':       ( 30, -20),
        'tree ensemble':  ( 30, -42),
        'catboost':       ( 30,  20),
        'random forest':  (-15,  32),
        'mlp erm':        ( 45,  15),
        'cnn erm':        ( 38,  24),
        'CORAL':          ( 50,  12),
        'MMD':            ( 50,   5),
        'DANN':           ( 50, -12),
    }
    arrow_style = dict(arrowstyle='-', color='#555555', lw=0.8,
                       shrinkA=0, shrinkB=4)
    for p in points:
        dx, dy = offsets.get(p['name'], (30, 15))
        ax.annotate(
            p['name'], xy=(p['infer'], p['f1']),
            xytext=(dx, dy), textcoords='offset points',
            fontsize=10, ha='left' if dx > 0 else 'right',
            va='center',
            arrowprops=arrow_style,
        )

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig14_pareto.pdf')
    fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
