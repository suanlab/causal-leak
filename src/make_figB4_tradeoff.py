"""Paper B, Figure 4 — the decision plane (vector PDF).

False-alarm rate against mean macro-F1 across the evaluation folds, with
inference cost shown as marker area. Good configurations sit toward the upper
left: equal typing accuracy at fewer unnecessary dispatches.

The figure exists to make one comparison legible. LightGBM leads on the single
canonical fold but not on the fold-averaged score, and it raises three times as
many false alarms, so it does not dominate the recommended ensemble on any axis
the deployment argument uses.

Reads results/operational_folds_results.json and
results/compute_cost_results.json.
Writes results/figs/figB4_tradeoff.pdf.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'results', 'figs')
os.makedirs(FIGS, exist_ok=True)

MODELS = [
    ('xgb_hier', 'xgboost_hier', 'GBT hierarchical', '#92c5de'),
    ('tree_only_soft_avg', 'tree_ensemble', 'Tree-only soft average', '#2166ac'),
    ('xgb_flat', 'xgboost_flat', 'GBT flat', '#f4a582'),
    ('lightgbm', 'lightgbm', 'LightGBM', '#b2182b'),
]


def main():
    op = json.load(open(os.path.join(R, 'operational_folds_results.json')))
    cc = {b['model']: b for b in
          json.load(open(os.path.join(R, 'compute_cost_results.json')))['benchmarks']}
    s = op['summary_over_far_estimable_folds']
    # Typing is measurable on every fold, so macro-F1 is averaged over all
    # five; the false-alarm rate uses only the folds able to estimate it.
    f1_all = {k: float(np.mean([f['models'][k]['f1_macro'] for f in op['folds']]))
              for k, _, _, _ in MODELS}

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, cost_key, label, colour in MODELS:
        far = s[key]['far_weighted'] * 100
        f1 = f1_all[key]
        ms = cc[cost_key]['inference_ms_per_1k_rows']
        ax.scatter(far, f1, s=ms * 22, color=colour, edgecolor='black',
                   linewidth=0.8, zorder=3, alpha=0.9)
        ax.annotate(f'{label}\n{ms:.1f} ms/1k', (far, f1),
                    textcoords='offset points', xytext=(11, -4),
                    fontsize=7.5, va='center')

    ax.set_xlabel('false-alarm rate (%) over 22,121 normal recordings, 4 folds',
                  fontsize=9)
    ax.set_ylabel('macro-F1, mean over 5 folds', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    xs = [s[k]['far_weighted'] * 100 for k, _, _, _ in MODELS]
    ys = [f1_all[k] for k, _, _, _ in MODELS]
    ax.set_xlim(-0.012, max(xs) * 1.75)
    pad = (max(ys) - min(ys)) * 0.9 + 0.002
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.annotate('better', xy=(0.06, 0.93), xycoords='axes fraction',
                fontsize=8, style='italic', color='#555555')
    ax.annotate('', xy=(0.02, 0.97), xytext=(0.14, 0.88),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
    ax.text(0.98, 0.03, 'marker area $\\propto$ inference cost',
            transform=ax.transAxes, ha='right', fontsize=7,
            color='#555555')

    fig.tight_layout()
    out = os.path.join(FIGS, 'figB4_tradeoff.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')
    for key, ck, label, _ in MODELS:
        print(f"  {label:<24} FAR={s[key]['far_weighted']*100:.4f}%  "
              f"F1={f1_all[key]:.4f}  "
              f"{cc[ck]['inference_ms_per_1k_rows']:.2f} ms/1k")


if __name__ == '__main__':
    main()
