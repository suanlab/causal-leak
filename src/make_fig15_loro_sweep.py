"""Figure 15 — Per-LORO-fold detection-rate at calibration-derived
FAR targets {0.5%, 1%, 2%, 5%}.

Replaces the original Chungbuk-only Stage-1 threshold sweep (uninformative)
with a deployment-honest evaluation across all LORO folds with at least
one normal sample. Threshold is selected from a held-out calibration
split of the training pool; the test fold is never touched during
threshold tuning.

Reads results/threshold_sweep_loro_results.json. Writes
results/figs/fig15_loro_sweep.pdf (vector).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

REGION_NAMES = {27: 'Daegu', 29: 'Gwangju', 43: 'Chungbuk',
                44: 'Chungnam', 45: 'Jeonbuk', 46: 'Jeonnam',
                47: 'Gyeongbuk', 48: 'Gyeongnam', 50: 'Jeju'}


def main():
    path = os.path.join(RESULTS_DIR, 'threshold_sweep_loro_results.json')
    if not os.path.exists(path):
        print(f'no {path} — run run_threshold_sweep_loro.py')
        return
    with open(path) as f:
        d = json.load(f)
    folds = d['folds']
    far_targets = d['far_targets']
    far_keys = [f'far_{f:.3f}' for f in far_targets]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Panel A: per-fold detection rate at each cal-FAR target
    n_folds = len(folds)
    width = 0.18
    x = np.arange(n_folds)
    for i, k in enumerate(far_keys):
        det = [f['operating_points'][k]['detection_rate'] for f in folds]
        bars = axes[0].bar(x + (i - 1.5) * width, det, width=width,
                           label=f'cal-FAR={far_targets[i]*100:g}%')
    axes[0].set_xticks(x)
    fold_labels = [f"{f['region']}\n({REGION_NAMES.get(f['region'], '?')}"
                   f"\\,n={f['n_test']/1000:.0f}k\\,n$_{{N}}$={f['n_normal_test']})"
                   for f in folds]
    axes[0].set_xticklabels(fold_labels, fontsize=7)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel('Test-fold detection rate')
    axes[0].set_xlabel('Held-out region (region code, n$_{{N}}$=normal samples)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(loc='lower right', fontsize=7, ncol=2)
    axes[0].set_title('(a) Per-fold detection at calibration-derived FAR',
                       fontsize=9)

    # Panel B: target FAR vs achieved test FAR (calibration faithfulness)
    fars_target = []
    fars_achieved_weighted = []
    fars_achieved_max = []
    fars_achieved_min = []
    for k in far_keys:
        op_list = [f['operating_points'][k] for f in folds]
        ns = np.array([f['n_test'] for f in folds])
        achieved = np.array([o['achieved_far'] for o in op_list])
        wm = float((ns * achieved).sum() / ns.sum())
        fars_target.append(float(k.split('_')[1]))
        fars_achieved_weighted.append(wm)
        fars_achieved_max.append(float(achieved.max()))
        fars_achieved_min.append(float(achieved.min()))

    axes[1].plot([0, 0.06], [0, 0.06], '--', color='gray',
                 linewidth=0.8, label='target = achieved (ideal)')
    axes[1].plot(fars_target, fars_achieved_weighted, 'o-',
                 color='#2E86C1', linewidth=1.2,
                 label=r'weighted mean achieved FAR')
    axes[1].fill_between(fars_target, fars_achieved_min,
                         fars_achieved_max, color='#2E86C1', alpha=0.15,
                         label='min--max across 5 folds')
    axes[1].set_xlabel('Calibration FAR target')
    axes[1].set_ylabel('Test-set achieved FAR')
    axes[1].set_xlim(0.0, 0.06)
    axes[1].set_ylim(0.0, 0.07)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=7)
    axes[1].set_title('(b) Calibration faithfulness on held-out test',
                       fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig15_loro_sweep.pdf')
    fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
