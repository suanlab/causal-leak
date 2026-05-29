"""Figure 10 — Hierarchical Stage-1 threshold sweep (E7).

Plots missed-leak rate, false-alarm rate, and downstream 4-class
macro-F1 as a function of the Stage-1 binary-gate threshold T on
the Chungbuk LORO fold.

Reads results/cost_metrics_results.json and writes
results/figs/fig10_threshold.pdf (vector PDF).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def main():
    path = os.path.join(RESULTS_DIR, 'cost_metrics_results.json')
    if not os.path.exists(path):
        print(f'no {path} — run run_cost_metrics.py first')
        return
    with open(path) as f:
        data = json.load(f)

    sweep = data.get('e7_threshold_sweep', [])
    if not sweep:
        print('no E7 data')
        return

    T = [r['threshold'] for r in sweep]
    miss = [r['missed_leak_rate'] for r in sweep]
    far = [r['false_alarm_rate'] for r in sweep]
    f1_4 = [r['f1_macro_4cls_on_abnormal'] for r in sweep]
    f1_5 = [r['f1_macro_5cls'] for r in sweep]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
    ax1.plot(T, miss, 'o-', color='#d62728', label='missed-leak rate')
    ax1.plot(T, far, 's-', color='#1f77b4', label='false-alarm rate')
    ax1.set_xlabel('Stage-1 threshold $T$')
    ax1.set_ylabel('Rate', color='black')
    ax1.set_ylim(-0.02, 0.40)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(T, f1_4, '^-', color='#2ca02c', label='4-class macro-F1')
    ax2.plot(T, f1_5, 'd-', color='#9467bd', label='5-class macro-F1')
    ax2.set_ylabel('Macro-F1')
    ax2.set_ylim(0.20, 0.55)
    ax2.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig10_threshold.pdf')
    fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
