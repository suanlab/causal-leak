"""Figure 11 — Computational cost benchmark (E6).

Bar charts showing training time, inference time per 1k rows,
peak RAM, and peak GPU memory for the top models on the
canonical Chungbuk LORO fold.

Reads results/compute_cost_results.json and writes
results/figs/fig11_cost.pdf (vector PDF).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def main():
    path = os.path.join(RESULTS_DIR, 'compute_cost_results.json')
    if not os.path.exists(path):
        print(f'no {path} — run run_compute_cost.py first')
        return
    with open(path) as f:
        data = json.load(f)

    bench = data.get('benchmarks', [])
    if not bench:
        print('no compute-cost data')
        return

    names = [r['model'] for r in bench]
    train = [r['train_time_s'] for r in bench]
    infer = [r['inference_ms_per_1k_rows'] for r in bench]
    ram = [r['peak_ram_mb'] for r in bench]
    gpu = [r['peak_gpu_mb'] for r in bench]
    f1 = [r['test_f1_macro'] for r in bench]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.8))

    axes[0].barh(range(len(names)), train, color='#1f77b4')
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('train time (s, log)')
    axes[0].set_xscale('log')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].set_title('Training time', fontsize=10)

    axes[1].barh(range(len(names)), infer, color='#2ca02c')
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('infer (ms / 1k rows, log)')
    axes[1].set_xscale('log')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_title('Inference', fontsize=10)

    axes[2].barh(range(len(names)), ram, color='#ff7f0e')
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].set_xlabel('peak RAM (MB)')
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].set_title('Memory (CPU)', fontsize=10)

    axes[3].barh(range(len(names)), f1, color='#d62728')
    axes[3].set_yticks(range(len(names)))
    axes[3].set_yticklabels([])
    axes[3].invert_yaxis()
    axes[3].set_xlabel('OOD macro-F1')
    axes[3].axvline(0.4326, color='black', linestyle='--', linewidth=0.8,
                    alpha=0.6)
    axes[3].grid(axis='x', alpha=0.3)
    axes[3].set_title('OOD F1', fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig11_cost.pdf')
    fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
