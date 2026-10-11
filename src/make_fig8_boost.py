"""Figure 8 — Performance-boosting recipes and final ensemble.

Left: CNN augmentation/loss recipe comparison (Phase C).
Right: ensemble of best tree models vs individual models (Phase D).
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figs')


def main():
    aug = json.load(open(os.path.join(RESULTS_DIR,
                                      'augmentation_results.json')))
    ens = json.load(open(os.path.join(RESULTS_DIR, 'ensemble_results.json')))
    cnn_erm_f1 = None
    irm_path = os.path.join(RESULTS_DIR, 'irm_cnn_results.json')
    if os.path.exists(irm_path):
        runs = json.load(open(irm_path))
        # {'seed', 'runs'} from the multi-seed-aware runner, or a bare list
        runs = runs['runs'] if isinstance(runs, dict) else runs
        for run in runs:
            if run.get('method') == 'erm':
                cnn_erm_f1 = run['test']['f1_macro']
                break

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # Left: augmentation recipes
    runs = aug['runs']
    labels = [r['name'].replace('cnn_', '').replace('mlp_causal_', 'MLP-causal\n')
              for r in runs]
    f1s = [r['test_f1'] for r in runs]
    accs = [r['test_acc'] for r in runs]
    x = np.arange(len(runs))
    w = 0.4
    axes[0].bar(x - w / 2, f1s, w, label='Macro-F1', color='#e74c3c')
    axes[0].bar(x + w / 2, accs, w, label='Accuracy', color='#3498db')
    if cnn_erm_f1 is not None:
        axes[0].axhline(cnn_erm_f1, color='#e74c3c', linestyle='--',
                        alpha=0.5,
                        label=f'Phase 2B CNN ERM F1={cnn_erm_f1:.4f}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=13, rotation=20, ha='right')
    axes[0].set_ylim(0, 0.45)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Augmentation + loss recipe sweep\n(CNN on raw 513-band spectrum)')
    axes[0].legend(loc='lower left', fontsize=13)
    for xi, (a, f) in enumerate(zip(accs, f1s)):
        axes[0].text(xi - w / 2, f + 0.005, f'{f:.3f}',
                     ha='center', fontsize=12)
        axes[0].text(xi + w / 2, a + 0.005, f'{a:.3f}',
                     ha='center', fontsize=12)

    # Right: ensemble comparison
    methods = list(ens['ensembles'].keys())
    f1s_e = [ens['ensembles'][m]['f1'] for m in methods]
    accs_e = [ens['ensembles'][m]['acc'] for m in methods]
    xe = np.arange(len(methods))
    axes[1].bar(xe - w / 2, f1s_e, w, label='Macro-F1', color='#e74c3c')
    axes[1].bar(xe + w / 2, accs_e, w, label='Accuracy', color='#3498db')
    axes[1].axhline(0.4287, color='#e74c3c', linestyle='--', alpha=0.5,
                    label='Phase 2D hierarchical F1=0.4287')
    axes[1].set_xticks(xe)
    pretty = {'xgb_causal': 'XGB\ncausal',
              'xgb_hier': 'XGB\nhierarch',
              'cnn_val_selected': 'CNN\n(val-selected)',
              'soft_avg': 'Soft avg\n(3 models)',
              'tree_only_soft_avg': 'Tree only\nsoft avg'}
    axes[1].set_xticklabels([pretty.get(m, m) for m in methods],
                            fontsize=13)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Ensemble on Chungbuk LORO fold')
    axes[1].legend(loc='lower left', fontsize=13)
    for xi, (a, f) in enumerate(zip(accs_e, f1s_e)):
        axes[1].text(xi - w / 2, f + 0.008, f'{f:.3f}',
                     ha='center', fontsize=12,
                     color='red' if f > 0.4287 else 'black')
        axes[1].text(xi + w / 2, a + 0.008, f'{a:.3f}',
                     ha='center', fontsize=12)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig8_boost_ensemble.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
