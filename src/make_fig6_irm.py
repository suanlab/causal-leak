"""Phase 2B figure — compare XGBoost baselines to MLP / CNN with
ERM / IRM / GroupDRO on LORO region 43."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import LABEL_NAMES  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figs')


def main():
    # Baselines from Phase 2C causal-bands experiment
    with open(os.path.join(RESULTS_DIR, 'causal_bands_results.json')) as f:
        cb = json.load(f)
    cb_lookup = {d['config']: d for d in cb}
    xgb_sig = cb_lookup['signal_all']
    xgb_causal = cb_lookup['causal_65_422Hz']

    # The multi-seed-aware runners wrap their run list as {'seed', 'runs'};
    # artefacts written before that change are bare lists. Accept both.
    with open(os.path.join(RESULTS_DIR, 'irm_mlp_results.json')) as f:
        mlp = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'irm_cnn_results.json')) as f:
        cnn = json.load(f)
    mlp = mlp['runs'] if isinstance(mlp, dict) else mlp
    cnn = cnn['runs'] if isinstance(cnn, dict) else cnn

    rows = [
        ('XGB\nsignal_all', xgb_sig['acc'], xgb_sig['f1_macro'],
         xgb_sig['f1_per_class']),
        ('XGB\ncausal', xgb_causal['acc'], xgb_causal['f1_macro'],
         xgb_causal['f1_per_class']),
    ]
    for r in mlp:
        rows.append((f'MLP\n{r["method"]}', r['test']['acc'],
                     r['test']['f1_macro'], r['test']['f1_per_class']))
    for r in cnn:
        rows.append((f'CNN\n{r["method"]}', r['test']['acc'],
                     r['test']['f1_macro'], r['test']['f1_per_class']))

    labels = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    f1s = [r[2] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    x = np.arange(len(rows))
    w = 0.38
    axes[0].bar(x - w / 2, accs, w, color='#2E86AB', label='Accuracy')
    axes[0].bar(x + w / 2, f1s, w, color='#E63946', label='Macro-F1')
    best = max(f1s)
    axes[0].axhline(best, linestyle='--', color='#E63946', alpha=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=13)
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 0.5)
    axes[0].set_title('Phase 2B — IRM / GroupDRO vs. Causal XGBoost '
                      '(LORO region 43)')
    axes[0].legend(loc='upper right')
    for xi, (a, f) in enumerate(zip(accs, f1s)):
        axes[0].text(xi - w / 2, a + 0.008, f'{a:.3f}',
                     ha='center', fontsize=11)
        axes[0].text(xi + w / 2, f + 0.008, f'{f:.3f}',
                     ha='center', fontsize=11)

    mat = np.array([r[3] for r in rows])
    im = axes[1].imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(LABEL_NAMES)
    axes[1].set_yticks(range(len(rows)))
    axes[1].set_yticklabels([lbl.replace('\n', ' ') for lbl in labels],
                            fontsize=13)
    axes[1].set_title('Per-class F1')
    for i in range(len(rows)):
        for j in range(5):
            axes[1].text(j, i, f'{mat[i, j]:.2f}',
                         ha='center', va='center', fontsize=11,
                         color='white' if mat[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig6_irm_comparison.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
