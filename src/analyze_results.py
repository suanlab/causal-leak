"""Post-experiment analysis: summary table, per-class F1, confusion matrices.

Reads results/matrix_results.json and writes:
  - matrix_summary.md       (headline table)
  - per_class_f1.md         (class-level breakdown)
  - confusion_matrices.txt  (readable CMs for each cell)
"""
import json
import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
LABELS = ['in', 'out', 'noise', 'other', 'normal']


def load():
    with open(os.path.join(RESULTS_DIR, 'matrix_results.json')) as f:
        return json.load(f)


def headline_table(results):
    lines = [
        '# Shortcut Learning 2x2 Experiment — Headline',
        '',
        '| Feature Set | Split Protocol | # Feat | Accuracy | Macro-F1 |',
        '|-------------|----------------|-------:|---------:|---------:|',
    ]
    order = ['paper_split', 'random', 'group_site']
    key = lambda r: (r['feat_mode'] != 'full', order.index(r['split_mode']))
    for r in sorted(results, key=key):
        lines.append(
            f'| {r["feat_mode"]} | {r["split_mode"]} | {r["n_features"]} '
            f'| {r["acc_mean"]:.4f} ± {r["acc_std"]:.4f} '
            f'| {r["f1_mean"]:.4f} ± {r["f1_std"]:.4f} |'
        )
    lines.append('')
    lines.append('**Read as**:')
    lines.append('- `paper_split` = original train/val split (shortcut-friendly)')
    lines.append('- `random`      = 5-fold row-level CV')
    lines.append('- `group_site`  = 5-fold stratified GroupKFold by site '
                 '(OOD: test sites unseen at train time)')
    return '\n'.join(lines) + '\n'


def per_class_table(results):
    lines = [
        '# Per-Class Macro-F1 (mean over folds)',
        '',
        '| Feat | Split | ' + ' | '.join(LABELS) + ' |',
        '|------|-------|' + '|'.join(['-----:'] * len(LABELS)) + '|',
    ]
    for r in results:
        per_class = np.array([f['f1_per_class'] for f in r['folds']])
        means = per_class.mean(axis=0)
        row = (f'| {r["feat_mode"]} | {r["split_mode"]} | '
               + ' | '.join(f'{v:.3f}' for v in means) + ' |')
        lines.append(row)
    return '\n'.join(lines) + '\n'


def confusion_dump(results):
    out = []
    for r in results:
        cm_sum = np.sum([np.array(f['cm']) for f in r['folds']], axis=0)
        total = cm_sum.sum()
        out.append(f'=== {r["feat_mode"]} / {r["split_mode"]} '
                   f'(acc={r["acc_mean"]:.4f}, n={int(total)}) ===')
        hdr = '         ' + ' '.join(f'{l:>7}' for l in LABELS)
        out.append(hdr)
        for i, label in enumerate(LABELS):
            row = f'true {label:>6} ' + ' '.join(
                f'{cm_sum[i, j]:7d}' for j in range(len(LABELS)))
            out.append(row)
        out.append('')
    return '\n'.join(out)


def main():
    results = load()
    with open(os.path.join(RESULTS_DIR, 'matrix_summary.md'), 'w') as f:
        f.write(headline_table(results))
    with open(os.path.join(RESULTS_DIR, 'per_class_f1.md'), 'w') as f:
        f.write(per_class_table(results))
    with open(os.path.join(RESULTS_DIR, 'confusion_matrices.txt'), 'w') as f:
        f.write(confusion_dump(results))
    print('Analysis written to results/')
    print()
    print(headline_table(results))


if __name__ == '__main__':
    main()
