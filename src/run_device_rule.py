"""Can normal-vs-rest be separated by the sensor's own threshold rule?

The AI Hub page for this dataset defines `lrate` as the share of 10 daily leak
checks (two-hour early-morning window) whose response exceeded a fixed
threshold. The 20 MAX columns are, alternately, the peak frequency (even
index) and peak amplitude (odd index) of each of those 10 checks. Every
in/out/noise/other recording has lrate = 90 and every normal one 30-70, so
class membership is conditioned on how often the device's threshold fired.

This script fits no model. It scores the rule

    non-normal  iff  #{checks with peak amplitude >= T} >= k

with (T, k) chosen on the training regions only, by balanced accuracy, under
leave-one-region-out; then applies it unchanged to the held-out region. It
also reports how closely the count recovers lrate itself.

The rule uses only MAX columns, which are in the 534-column signal set, so any
learned normal-vs-rest gate on that set has it available.

Output: results/device_rule_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
NORMAL = 4
AMP_COLS = [f'MAX{i}' for i in range(1, 20, 2)]     # odd index = amplitude
T_GRID = np.arange(20, 1001, 10)
K_GRID = np.arange(1, 11)


def counts(amp, T):
    return (amp[:, None, :] >= T[None, :, None]).sum(2)        # n x len(T)


def balanced_acc(is_norm, flagged):
    tpr = flagged[~is_norm].mean(0)                              # non-normal flagged
    tnr = (~flagged[is_norm]).mean(0)                            # normal not flagged
    return (tpr + tnr) / 2


def fit_rule(amp, is_norm):
    c = counts(amp, T_GRID)                                      # n x T
    best = (-1, None, None)
    for k in K_GRID:
        ba = balanced_acc(is_norm, c >= k)
        j = int(ba.argmax())
        if ba[j] > best[0]:
            best = (float(ba[j]), int(T_GRID[j]), int(k))
    return best


def score(amp, is_norm, sid, T, k):
    flagged = (amp >= T).sum(1) >= k
    fa = is_norm & flagged
    miss = ~is_norm & ~flagged
    return {
        'n_normal': int(is_norm.sum()), 'n_rest': int((~is_norm).sum()),
        'false_flags': int(fa.sum()), 'missed': int(miss.sum()),
        'normal_sensors': len(set(sid[is_norm].tolist())),
        'false_flag_sensors': len(set(sid[fa].tolist())),
        'acc': float((flagged != is_norm).mean()),
    }


def main():
    df = load_all()
    y = df['label'].to_numpy()
    is_norm = y == NORMAL
    amp = df[AMP_COLS].to_numpy(dtype=np.float32)
    reg = df['region_lvl1'].to_numpy()
    sid = df['sid'].astype(str).to_numpy()
    lrate = df['lrate'].to_numpy()

    folds = []
    for r in sorted(set(reg.tolist())):
        tr, te = reg != r, reg == r
        ba, T, k = fit_rule(amp[tr], is_norm[tr])
        rec = {'region': int(r), 'T': T, 'k': k, 'train_balanced_acc': ba,
               **score(amp[te], is_norm[te], sid[te], T, k)}
        folds.append(rec)
        print(f"region {r}: T={T} k={k}  normal {rec['n_normal']:>6} "
              f"false {rec['false_flags']:>3} ({rec['false_flag_sensors']} sensors)"
              f"  rest {rec['n_rest']:>6} missed {rec['missed']:>4}", flush=True)

    tot = {k: sum(f[k] for f in folds)
           for k in ('n_normal', 'n_rest', 'false_flags', 'missed')}
    far = [f for f in folds if f['region'] in (27, 29, 44, 45)]
    tot_far = {k: sum(f[k] for f in far) for k in
               ('n_normal', 'n_rest', 'false_flags', 'missed', 'normal_sensors',
                'false_flag_sensors')}

    # How closely does the count recover lrate? Use the pooled rule threshold.
    ba, T_all, k_all = fit_rule(amp, is_norm)
    hits = (amp >= T_all).sum(1) * 10
    norm_le = float((hits[is_norm] <= lrate[is_norm]).mean())
    norm_eq = float((hits[is_norm] == lrate[is_norm]).mean())

    out = {
        'rule': 'non-normal iff #(per-check peak amplitude >= T) >= k',
        'amp_cols': AMP_COLS, 'folds': folds, 'pooled_loro': tot,
        'far_regions_loro': tot_far,
        'pooled_fit': {'T': T_all, 'k': k_all, 'balanced_acc': ba},
        'lrate_recovery_at_pooled_T': {
            'normal_hits_le_lrate': norm_le, 'normal_hits_eq_lrate': norm_eq},
    }
    with open(os.path.join(RES, 'device_rule_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nLORO pooled: false flags {tot['false_flags']}/{tot['n_normal']}, "
          f"missed {tot['missed']}/{tot['n_rest']}")
    print(f"FAR regions: false flags {tot_far['false_flags']}/{tot_far['n_normal']}"
          f" ({tot_far['false_flag_sensors']}/{tot_far['normal_sensors']} sensors),"
          f" missed {tot_far['missed']}/{tot_far['n_rest']}")
    print(f"pooled fit T={T_all} k={k_all} bal.acc={ba:.4f}; normal hits<=lrate "
          f"{norm_le:.4f}, ==lrate {norm_eq:.4f}")


if __name__ == '__main__':
    main()
