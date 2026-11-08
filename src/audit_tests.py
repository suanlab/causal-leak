"""Test statistics of the audit protocol, shared by every script that uses them.

A1  lift_test        lift of a majority-label lookup on a unit over the
                     largest class share, against a null that permutes the
                     units' lookup labels across units.
A2  identify_threshold  T1: every threshold T for which a logged device field
                     equals scale * #{checks with peak >= T} on the rows that
                     carry a measured (non-code) value. Unique solution = pass.
    identify_k       the range of k that leaves the coded rows' partition
                     unchanged at the identified threshold (the boundary gap).
    depletion_test   T3: mass in a boundary bin of the count histogram against
                     the smaller of its two neighbours (one-sided Poisson), and
                     below an inclusion floor against the first kept bin. The
                     smaller neighbour is a conservative reference: a censored
                     edge bin (count 0) cannot make an ordinary bin look
                     depleted.

T2 (sufficiency) compares a fitted classifier with the rule and lives with the
model code that fits it.
"""
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- A1
def lift_test(train_units, train_y, test_units, test_y, n_perm=1000, rng=None):
    """Lift L = (a - p_max) / (1 - p_max) of a majority-label lookup.

    a is the lookup's accuracy on the test rows, with rows whose unit never
    occurs in training counted as errors; p_max is the largest class share
    among test rows. The null permutes the units' majority labels across
    units, keeping unit sizes. Returns L, the null mean and 95th percentile,
    the one-sided permutation p-value, and the share of test rows whose unit
    occurs in training (the lookup informs only through these).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    train_units = np.asarray(train_units).astype(str)
    test_units = np.asarray(test_units).astype(str)
    train_y, test_y = np.asarray(train_y), np.asarray(test_y)
    classes = np.unique(np.concatenate([train_y, test_y]))
    cidx = {c: i for i, c in enumerate(classes)}
    units = np.unique(train_units)
    uidx = {u: i for i, u in enumerate(units)}
    counts = np.zeros((len(units), len(classes)), dtype=int)
    np.add.at(counts, ([uidx[u] for u in train_units], [cidx[c] for c in train_y]), 1)
    majority = counts.argmax(1)                         # label index per unit
    # Test rows as a units x classes table; unseen units go to a sink row.
    seen = np.array([u in uidx for u in test_units])
    tab = np.zeros((len(units), len(classes)), dtype=int)
    np.add.at(tab, ([uidx[u] for u in test_units[seen]],
                    [cidx[c] for c in test_y[seen]]), 1)
    n = len(test_y)
    p_max = float(np.bincount([cidx[c] for c in test_y]).max() / n)

    def acc(lab):
        return tab[np.arange(len(units)), lab].sum() / n

    to_l = lambda a: (a - p_max) / (1 - p_max)  # noqa: E731
    L = float(to_l(acc(majority)))
    null = np.array([to_l(acc(rng.permutation(majority))) for _ in range(n_perm)])
    return {'lift': L, 'lookup_accuracy': float(acc(majority)), 'p_max': p_max,
            'null_mean': float(null.mean()), 'null_q95': float(np.quantile(null, 0.95)),
            'p_value': float((1 + (null >= L).sum()) / (1 + n_perm)),
            'n_perm': int(n_perm), 'overlap': float(seen.mean()),
            'n_units': int(len(units))}


# ---------------------------------------------------------------- A2 / T1
def counts_at(amp, T):
    return (np.asarray(amp) >= T).sum(1)


def identify_threshold(amp, logged, code, scale=10, grid=np.arange(1, 2001)):
    """T1: all T with logged == scale * count on every row whose logged value
    is below the code; an empty list means no identity exists."""
    amp, logged = np.asarray(amp), np.asarray(logged)
    m = logged < code
    if not m.any():
        return []
    a, l = amp[m], logged[m]
    return [int(T) for T in grid if np.array_equal(counts_at(a, T) * scale, l)]


def identify_k(amp, logged, code, T):
    """k_lo = 1 + largest count among measured rows; k_hi = largest k that
    leaves the same number of coded rows below it (the boundary gap)."""
    c = counts_at(amp, T)
    logged = np.asarray(logged)
    k_lo = int(c[logged < code].max()) + 1
    below = lambda k: int((c[logged >= code] < k).sum())  # noqa: E731
    k_hi = max(k for k in range(k_lo, amp.shape[1] + 1) if below(k) == below(k_lo))
    return k_lo, k_hi


# ---------------------------------------------------------------- A2 / T3
def depletion_test(hist, boundary=None, floor=None):
    """T3 on a count histogram (index = count).

    boundary: interior bin b; expected mass min(n[b-1], n[b+1]).
    floor:    first bin f whose recordings the release keeps; the bin below
              it is compared with n[f].
    Each returns observed, expected, ratio and the one-sided Poisson
    probability of observing at most that many under the expectation.
    """
    h = np.asarray(hist, dtype=float)
    out = {}

    def res(obs, exp):
        return {'observed': int(obs), 'expected': float(exp),
                'ratio': float(obs / exp) if exp > 0 else None,
                'p_value': float(stats.poisson.cdf(obs, exp)) if exp > 0 else None}

    if boundary is not None:
        b = boundary
        out['boundary'] = {'bin': int(b), **res(h[b], min(h[b - 1], h[b + 1]))}
    if floor is not None:
        f = floor
        out['floor'] = {'bin': int(f - 1), **res(h[f - 1], h[f])}
    return out
