"""Test statistics of the audit protocol, shared by every script that uses them.

A1  lift_test        lift of a majority-label lookup on a unit over the
                     largest class share, against a null that permutes the
                     units' lookup labels across units. Rejects when the
                     permutation p <= 0.05 AND the lift reaches a minimum
                     effect (default 0.25): with thousands of units the test
                     alone rejects lifts near zero.
A2  identify_threshold  T1: every threshold T for which a logged device field
                     equals scale * #{checks with peak >= T} on the rows that
                     carry a measured (non-code) value. Unique solution = pass.
    identify_k       the range of k that leaves the coded rows' partition
                     unchanged at the identified threshold (the boundary gap).
    depletion_test   T3, a support diagnostic: ratio of the mass in a
                     boundary bin, fixed in advance from the logged code, to
                     the smaller of its two neighbours (and of the bin below
                     an inclusion floor to the first kept bin). Passes when
                     the one-sided 95 % upper bound of the ratio is below 0.5.
                     The bound is the larger of a cluster-bootstrap bound
                     (resampling sensors or files) and an exact Poisson bound
                     on the boundary count over the reference: a cluster
                     bootstrap of an empty bin returns zero in every
                     resample, so on its own it would claim certainty about
                     a bin that was simply not observed. A pass shows the
                     release is depleted at the boundary, which selection
                     would produce; it does not identify the mechanism.

T2 (sufficiency) compares a fitted classifier with the rule and lives with the
model code that fits it.
"""
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- A1
def lift_test(train_units, train_y, test_units, test_y, n_perm=1000, rng=None,
              min_lift=0.25, alpha=0.05):
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
            'n_units': int(len(units)), 'min_lift': min_lift,
            'reject': bool((1 + (null >= L).sum()) / (1 + n_perm) <= alpha
                           and L >= min_lift)}


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
def depletion_test(g, clusters, boundary=None, floor=None, max_ratio=0.5,
                   n_boot=2000, rng=None):
    """T3 on a count-valued device statistic g with recording clusters.

    boundary: interior bin b, fixed in advance from the logged code; ratio
              n[b] / min(n[b-1], n[b+1]).
    floor:    first bin f the release keeps; ratio n[f-1] / n[f].
    Each result gives observed and reference mass, the ratio, its one-sided
    95 % upper bound from resampling clusters with replacement, 'pass' (upper
    bound below max_ratio), and a one-sided Poisson p for reference.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    g = np.asarray(g).astype(int)
    clusters = np.asarray(clusters).astype(str)
    keys, inv = np.unique(clusters, return_inverse=True)
    H = np.zeros((len(keys), max(11, g.max() + 2)), dtype=float)
    np.add.at(H, (inv, g), 1)                      # clusters x bins
    h = H.sum(0)
    picks = rng.integers(0, len(keys), (n_boot, len(keys)))
    boot = np.stack([H[p].sum(0) for p in picks])  # n_boot x bins

    def one(b_obs, b_refs):
        obs = h[b_obs]
        ref = min(h[r] for r in b_refs)
        with np.errstate(divide='ignore', invalid='ignore'):
            rb = boot[:, b_obs] / np.min(boot[:, b_refs], axis=1)
        rb = np.where(np.isfinite(rb), rb, np.inf)
        boot_upper = float(np.quantile(rb, 0.95))
        # Exact one-sided 95 % Poisson upper limit of the boundary count.
        exact_upper = float(stats.chi2.ppf(0.95, 2 * (obs + 1)) / 2 / ref) if ref > 0 else np.inf
        upper = max(boot_upper, exact_upper)
        return {'bin': int(b_obs), 'observed': int(obs), 'expected': float(ref),
                'ratio': float(obs / ref) if ref > 0 else None,
                'ratio_upper95': upper, 'ratio_upper95_bootstrap': boot_upper,
                'ratio_upper95_exact': exact_upper, 'pass': bool(upper < max_ratio),
                'p_value': float(stats.poisson.cdf(obs, ref)) if ref > 0 else None}

    out = {'n_clusters': int(len(keys)), 'max_ratio': max_ratio}
    if boundary is not None:
        out['boundary'] = one(boundary, [boundary - 1, boundary + 1])
    if floor is not None:
        out['floor'] = one(floor - 1, [floor])
    return out
