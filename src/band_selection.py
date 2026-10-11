"""Nested, leakage-free frequency-band selection.

Fixes the selection-leakage defect recorded as Blocker 1 in ``AUDIT.md``:
``run_freq_analysis.py`` ranked band groups by permutation importance computed
*on the held-out region*, and ``run_causal_bands.py`` then evaluated the
resulting 65-422 Hz subset *on that same region*. Band choice therefore saw
the evaluation fold, so the reported causal-band advantage was optimistically
biased even though the final classifier never trained on held-out rows.

The protocol here makes selection and evaluation disjoint by construction:

    for an evaluation region R:
        pool          = every region except R          <- selection sees only this
        inner folds   = leave-one-region-out *within* pool
        band groups   = ranked by mean inner-fold permutation importance
        final model   = trained on pool, evaluated once on R

``select_bands_nested`` returns both the chosen band set and a manifest that
records every region whose rows influenced the choice. ``verify_manifest``
re-checks that the evaluation region is absent from that record, so the
exclusion is machine-checkable rather than a claim in prose.
"""
import json
import os

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# Same log-spaced grouping used by run_robust_bands.py, so selected bands stay
# comparable with the previously reported attribution tables.
BAND_GROUPS = [
    ('10-121', 10, 121),
    ('121-422', 121, 422),
    ('422-1470', 422, 1470),
    ('1470-5120', 1470, 5121),
]

# A region is usable as an evaluation or inner-selection fold only if it
# carries enough of the label space for macro-F1 to mean anything.
MIN_CLASSES = 4
MIN_ROWS = 1000

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective='multi:softprob', num_class=5,
    tree_method='hist', n_jobs=-1, random_state=42,
    eval_metric='mlogloss', verbosity=0,
)


def bands_in_range(lo, hi):
    """HZ column names whose frequency f satisfies lo <= f < hi."""
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]



def _jsonable(v):
    """Environment labels may be numpy ints or strings; normalise for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.str_,)):
        return str(v)
    return v


def _usable_env_values(env_labels, y):
    """Environment values with enough rows and classes to be an inner fold."""
    out = []
    for v in sorted(set(env_labels.tolist())):
        m = env_labels == v
        if m.sum() >= MIN_ROWS and len(np.unique(y[m])) >= MIN_CLASSES:
            out.append(_jsonable(v))
    return out


def usable_regions(df, y, regions=None):
    """Regions with enough rows and label coverage to serve as a fold."""
    out = []
    for r in sorted(df['region_lvl1'].unique().tolist()):
        if regions is not None and r not in regions:
            continue
        mask = df['region_lvl1'].values == r
        if mask.sum() >= MIN_ROWS and len(np.unique(y[mask])) >= MIN_CLASSES:
            out.append(int(r))
    return out


def _macro_f1(clf, X, y, idx):
    return float(f1_score(y[idx], clf.predict(X[idx]), average='macro',
                          zero_division=0))


def _group_importance(clf, X, y, idx, col_idx, seed):
    """Drop in macro-F1 on `idx` when each band group is permuted."""
    base = _macro_f1(clf, X, y, idx)
    rng = np.random.RandomState(seed)
    X_eval = X[idx]
    drops = {}
    for gname, lo, hi in BAND_GROUPS:
        cols = [col_idx[b] for b in bands_in_range(lo, hi) if b in col_idx]
        X_perm = X_eval.copy()
        X_perm[:, cols] = X_eval[rng.permutation(len(idx))][:, cols]
        perm_f1 = float(f1_score(y[idx], clf.predict(X_perm), average='macro',
                                 zero_division=0))
        drops[gname] = base - perm_f1
    return base, drops


def select_bands_nested(df, X, y, cols, eval_region, seed=42, verbose=True,
                        env=None, inner_env=None):
    """Choose frequency bands using only data outside `eval_region`.

    Returns ``(selected_groups, feature_cols, manifest)``. A band group is
    selected when permuting it degrades macro-F1 on average across the inner
    folds -- i.e. the model genuinely depends on it to generalise to an
    environment it has not seen. The rule is fixed in advance and applied
    identically to every evaluation environment; nothing about `eval_region`
    enters it.

    `env` is the per-row environment label defining what is held out
    (defaults to `region_lvl1`, so `eval_region` names a region). `inner_env`
    is the per-row label defining the inner leave-one-out folds; it defaults
    to `env`. Passing a coarser `inner_env` than `env` -- region-level inner
    folds for a county-level evaluation, say -- keeps the inner loop tractable
    without ever letting the evaluation environment into selection, since its
    rows are removed from the pool before the inner split is formed.
    """
    col_idx = {c: i for i, c in enumerate(cols)}
    regions = df['region_lvl1'].values if env is None else np.asarray(env)

    pool_mask = regions != eval_region
    pool_idx = np.where(pool_mask)[0]

    if inner_env is None:
        inner_labels = regions
        inner_regions = [r for r in usable_regions(df, y) if r != eval_region]
    else:
        inner_labels = np.asarray(inner_env)
        inner_regions = _usable_env_values(inner_labels[pool_idx], y[pool_idx])

    if verbose:
        print(f'  selection pool: {len(pool_idx)} rows, '
              f'inner folds: {inner_regions}', flush=True)

    per_fold = {}
    for r in inner_regions:
        inner_te = np.where(pool_mask & (inner_labels == r))[0]
        inner_tr = np.where(pool_mask & (inner_labels != r))[0]
        clf = XGBClassifier(**{**XGB_PARAMS, 'random_state': seed})
        clf.fit(X[inner_tr], y[inner_tr])
        base, drops = _group_importance(clf, X, y, inner_te, col_idx, seed)
        per_fold[str(r)] = {'base_f1': base, 'drops': drops,
                            'n_train': int(len(inner_tr)),
                            'n_test': int(len(inner_te))}
        if verbose:
            summary = '  '.join(f'{g}={drops[g]:+.4f}' for g, _, _ in BAND_GROUPS)
            print(f'    inner r={r:3d} base_f1={base:.4f}  {summary}',
                  flush=True)

    stats = {}
    for gname, _, _ in BAND_GROUPS:
        vals = np.array([per_fold[str(r)]['drops'][gname]
                         for r in inner_regions])
        stats[gname] = {
            'mean_drop': float(vals.mean()),
            'std_drop': float(vals.std()),
            'wins': int((vals > 0).sum()),
            'n_folds': int(len(vals)),
            'per_fold': [float(v) for v in vals],
        }

    selected = [g for g, _, _ in BAND_GROUPS if stats[g]['mean_drop'] > 0]
    if not selected:
        # Degenerate case: keep the single least-harmful group so the fold
        # still produces a model rather than silently dropping out.
        selected = [max(stats, key=lambda g: stats[g]['mean_drop'])]

    band_cols = []
    for gname, lo, hi in BAND_GROUPS:
        if gname in selected:
            band_cols.extend(bands_in_range(lo, hi))
    max_cols = [c for c in cols if c.startswith('MAX')]
    feature_cols = ['month'] + band_cols + max_cols

    manifest = {
        'eval_region': _jsonable(eval_region),
        'selection_input_regions': sorted(
            {_jsonable(r) for r in regions[pool_idx].tolist()}),
        'inner_fold_regions': [_jsonable(r) for r in inner_regions],
        'n_selection_rows': int(len(pool_idx)),
        'selection_rule': 'mean inner-fold permutation macro-F1 drop > 0',
        'band_group_stats': stats,
        'selected_groups': selected,
        'n_selected_features': len(feature_cols),
        'per_fold': per_fold,
    }
    return selected, feature_cols, manifest


def verify_manifest(manifest):
    """Raise if the evaluation region influenced its own band selection."""
    r = manifest['eval_region']
    if r in manifest['selection_input_regions']:
        raise AssertionError(
            f'LEAKAGE: eval region {r} present in selection inputs')
    if r in manifest['inner_fold_regions']:
        raise AssertionError(
            f'LEAKAGE: eval region {r} used as an inner selection fold')
    return True


# ---------------------------------------------------------------------------
# Leakage-free feature set, resolved from the nested selection run.
# ---------------------------------------------------------------------------
_NESTED_RESULTS = os.path.join(
    os.path.dirname(__file__), '..', 'results',
    'nested_causal_bands_results.json')

_MAX_COLS = [f'MAX{i}' for i in range(20)]


def causal_feature_cols(region=43, results_path=None):
    """Feature columns for `region`, as chosen without ever seeing `region`.

    Replaces the hard-coded ``['month'] + bands_in_range(65, 422) + MAX_COLS``
    that appeared across the experiment scripts. That 65-422 Hz subset was
    derived from permutation importance measured *on* region 43 and then
    evaluated on region 43; see ``results/FINDINGS_NESTED_BANDS.md``.

    The band groups are read back from the nested-selection run so the feature
    set is traceable to a fold whose manifest proves the evaluation region was
    excluded, rather than being a constant restated by hand.
    """
    path = results_path or _NESTED_RESULTS
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found — run src/run_nested_causal_bands.py first. '
            'The 65-422 Hz constant it replaces is retracted (selection '
            'leakage); see results/FINDINGS_NESTED_BANDS.md.')
    with open(path) as fh:
        res = json.load(fh)

    fold = next((f for f in res['folds'] if f['region'] == region), None)
    if fold is None:
        available = [f['region'] for f in res['folds']]
        raise ValueError(
            f'no leakage-free band selection for region {region}; '
            f'available: {available}. Run nested selection for this fold '
            'rather than borrowing another region\'s bands.')

    verify_manifest(fold['manifest'])

    band_cols = []
    for gname, lo, hi in BAND_GROUPS:
        if gname in fold['selected_groups']:
            band_cols.extend(bands_in_range(lo, hi))
    return ['month'] + band_cols + _MAX_COLS


def causal_band_provenance(region=43, results_path=None):
    """Human-readable description of where `region`'s bands came from."""
    path = results_path or _NESTED_RESULTS
    with open(path) as fh:
        res = json.load(fh)
    fold = next(f for f in res['folds'] if f['region'] == region)
    return (f"region {region}: groups {fold['selected_groups']} selected from "
            f"inner folds {fold['manifest']['inner_fold_regions']} "
            f"({fold['manifest']['n_selection_rows']} rows, region {region} excluded)")


# ---------------------------------------------------------------------------
# Cache. A nested selection costs one model fit per inner fold, so scripts
# that share an evaluation axis should not each pay for it.
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'nested_band_cache.json')


def _load_cache():
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as fh:
            return json.load(fh)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, 'w') as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


def nested_bands_cached(df, X, y, cols, eval_env, kind='region',
                        env=None, inner_env=None, seed=42, verbose=True):
    """Nested band selection for `eval_env`, memoised across scripts.

    `kind` names the evaluation axis ('region', 'sigungu', 'temporal', ...) and
    only separates cache namespaces; correctness comes from `env`/`inner_env`.
    Returns ``(feature_cols, manifest)``.
    """
    key = f'{kind}:{_jsonable(eval_env)}:seed{seed}'
    cache = _load_cache()
    if key in cache:
        entry = cache[key]
        verify_manifest(entry['manifest'])
        if verbose:
            print(f'  [cache hit] {key}: groups '
                  f"{entry['manifest']['selected_groups']}", flush=True)
        return entry['feature_cols'], entry['manifest']

    selected, feature_cols, manifest = select_bands_nested(
        df, X, y, cols, eval_env, seed=seed, verbose=verbose,
        env=env, inner_env=inner_env)
    verify_manifest(manifest)
    cache = _load_cache()          # re-read: another process may have written
    cache[key] = {'feature_cols': feature_cols, 'manifest': manifest}
    _save_cache(cache)
    return feature_cols, manifest
