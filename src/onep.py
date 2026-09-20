"""
CAMELS 1P set support: one parameter varied at a time, others held at
fiducial. The point of 1P is to check whether a parameter that's null in
the LH quartile-contrast test (marginalized over the other 5, with seed
noise) is truly null, or just swamped by that marginalization.

Directory convention (verified against real CAMELS-IllustrisTNG 1P data,
including a double-digit index like `1P_p10_1`): after stripping the `1P_`
prefix, labels are

    "0"           the single shared fiducial point, reused across every
                  parameter's sweep (all parameters sit at their fiducial
                  value here) -- not its own separate parameter group
    "p<N>_<M>"    parameter N, positive step M
    "p<N>_n<M>"   parameter N, negative step M (e.g. "p1_n2" -> step -2)

The exact grid size, step spacing, whether multiple seeds exist per step,
and how many parameters are varied are release-specific and not hardcoded
here. In particular, a 1P set need not vary the same 6 parameters as the LH
set at all -- the CAMELS-IllustrisTNG 1P set actually on disk here varies
28 astrophysics parameters (WindEnergyIn1e51erg, RadioFeedbackFactor, ...),
a finer decomposition of feedback than the LH set's 4 lumped amplitudes,
with only the 2 cosmological columns corresponding directly. See
`infer_1p_parameter_names`, which discovers columns from the parameter
table itself rather than assuming `config.ALL_PARAMS`, and notebook 04,
which lists what's actually on disk before running anything.
"""

import re

import numpy as np
import pandas as pd

# Only these correspond directly to LH's parameter names; used solely to
# cross-check the 1P set's cosmological columns against the LH sanity
# check, never to force-map the astrophysics columns.
COSMO_ALIASES = {"Omega0": "Omega_m", "sigma8": "sigma_8"}

_LABEL_RE = re.compile(r"^p(\d+)_(n)?(\d+)$")


def parse_1p_label(label):
    """
    "p1_3" -> (1, 3), "p1_n2" -> (1, -2), "p10_1" -> (10, 1):
    (param_index, step_index).
    "0" (the shared fiducial) -> (None, 0) -- it belongs to every
    parameter's group, not a parameter of its own.
    Raises ValueError on anything else.
    """
    if label == "0":
        return None, 0

    m = _LABEL_RE.match(label)
    if not m:
        raise ValueError(
            f"'{label}' is not a 1P label of the form 'p<N>_<M>', 'p<N>_n<M>', or '0'"
        )
    param_index, negative, step = m.groups()
    step = int(step)
    if negative:
        step = -step
    return int(param_index), step


def step_label(param_index, step):
    """
    Reconstruct the on-disk label for a non-zero (param_index, step):
    "p<N>_<M>" for positive steps, "p<N>_n<M>" for negative. Step 0 (the
    shared fiducial) has no single canonical label to reconstruct -- use
    `fiducial_value` to look up a parameter's value there instead.
    """
    if step > 0:
        return f"p{param_index}_{step}"
    if step < 0:
        return f"p{param_index}_n{abs(step)}"
    raise ValueError("step 0 is the shared fiducial; use fiducial_value(), not step_label()")


def fiducial_value(theta_1p, name):
    """
    The shared fiducial's value for column `name`, read from `theta_1p`
    (from `params.load_1p_params`) without assuming the parameter table
    has an explicit row labeled "0".

    This matters because the *simulations* dedupe the fiducial to one
    shared run (one snapshot directory, "1P_0"), but the *parameter
    table* documenting each named recipe isn't guaranteed to mirror that:
    some CAMELS releases instead list a separate fiducial row per
    parameter (e.g. "p1_0", "p2_0", ...), so a literal `theta_1p.loc["0"]`
    lookup can raise KeyError even though the table clearly implies a
    fiducial value throughout. Since every row except `name`'s own
    handful of swept rows sits at the fiducial value for that column, the
    fiducial value is simply the most common ("mode") value in the column
    -- robust to whichever row-labeling convention the table actually
    uses.
    """
    counts = theta_1p[name].value_counts()
    if len(counts) == 0:
        raise ValueError(f"'{name}' has no values to determine a fiducial from")
    if len(counts) > 1 and counts.iloc[1] == counts.iloc[0]:
        raise ValueError(
            f"'{name}' has no single dominant value ({counts.iloc[0]} rows "
            f"tied at the top) -- can't identify its fiducial value unambiguously"
        )
    return counts.index[0]


def infer_1p_parameter_names(theta_1p, tol=1e-8, exclude=("seed",)):
    """
    For each param_index group in `theta_1p` (indexed by 1P labels), find
    which single column actually varies across that group -- discovered
    from `theta_1p`'s own columns, not assumed to be any particular fixed
    list, since a 1P set's parameterization can differ entirely from the
    LH set's (see module docstring).

    Parameters
    ----------
    theta_1p : DataFrame indexed by 1P labels (from `params.load_1p_params`)
    tol      : a column with range <= tol within a group is "not varying"
    exclude  : columns never considered as candidates (e.g. the random seed,
               which varies but isn't a physics parameter)

    Returns
    -------
    mapping   : dict {param_index: column_name}, only for groups that
                cleanly match exactly one varying column. The shared
                fiducial ("0", param_index None) is never a key here.
    ambiguous : dict {param_index: [varying_column_names]} for any group
                that matched zero or more-than-one columns -- inspect these
                by hand rather than trusting the inferred mapping

    Non-numeric columns (e.g. a leftover text label) can never be a physics
    parameter, so they're dropped from consideration up front rather than
    raising deep inside a `max() - min()` -- `params.load_1p_params` already
    drops the raw label column it derives `sim_id` from, but this is a
    second line of defense for any other non-numeric column a release might
    include.
    """
    non_numeric = [c for c in theta_1p.columns if not pd.api.types.is_numeric_dtype(theta_1p[c])]
    candidates = [c for c in theta_1p.columns if c not in exclude and c not in non_numeric]

    parsed = {label: parse_1p_label(label) for label in theta_1p.index}
    param_indices = sorted({p for p, _ in parsed.values() if p is not None})
    fiducial_labels = [lab for lab, (p, _) in parsed.items() if p is None]

    mapping = {}
    ambiguous = {}

    for pidx in param_indices:
        group_labels = [lab for lab, (p, _) in parsed.items() if p == pidx]
        group_labels = group_labels + fiducial_labels  # shared center point
        group = theta_1p.loc[group_labels, candidates]

        ranges = group.max() - group.min()
        varying = list(ranges[ranges > tol].index)

        if len(varying) == 1:
            mapping[pidx] = varying[0]
        else:
            ambiguous[pidx] = varying

    return mapping, ambiguous


def group_steps(sim_ids, param_index):
    """
    Given a 1P run's `sim_ids` and a target `param_index`, return the
    subset of labels belonging to that parameter PLUS the shared fiducial
    label (if present in `sim_ids`), sorted by step_index.
    """
    parsed = {s: parse_1p_label(s) for s in sim_ids}
    labels = [
        s for s, (p, _) in parsed.items() if p == param_index or p is None
    ]
    return sorted(labels, key=lambda s: parsed[s][1])


def mean_cdf_by_step(summaries, sim_ids, param_index, n_k, n_r):
    """
    For one target parameter, average `summaries` over any repeated steps
    (multiple seeds at the same step_index, if present) and return the
    per-step mean CDFs alongside their step indices. The shared fiducial
    (label "0", if present in `sim_ids`) is included as every parameter's
    step-0 point.

    Parameters
    ----------
    summaries   : (n_sims, n_k*n_r) kNN-CDF summaries, aligned to sim_ids
    sim_ids     : 1P labels, same order as summaries' rows
    param_index : which parameter (per parse_1p_label) to group by

    Returns
    -------
    step_indices : sorted ndarray of distinct step indices for this parameter
    mean_cdfs    : ndarray (n_steps, n_k, n_r), mean summary at each step
    """
    sim_ids = np.asarray(sim_ids)
    parsed = [parse_1p_label(s) for s in sim_ids]
    steps = np.array([s for _, s in parsed])
    is_param = np.array([p == param_index or p is None for p, _ in parsed])

    step_indices = np.sort(np.unique(steps[is_param]))
    mean_cdfs = np.empty((len(step_indices), n_k, n_r))

    for i, step in enumerate(step_indices):
        rows = is_param & (steps == step)
        mean_cdfs[i] = summaries[rows].mean(axis=0).reshape(n_k, n_r)

    return step_indices, mean_cdfs


def monotonic_trend(step_values, response_at_r):
    """
    Spearman correlation between a 1P parameter's step values and a scalar
    response (e.g. mean-CDF value at one r, for one k), as a cheap,
    ordered-data-appropriate stand-in for the permutation test LH uses --
    a handful of ordered grid points isn't where a shuffle-label null makes
    sense the way it does at 1000 simulations. With no repeated seeds per
    step in a typical 1P release, each point is a single noisy realization,
    so treat a significant result here as a lead, not a confirmed detection
    at LH's standard of rigor.

    Returns (rho, p_value). Needs at least 3 distinct steps.
    """
    from scipy.stats import spearmanr

    if len(step_values) < 3:
        raise ValueError("need at least 3 steps for a trend test")
    return spearmanr(step_values, response_at_r)
