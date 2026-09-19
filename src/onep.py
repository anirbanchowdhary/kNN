"""
CAMELS 1P set support: one parameter varied at a time, others held at
fiducial. The point of 1P is to check whether a parameter that's null in
the LH quartile-contrast test (marginalized over the other 5, with seed
noise) is truly null, or just swamped by that marginalization.

Directory convention: `1P_p<param_index>_<step_index>` (e.g. `1P_p1_3`),
alongside `LH` at the same Generation level -- see `config.SIM_PATH_1P`.

The exact grid size, step spacing, and whether multiple seeds exist per
step are release-specific and not hardcoded here; `infer_1p_parameter_names`
is the empirical check that stands in for trusting the p1..p6 convention
blindly -- see notebook 04, which lists what's actually on disk before
running anything.
"""

import re

import numpy as np
import pandas as pd


_LABEL_RE = re.compile(r"^p(\d+)_(\d+)$")


def parse_1p_label(label):
    """
    "p1_3" -> (1, 3): (param_index, step_index), both 1-based as CAMELS
    names them. Raises ValueError on anything that doesn't match.
    """
    m = _LABEL_RE.match(label)
    if not m:
        if label == '0' :
            return 0,0 
        elif label.split('_')[1][0]== 'n' :
            negative_ = re.compile(r"^p(\d+)_n(\d+)$")
            m = negative_.match(label)
            return int(m.group(1)), -1*int(m.group(2))
        else :
            raise ValueError(f"'{label}' is not a 1P label of the form 'p<N>_<M>'")
    return int(m.group(1)), int(m.group(2))


def infer_1p_parameter_names(theta_1p, all_params, tol=1e-8):
    """
    For each param_index group in `theta_1p` (indexed by "p<N>_<M>" labels),
    find which single column of `all_params` actually varies across that
    group -- the empirical check standing in for trusting the CAMELS
    p1..p6 -> parameter-name convention blindly.

    Parameters
    ----------
    theta_1p   : DataFrame indexed by "p<N>_<M>" labels (from
                 `params.load_1p_params`), columns include `all_params`
    all_params : the 6 parameter names to check (config.ALL_PARAMS)
    tol        : a column with range <= tol within a group is "not varying"

    Returns
    -------
    mapping : dict {param_index: parameter_name}, only for groups that
              cleanly match exactly one varying column
    ambiguous : dict {param_index: [varying_column_names]} for any group
                that matched zero or more-than-one columns -- inspect these
                by hand rather than trusting the inferred mapping
    """
    parsed = {label: parse_1p_label(label) for label in theta_1p.index}
    param_indices = sorted({p for p, _ in parsed.values()})

    mapping = {}
    ambiguous = {}

    for pidx in param_indices:
        group_labels = [lab for lab, (p, _) in parsed.items() if p == pidx]
        group = theta_1p.loc[group_labels, all_params]

        ranges = group.max() - group.min()
        varying = list(ranges[ranges > tol].index)

        if len(varying) == 1:
            mapping[pidx] = varying[0]
        else:
            ambiguous[pidx] = varying

    return mapping, ambiguous


def group_steps(sim_ids, param_index):
    """
    Given a 1P run's `sim_ids` (string labels "p<N>_<M>") and a target
    `param_index`, return the subset of labels belonging to that parameter,
    sorted by step_index.
    """
    labels = [s for s in sim_ids if parse_1p_label(s)[0] == param_index]
    return sorted(labels, key=lambda s: parse_1p_label(s)[1])


def mean_cdf_by_step(summaries, sim_ids, param_index, n_k, n_r):
    """
    For one target parameter, average `summaries` over any repeated steps
    (multiple seeds at the same step_index, if present) and return the
    per-step mean CDFs alongside their step indices and parameter values.

    Parameters
    ----------
    summaries   : (n_sims, n_k*n_r) kNN-CDF summaries, aligned to sim_ids
    sim_ids     : string labels "p<N>_<M>", same order as summaries' rows
    param_index : which parameter (1-based, per parse_1p_label) to group by

    Returns
    -------
    step_indices : sorted ndarray of distinct step indices for this parameter
    mean_cdfs    : ndarray (n_steps, n_k, n_r), mean summary at each step
    """
    sim_ids = np.asarray(sim_ids)
    steps = np.array([parse_1p_label(s)[1] for s in sim_ids])
    is_param = np.array([parse_1p_label(s)[0] == param_index for s in sim_ids])

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
    sense the way it does at 1000 simulations.

    Returns (rho, p_value). Needs at least 3 distinct steps.
    """
    from scipy.stats import spearmanr

    if len(step_values) < 3:
        raise ValueError("need at least 3 steps for a trend test")
    return spearmanr(step_values, response_at_r)
