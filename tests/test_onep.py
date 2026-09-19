import numpy as np
import pandas as pd
import pytest

from src.onep import (
    parse_1p_label,
    infer_1p_parameter_names,
    group_steps,
    mean_cdf_by_step,
    monotonic_trend,
)


def test_parse_1p_label():
    assert parse_1p_label("p1_3") == (1, 3)
    assert parse_1p_label("p6_0") == (6, 0)


def test_parse_1p_label_rejects_bad_format():
    for bad in ["LH_1", "p1", "1_3", "pA_3"]:
        with pytest.raises(ValueError):
            parse_1p_label(bad)


def _synthetic_1p_theta(all_params, n_steps=5, seed=0):
    """
    6 groups (p1..p6), each with n_steps rows: exactly one column varies
    per group, the rest sit at a shared fiducial value.
    """
    rng = np.random.default_rng(seed)
    fiducial = {p: rng.uniform(0.3, 0.7) for p in all_params}

    rows = {}
    for pidx, target in enumerate(all_params, start=1):
        grid = np.linspace(0.1, 0.9, n_steps)
        for step, val in enumerate(grid):
            label = f"p{pidx}_{step}"
            row = dict(fiducial)
            row[target] = val
            rows[label] = row

    return pd.DataFrame(rows).T[all_params]


def test_infer_1p_parameter_names_recovers_the_mapping():
    all_params = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
    theta_1p = _synthetic_1p_theta(all_params)

    mapping, ambiguous = infer_1p_parameter_names(theta_1p, all_params)

    assert ambiguous == {}
    assert mapping == {i + 1: p for i, p in enumerate(all_params)}


def test_infer_1p_parameter_names_flags_ambiguous_group():
    all_params = ["Omega_m", "sigma_8"]
    theta_1p = _synthetic_1p_theta(all_params, n_steps=4)

    # Break group p1: make a second column vary too, so it's no longer
    # exactly-one-varying-column.
    theta_1p.loc[theta_1p.index.str.startswith("p1_"), "sigma_8"] = np.linspace(0.1, 0.9, 4)

    mapping, ambiguous = infer_1p_parameter_names(theta_1p, all_params)

    assert 1 in ambiguous
    assert set(ambiguous[1]) == {"Omega_m", "sigma_8"}
    assert 1 not in mapping
    assert mapping[2] == "sigma_8"


def test_group_steps_filters_and_sorts():
    sim_ids = ["p2_1", "p1_3", "p1_0", "p3_0", "p1_1"]
    assert group_steps(sim_ids, 1) == ["p1_0", "p1_1", "p1_3"]
    assert group_steps(sim_ids, 3) == ["p3_0"]
    assert group_steps(sim_ids, 9) == []


def test_mean_cdf_by_step_averages_repeated_steps():
    n_k, n_r = 2, 3
    # p1 has two sims at step 0 (different seeds) and one at step 1
    sim_ids = ["p1_0", "p1_0", "p1_1", "p2_0"]
    summaries = np.array([
        np.full(n_k * n_r, 1.0),
        np.full(n_k * n_r, 3.0),   # step 0 mean should be 2.0
        np.full(n_k * n_r, 5.0),   # step 1
        np.full(n_k * n_r, 99.0),  # different parameter, must be excluded
    ])

    steps, mean_cdfs = mean_cdf_by_step(summaries, sim_ids, param_index=1, n_k=n_k, n_r=n_r)

    assert list(steps) == [0, 1]
    assert np.allclose(mean_cdfs[0], 2.0)
    assert np.allclose(mean_cdfs[1], 5.0)


def test_monotonic_trend_detects_a_real_trend():
    values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    response = values * 2 + 0.01  # monotone increasing, tiny noise-free offset

    rho, p = monotonic_trend(values, response)
    assert rho > 0.99
    assert p < 0.01


def test_monotonic_trend_requires_at_least_three_steps():
    with pytest.raises(ValueError):
        monotonic_trend(np.array([0.1, 0.2]), np.array([1.0, 2.0]))
