import numpy as np
import pandas as pd
import pytest

from src.params import align_to_params


def test_align_to_params_preserves_sim_id_order():
    # theta_all is deliberately NOT sorted by sim_id, and sim_ids is a
    # scrambled subset -- align_to_params must reorder to match sim_ids
    # exactly, regardless of theta_all's own storage order.
    theta_all = pd.DataFrame(
        {"Omega_m": [0.1, 0.2, 0.3, 0.4, 0.5]},
        index=[30, 10, 50, 20, 40],
    )
    sim_ids = np.array([50, 10, 40])

    aligned = align_to_params(sim_ids, theta_all)

    assert list(aligned.index) == list(sim_ids)
    assert aligned.loc[50, "Omega_m"] == 0.3
    assert aligned.loc[10, "Omega_m"] == 0.2
    assert aligned.loc[40, "Omega_m"] == 0.5


def test_align_to_params_row_matches_residuals_row():
    # Regression test for the exact bug class the reference project hit:
    # residuals[i] and theta.iloc[i] must refer to the same simulation
    # even when sim_ids arrives in a different order than theta_all.
    theta_all = pd.DataFrame({"A_SN1": np.arange(100) / 10.0}, index=np.arange(100))
    sim_ids = np.array([7, 3, 99, 0, 55])
    residuals = np.array([[sid] for sid in sim_ids], dtype=float)  # residual == sim_id, by construction

    aligned = align_to_params(sim_ids, theta_all)

    for i, sid in enumerate(sim_ids):
        assert residuals[i, 0] == sid
        assert aligned.index[i] == sid


def test_align_to_params_missing_sim_id_raises():
    theta_all = pd.DataFrame({"Omega_m": [0.1, 0.2]}, index=[1, 2])
    with pytest.raises(KeyError):
        align_to_params(np.array([1, 999]), theta_all)
