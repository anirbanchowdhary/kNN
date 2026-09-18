import numpy as np
import pandas as pd

from src.sensitivity import (
    scale_resolved_response,
    sensitivity_table,
    summary_dataframe,
)


def _synthetic_case(seed=0, n_sims=400, n_k=3, n_r=40, signal_bins=None):
    rng = np.random.default_rng(seed)
    n_total = n_k * n_r

    theta = pd.DataFrame({
        "Omega_m": rng.uniform(0.1, 0.5, n_sims),
        "sigma_8": rng.uniform(0.6, 1.0, n_sims),
        "A_SN1": rng.uniform(0.25, 4.0, n_sims),
    }, index=np.arange(n_sims))

    residuals = rng.normal(0, 0.01, (n_sims, n_total))

    if signal_bins is not None:
        om_centered = (theta["Omega_m"] - theta["Omega_m"].mean()).values
        bump = np.zeros(n_total)
        bump[signal_bins] = 1.0
        residuals += 0.05 * np.outer(om_centered, bump)

    return residuals, theta, n_k, n_r


def test_scalar_R_matches_rms_of_scale_resolved_diff():
    residuals, theta, n_k, n_r = _synthetic_case()
    rgrid = np.logspace(-1.5, 1.2, n_r)
    kvals = list(range(n_k))

    table = sensitivity_table(
        residuals, theta, n_k=n_k, rgrid=rgrid, kvals=kvals,
        params=["Omega_m", "sigma_8", "A_SN1"], n_boot=200, n_null=200,
    )

    for p in table["params"]:
        obs = table[p]["obs"]
        assert obs.shape == (n_k, n_r)

        hand_rolled = np.sqrt(np.nanmean(
            scale_resolved_response(residuals, theta[p].values, n_k) ** 2
        ))
        assert np.isclose(table[p]["R_scalar"], hand_rolled)


def test_injected_signal_is_detected_at_the_right_bins():
    n_k, n_r = 3, 40
    signal_bins = slice(0 * n_r + n_r // 2, 0 * n_r + n_r)  # k=0 block, large-r half
    residuals, theta, n_k, n_r = _synthetic_case(n_k=n_k, n_r=n_r, signal_bins=signal_bins)
    rgrid = np.logspace(-1.5, 1.2, n_r)
    kvals = list(range(n_k))

    table = sensitivity_table(
        residuals, theta, n_k=n_k, rgrid=rgrid, kvals=kvals,
        params=["Omega_m", "sigma_8"], n_boot=300, n_null=300,
    )

    om_sig = table["Omega_m"]["significant"]
    sigma8_sig = table["sigma_8"]["significant"]

    # Signal was injected only into Omega_m's large-r half of k=0.
    assert om_sig[0, n_r // 2:].sum() > om_sig[0, :n_r // 2].sum()
    # sigma_8 got no injected signal, so it shouldn't show up as densely
    # significant as Omega_m does in its injected region.
    assert om_sig[0, n_r // 2:].sum() > sigma8_sig.sum()


def test_summary_dataframe_shape():
    residuals, theta, n_k, n_r = _synthetic_case()
    rgrid = np.logspace(-1.5, 1.2, n_r)
    kvals = list(range(n_k))

    table = sensitivity_table(
        residuals, theta, n_k=n_k, rgrid=rgrid, kvals=kvals,
        params=["Omega_m", "sigma_8", "A_SN1"], n_boot=100, n_null=100,
    )
    df = summary_dataframe(table)

    assert set(df["parameter"]) == {"Omega_m", "sigma_8", "A_SN1"}
    assert {"R_obs", "ci_lo", "ci_hi", "p_value", "significant"} <= set(df.columns)
