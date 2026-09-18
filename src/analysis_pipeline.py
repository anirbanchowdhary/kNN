# src/analysis_pipeline.py
"""
Shared helper for the kNN sensitivity workflow.

Consolidates the "load summaries -> remove abundance -> load params
-> compute RMS response per parameter (-> optional PCA)" block that
was previously copy-pasted across multiple notebook cells.
"""

import numpy as np
import pandas as pd

from .remove_abundance import remove_abundance
from .parameter_sensitivity import load_params, rms_response
from .pca_analysis import run_pca


DEFAULT_PARAMS = [
    "Omega_m",
    "sigma_8",
    "A_SN1",
    "A_AGN1",
    "A_SN2",
    "A_AGN2",
]

DEFAULT_PARAMS_FILE = (
    "../CAMELS-master/docs/params/IllustrisTNG/"
    "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
)


def standardize(residuals):
    """
    Divide each summary-statistic bin by its std across simulations.

    After this, every bin has unit variance and R(p) measures the
    quartile mean-shift in units of cross-simulation sigma -- a
    dimensionless effect size that's comparable across different
    summary statistics (e.g. kNN-CDF vs 2PCF), which raw R(p) is not
    since the two statistics live on different scales.
    """
    sigma = residuals.std(axis=0)
    sigma = sigma.copy()
    sigma[sigma == 0] = 1.0  # guard against dead bins
    return residuals / sigma


def load_and_analyze(
    npz_path,
    params=None,
    params_file=DEFAULT_PARAMS_FILE,
    run_pca_flag=False,
    standardize_flag=False,
):
    """
    Load a kNN/2PCF summary .npz file, remove the abundance trend,
    optionally standardize (z-score per bin), compute the RMS
    sensitivity for each parameter, and optionally run PCA on the
    residuals.

    Parameters
    ----------
    npz_path          : path to the .npz produced by generate_knn.run_suite
                         (or generate_2pcf.run_suite, generate_knn_active.run_suite)
    params             : list of parameter names (default: DEFAULT_PARAMS)
    params_file        : path to the LH parameter file
    run_pca_flag       : if True, also run PCA on the residuals
    standardize_flag   : if True, z-score residuals per bin before computing
                          sensitivity/PCA. Needed when comparing R(p) across
                          different summary statistics (e.g. kNN vs 2PCF),
                          since otherwise the raw scales aren't comparable.

    Returns
    -------
    dict with keys:
        sim_ids, summaries, nbh, residuals, theta,
        sensitivity (DataFrame: parameter, rms),
        pca, scores, explained   (only if run_pca_flag=True, else None)
    """
    if params is None:
        params = DEFAULT_PARAMS

    data = np.load(npz_path, allow_pickle=True)

    sim_ids = data["sim_ids"]
    summaries = data["summaries"]
    nbh = data["nbh"]

    residuals = remove_abundance(summaries, nbh)

    if standardize_flag:
        residuals = standardize(residuals)

    theta = load_params(sim_ids, params_file)

    rows = []
    for p in params:
        rms = rms_response(residuals, theta, p)
        rows.append({"parameter": p, "rms": rms})

    sensitivity = pd.DataFrame(rows)

    pca = scores = explained = None
    if run_pca_flag:
        pca, scores, explained = run_pca(residuals)

    return {
        "sim_ids": sim_ids,
        "summaries": summaries,
        "nbh": nbh,
        "residuals": residuals,
        "theta": theta,
        "sensitivity": sensitivity,
        "pca": pca,
        "scores": scores,
        "explained": explained,
    }


def sweep_sensitivity(npz_paths, sweep_labels, sweep_name, params=None,
                       params_file=DEFAULT_PARAMS_FILE):
    """
    Run load_and_analyze() over multiple .npz files (e.g. a snapshot
    sweep or a mass-cut sweep) and return one tidy long-form DataFrame
    with columns: [sweep_name, parameter, rms].

    Parameters
    ----------
    npz_paths    : list of .npz file paths
    sweep_labels : list of labels (e.g. snap numbers or mass cuts),
                   same length/order as npz_paths
    sweep_name   : column name for the sweep variable (e.g. "snap", "Mcut")
    """
    if params is None:
        params = DEFAULT_PARAMS

    rows = []
    for path, label in zip(npz_paths, sweep_labels):
        result = load_and_analyze(path, params=params, params_file=params_file)
        for _, row in result["sensitivity"].iterrows():
            rows.append({
                sweep_name: label,
                "parameter": row["parameter"],
                "rms": row["rms"],
            })

    return pd.DataFrame(rows)