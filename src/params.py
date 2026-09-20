"""
Loading CAMELS parameter tables (LH and 1P).

Deliberately join-based (never positional): every downstream function
takes DataFrames/arrays keyed by sim_id and looks values up by that key,
so a summaries array and a theta table can never silently drift out of
alignment with each other (see README for why this matters).
"""

import pandas as pd


def _load_params_table(params_file, prefix, cast):
    """
    Shared loader: whitespace-delimited file with a header row, first
    column holding labels like "<prefix><label>" (e.g. "LH_0", "1P_p1_3").
    Strips `prefix` and applies `cast` (int for LH, identity for 1P) to get
    the index.
    """
    params = pd.read_csv(params_file, sep=r"\s+")

    name_col = params.columns[0]
    params["sim_id"] = params[name_col].str.replace(prefix, "", regex=False).apply(cast)
    params = params.drop(columns=[name_col])

    return params.set_index("sim_id").sort_index()


def load_params(params_file):
    """
    Load the full LH parameter table, indexed by integer sim_id.

    The file is whitespace-delimited with a header row; the first column
    holds simulation names like "LH_0", "LH_1", ...
    """
    return _load_params_table(params_file, prefix="LH_", cast=int)


def load_1p_params(params_file):
    """
    Load the full 1P parameter table, indexed by the raw string label
    (e.g. "p1_3" for the file's "1P_p1_3" row) -- 1P has no single integer
    id, unlike LH, so this does not int-cast.
    """
    return _load_params_table(params_file, prefix="1P_", cast=str)


def align_to_params(sim_ids, theta_all):
    """
    Return the subset/order of `theta_all` matching `sim_ids` exactly,
    row for row.

    This is the single place alignment happens, so every caller gets the
    same guarantee: `result.index.to_numpy() == np.asarray(sim_ids)`
    (checked below) rather than each caller re-deriving its own
    positional join and risking the sim_id order-mismatch bug.
    """
    theta = theta_all.loc[sim_ids]
    assert list(theta.index) == list(sim_ids), (
        "theta rows are not aligned to sim_ids after .loc lookup; "
        "this should be impossible unless sim_ids has duplicates"
    )
    return theta
