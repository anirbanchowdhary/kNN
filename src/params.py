"""
Loading the CAMELS-IllustrisTNG LH parameter table.

Deliberately join-based (never positional): every downstream function
takes DataFrames/arrays keyed by sim_id and looks values up by that key,
so a summaries array and a theta table can never silently drift out of
alignment with each other (see README for why this matters).
"""

import pandas as pd


def load_params(params_file):
    """
    Load the full LH parameter table, indexed by integer sim_id.

    The file is whitespace-delimited with a header row; the first column
    holds simulation names like "LH_0", "LH_1", ...
    """
    params = pd.read_csv(params_file, delim_whitespace=True)

    name_col = params.columns[0]
    params["sim_id"] = (
        params[name_col].str.replace("LH_", "", regex=False).astype(int)
    )

    return params.set_index("sim_id").sort_index()


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
