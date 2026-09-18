# src/parameter_sensitivity.py


import numpy as np
import pandas as pd


def load_params(
    sim_ids,
    filename="LH_params.txt"
):

    params = pd.read_csv(
        filename,
        delim_whitespace=True
    )

    params["sim_id"] = (
        params.iloc[:,0]
        .str.replace(
            "LH_",
            "",
            regex=False
        )
        .astype(int)
    )

    params = (
        params
        .set_index("sim_id")
        .loc[sim_ids]
    )

    return params


def rms_response(
    residuals,
    theta,
    parameter
):

    q25 = theta[
        parameter
    ].quantile(0.25)

    q75 = theta[
        parameter
    ].quantile(0.75)

    low = (
        theta[parameter]
        < q25
    )

    high = (
        theta[parameter]
        > q75
    )

    diff = (
        residuals[high].mean(axis=0)
        -
        residuals[low].mean(axis=0)
    )

    return np.sqrt(
        np.mean(diff**2)
    )