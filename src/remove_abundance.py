# src/remove_abundance.py


import numpy as np

from sklearn.linear_model import LinearRegression


def remove_abundance(
    summaries,
    nbh
):

    X = np.log10(
        nbh
    ).reshape(-1,1)

    residuals = np.zeros_like(
        summaries
    )

    for j in range(
        summaries.shape[1]
    ):

        y = summaries[:,j]

        model = LinearRegression()

        model.fit(X,y)

        residuals[:,j] = (
            y
            -
            model.predict(X)
        )

    return residuals