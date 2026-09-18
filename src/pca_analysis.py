from sklearn.decomposition import PCA


def run_pca(residuals):

    pca = PCA()

    scores = pca.fit_transform(
        residuals
    )

    explained = (
        pca.explained_variance_ratio_
    )

    return (
        pca,
        scores,
        explained
    )