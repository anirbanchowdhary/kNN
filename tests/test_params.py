import pandas as pd
import pytest

from src.params import load_params, load_1p_params, load_cv_params


def _write(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_load_params_drops_the_name_column(tmp_path):
    # Regression: the raw label column (e.g. "#Name") used to derive
    # sim_id was left in the DataFrame, so it showed up as a bogus
    # "parameter" to any code that scans theta.columns.
    path = _write(tmp_path, "lh.txt", [
        "#Name Omega_m sigma_8 seed",
        "LH_0 0.3 0.8 1",
        "LH_1 0.35 0.85 2",
    ])
    theta = load_params(path)

    assert "#Name" not in theta.columns
    assert list(theta.columns) == ["Omega_m", "sigma_8", "seed"]
    assert list(theta.index) == [0, 1]


def test_load_1p_params_drops_the_name_column_and_keeps_string_labels(tmp_path):
    path = _write(tmp_path, "onep.txt", [
        "#Name Omega0 sigma8 seed",
        "1P_0 0.3 0.8 1",
        "1P_p1_1 0.35 0.8 2",
        "1P_p1_n1 0.25 0.8 3",
    ])
    theta = load_1p_params(path)

    assert "#Name" not in theta.columns
    assert list(theta.columns) == ["Omega0", "sigma8", "seed"]
    assert set(theta.index) == {"0", "p1_1", "p1_n1"}
    # every remaining column must be numeric -- this is exactly the
    # precondition infer_1p_parameter_names's max()-min() needs
    for col in theta.columns:
        assert pd.api.types.is_numeric_dtype(theta[col])


def test_load_cv_params_indexes_by_integer_realization(tmp_path):
    path = _write(tmp_path, "cv.txt", [
        "#Name Omega_m sigma_8 seed",
        "CV_0 0.3 0.8 1",
        "CV_1 0.3 0.8 2",
        "CV_2 0.3 0.8 3",
    ])
    theta = load_cv_params(path)

    assert "#Name" not in theta.columns
    assert list(theta.index) == [0, 1, 2]
    # CV holds the physics fixed -- only seed should vary
    assert theta["Omega_m"].nunique() == 1
    assert theta["sigma_8"].nunique() == 1
    assert theta["seed"].nunique() == 3
