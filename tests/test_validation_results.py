"""``EvaluationResult``: the one result type of the forecast-evaluation family."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from panelary.validation._results import (
    EVALUATION_SCHEMA,
    EvaluationResult,
    evaluation_table,
)


def _result(**overrides: object) -> EvaluationResult:
    kwargs: dict[str, object] = {
        "test": "sharpe_difference",
        "names": ("a", "b", "c"),
        "estimate": [0.1, -0.2, np.nan],
        "statistic": [1.0, -2.0, np.nan],
        "pvalue": [0.3, 0.04, np.nan],
        "reference": "cbb-studentized(b=6,B=999)",
        "alternative": "two-sided",
        "n_obs": 250,
        "ci": [[0.0, 0.2], [-0.3, -0.1], [np.nan, np.nan]],
        "block_length": 6,
        "n_resamples": 999,
        "seed": 7,
        "details": {"per_model": np.arange(3.0), "grid": np.arange(5.0)},
    }
    kwargs.update(overrides)
    return EvaluationResult(**kwargs)  # type: ignore[arg-type]


def test_to_frame_has_fixed_schema_and_one_row_per_model() -> None:
    frame = _result().to_frame()
    assert frame.schema == pl.Schema(EVALUATION_SCHEMA)
    assert frame.height == 3
    assert frame["n_obs"].to_list() == [250, 250, 250]
    assert frame["pvalue_adj"].null_count() == 3
    assert frame["ci_low"].to_list()[:2] == [0.0, -0.3]


def test_results_from_different_tests_stack() -> None:
    other = _result(
        test="kupiec",
        names=("x",),
        estimate=0.02,
        statistic=0.5,
        pvalue=0.48,
        reference="binomial-exact",
        n_obs=[250],
        ci=None,
        details={},
    )
    table = evaluation_table([_result(), other])
    assert table.height == 4
    assert table["test"].to_list() == ["sharpe_difference"] * 3 + ["kupiec"]
    assert evaluation_table([]).schema == pl.Schema(EVALUATION_SCHEMA)


def test_to_dict_is_json_safe() -> None:
    payload = _result().to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["estimate"] == [0.1, -0.2, None]


def test_getitem_slices_per_model_fields_only() -> None:
    one = _result()["b"]
    assert one.names == ("b",)
    assert one.pvalue.tolist() == [0.04]
    assert one.ci is not None and one.ci.tolist() == [[-0.3, -0.1]]
    assert one.details["per_model"].tolist() == [1.0]
    assert one.details["grid"].shape == (5,)
    with pytest.raises(KeyError):
        _result()["zzz"]


def test_validation_of_shapes_and_names() -> None:
    with pytest.raises(ValueError, match="shape"):
        _result(estimate=[1.0, 2.0])
    with pytest.raises(ValueError, match="unique"):
        _result(names=("a", "a", "b"))
    with pytest.raises(ValueError, match="adjustment"):
        _result(pvalue_adj=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="ci"):
        _result(ci=[[0.0, 1.0]])


def test_to_dict_handles_zero_dimensional_details() -> None:
    payload = _result(details={"scalar": np.float64(0.5), "zero_d": np.array(2.0)})
    out = payload.to_dict()
    assert out["details"]["zero_d"] == 2.0
    json.dumps(out, allow_nan=False)


def test_per_model_keys_resolve_the_length_m_ambiguity() -> None:
    # A length-T path whose T happens to equal M must not be sliced.
    path = np.arange(3.0)
    res = _result(
        details={"per_model": np.arange(3.0), "path": path}, per_model=("per_model",)
    )
    one = res["b"]
    assert one.details["per_model"].tolist() == [1.0]
    assert one.details["path"].tolist() == [0.0, 1.0, 2.0]
    assert one["b"].details["per_model"].tolist() == [1.0]
    with pytest.raises(ValueError, match="per_model"):
        _result(details={"path": np.arange(5.0)}, per_model=("path",))
