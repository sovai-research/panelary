"""Column contracts (pure Polars) and the optional dataframely backend."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import polars as pl
import pytest

from panelary._internal._deps import have
from panelary.quality import ColumnContract, PanelValidator, validate_panel
from panelary.quality._common import FAILED_CHECKS_COL


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c", "c"],
            "t": [1, 2, 1, 2, 1, 2],
            "px": [10.0, None, 9.0, float("nan"), -1.0, 12.0],
            "side": ["buy", "sell", "buy", "hold", "sell", "buy"],
            "code": [1, 2, 3, 3, 5, 6],
        }
    )


def _run(**kw):
    return validate_panel(
        _frame(), entity="id", time="t", raise_on_fail=False, emit_warnings=False, **kw
    )


# --------------------------------------------------------------------------- #
# dtype
# --------------------------------------------------------------------------- #
def test_dtype_class_instance_and_family():
    report = _run(
        schema={
            "px": pl.Float64,
            "code": pl.Int32,
            "side": "string",
            "t": ColumnContract(dtype="integer"),
        }
    )
    assert report.check("dtype", "px").passed
    assert report.check("dtype", "side").passed
    assert report.check("dtype", "t").passed
    bad = report.check("dtype", "code")
    assert bad.status == "fail"
    assert bad.observed == {"dtype": "Int64"} and bad.expected == {"dtype": "Int32"}
    assert bad.row_mask is None  # frame-level: no row can repair a dtype


def test_datetime_class_matches_any_unit_but_instance_is_exact():
    df = pl.DataFrame(
        {"id": ["a"], "t": [1], "ts": pl.Series([0], dtype=pl.Datetime("ms"))}
    )
    ok = validate_panel(df, entity="id", time="t", schema={"ts": pl.Datetime})
    assert ok.check("dtype", "ts").passed
    bad = validate_panel(
        df,
        entity="id",
        time="t",
        schema={"ts": pl.Datetime("us")},
        raise_on_fail=False,
    )
    assert not bad.check("dtype", "ts").passed


def test_unknown_dtype_family_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown dtype family"):
        PanelValidator(schema={"px": "floaty"})
    with pytest.raises(TypeError, match="polars dtype"):
        PanelValidator(schema={"px": 3})


# --------------------------------------------------------------------------- #
# row-level contracts
# --------------------------------------------------------------------------- #
def test_nullability_flags_rows_with_keys():
    report = _run(schema={"px": ColumnContract(nullable=False)})
    check = report.check("nullability", "px")
    assert check.status == "fail" and check.n_failing == 1
    assert check.offending == ({"keys": {"id": "a", "t": 2}, "row": 1},)
    assert report.invalid.select("id", "t").rows() == [("a", 2)]
    assert report.invalid[FAILED_CHECKS_COL].to_list() == [["nullability:px"]]


def test_nan_is_separate_from_null():
    report = _run(schema={"px": ColumnContract(allow_nan=False)})
    check = report.check("nan", "px")
    assert check.n_failing == 1 and check.offending[0]["keys"] == {"id": "b", "t": 2}


def test_unique_allowed_and_range():
    report = _run(
        schema={
            "code": ColumnContract(unique=True),
            "side": ColumnContract(allowed=["buy", "sell"]),
            "px": ColumnContract(ge=0.0, le=11.0),
        }
    )
    assert report.check("unique", "code").n_failing == 2
    allowed = report.check("allowed_values", "side")
    assert allowed.n_failing == 1 and allowed.offending[0]["side"] == "hold"
    rng = report.check("range", "px")
    # -1.0 and 12.0 are out of range; null and NaN are not range violations
    assert rng.n_failing == 2
    assert {tuple(r["keys"].values()) for r in rng.offending} == {("c", 1), ("c", 2)}


def test_required_and_strict_columns():
    report = _run(
        schema={"px": pl.Float64, "ghost": ColumnContract(required=True)},
        strict=True,
    )
    present = report.check("columns_present")
    assert present.status == "fail" and present.observed == {"missing": ["ghost"]}
    extra = report.check("extra_columns")
    assert extra.observed == {"extra": ["side", "code"]}
    optional = _run(schema={"ghost": ColumnContract(required=False)})
    assert optional.check("columns_present").passed


def test_rules_null_counts_as_valid():
    report = _run(rules={"positive_px": pl.col("px") > 0})
    check = report.check("rule", "positive_px")
    # -1.0 fails. Null is valid (SQL CHECK semantics); NaN > 0 holds because
    # polars orders NaN above every number.
    assert check.n_failing == 1
    assert check.offending[0]["keys"] == {"id": "c", "t": 1}
    with pytest.raises(TypeError, match="boolean polars expression"):
        _run(rules={"bad": "px > 0"})


def test_column_impact_override_downgrades_to_warning():
    report = _run(schema={"px": ColumnContract(nullable=False, impact="warn")})
    assert report.check("nullability", "px").status == "warn"
    assert report.ok and report.invalid.height == 0


def test_schema_checks_run_even_when_keys_are_unusable():
    df = _frame().drop("t")
    report = validate_panel(
        df,
        entity="id",
        time="t",
        schema={"px": ColumnContract(nullable=False)},
        raise_on_fail=False,
    )
    check = report.check("nullability", "px")
    assert check.offending == ({"keys": {}, "row": 1},)


def test_schema_must_be_mapping_or_dataframely():
    with pytest.raises(TypeError, match="mapping"):
        PanelValidator(schema=[("px", pl.Float64)])


# --------------------------------------------------------------------------- #
# dataframely backend
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(have("dataframely"), reason="dataframely is installed")
def test_dataframely_backend_missing_gives_install_hint():
    with pytest.raises(ImportError, match=r"panelary\[schema\]"):
        PanelValidator(schema={"px": pl.Float64}, backend="dataframely")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        PanelValidator(backend="pandera")  # type: ignore[arg-type]


def test_real_dataframely_schema():
    dy = pytest.importorskip("dataframely")

    class Prices(dy.Schema):
        id = dy.String(nullable=False)
        t = dy.Int64(nullable=False)
        px = dy.Float64(nullable=False)

    report = _run(schema=Prices)
    check = report.check("dataframely", "Prices")
    assert check.status == "fail"
    assert report.invalid.select("id", "t").rows() == [("a", 2)]


def _fake_dataframely() -> types.ModuleType:
    """A minimal stand-in for dataframely's documented surface.

    It models only what the adapter touches -- ``Schema`` subclasses declared
    with column objects carrying ``nullable``, ``Schema.filter(df, cast=)``
    returning ``(valid, failure)`` with ``failure.counts()`` -- so this test
    pins the adapter's plumbing (row-id threading, rule counts, the invalid
    split), not dataframely itself; ``test_real_dataframely_schema`` does that
    when the library is installed.
    """
    mod = types.ModuleType("dataframely")

    class Column:
        def __init__(self, nullable: bool = True, **kwargs: object) -> None:
            self.nullable = nullable
            self.kwargs = kwargs

    for name in (
        "Int64",
        "Int32",
        "Float64",
        "String",
        "Bool",
        "Date",
        "Datetime",
        "Any",
    ):
        setattr(mod, name, type(name, (Column,), {}))

    class Schema:
        @classmethod
        def _columns(cls) -> dict[str, Column]:
            out: dict[str, Column] = {}
            for klass in reversed(cls.__mro__):
                for key, value in vars(klass).items():
                    if isinstance(value, Column):
                        out[key] = value
            return out

        @classmethod
        def filter(cls, df: pl.DataFrame, *, cast: bool = False):
            cols = cls._columns()
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"missing columns: {missing}")
            counts: dict[str, int] = {}
            bad = pl.lit(False)
            for col, spec in cols.items():
                if not spec.nullable:
                    n = df.get_column(col).null_count()
                    if n:
                        counts[f"{col}|nullability"] = n
                    bad = bad | pl.col(col).is_null()
            good = df.filter(~bad).select(list(cols))
            return good, SimpleNamespace(counts=lambda: counts)

    Schema.__module__ = "dataframely"
    mod.Schema = Schema  # type: ignore[attr-defined]
    return mod


def test_dataframely_adapter_plumbing(monkeypatch):
    fake = _fake_dataframely()
    monkeypatch.setitem(sys.modules, "dataframely", fake)

    class Prices(fake.Schema):  # type: ignore[name-defined,misc]
        px = fake.Float64(nullable=False)  # type: ignore[attr-defined]

    report = _run(schema=Prices)
    check = report.check("dataframely", "Prices")
    assert check.observed == {"rule_counts": {"px|nullability": 1}, "rows": 1}
    assert check.offending == ({"keys": {"id": "a", "t": 2}, "row": 1},)
    assert report.invalid.select("id", "t").rows() == [("a", 2)]
    assert report.invalid[FAILED_CHECKS_COL].to_list() == [["dataframely:Prices"]]


def test_dataframely_backend_translates_contracts(monkeypatch):
    fake = _fake_dataframely()
    monkeypatch.setitem(sys.modules, "dataframely", fake)
    report = _run(
        schema={
            "px": ColumnContract(dtype=pl.Float64, nullable=False, ge=0.0),
            "side": "string",
        },
        backend="dataframely",
    )
    ids = [c.id for c in report.checks]
    # dtype / nullability went to dataframely; the range stayed pure Polars
    assert "dataframely:PanelaryContract" in ids and "range:px" in ids
    assert "nullability:px" not in ids and "dtype:px" not in ids
    assert report.check("dataframely").observed["rule_counts"] == {"px|nullability": 1}


def test_dataframely_schema_error_is_frame_level(monkeypatch):
    fake = _fake_dataframely()
    monkeypatch.setitem(sys.modules, "dataframely", fake)

    class Needs(fake.Schema):  # type: ignore[name-defined,misc]
        ghost = fake.Int64()  # type: ignore[attr-defined]

    report = _run(schema=Needs)
    check = report.check("dataframely")
    assert check.status == "fail" and check.row_mask is None
    assert "missing columns" in check.message
