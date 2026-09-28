"""Tests for ``panelary.clean`` canonicalisation and survivorship (M3).

Canonicalisation is deterministic and row-local except for one piece of
learned state: with ``cluster=`` the :class:`Canonicalizer` learns, at fit,
which spelling represents each fingerprint key. The leak-safety contract for
that state, cashed out:

1. **Fit on train only.** ``vocab_`` equals the key -> canonical table of the
   training rows and differs from a full-sample fit when the test fold prefers
   another spelling.
2. **Transform never refits.** Test output equals the frozen training
   vocabulary applied to the test rows; a deliberately leaky subclass that
   re-learns at transform time fails that check.
3. **Train outputs cannot see the test fold.** Fitting on train and
   transforming the whole panel leaves train rows unchanged when the test
   fold's strings are rewritten; the leaky variant that fits on all rows fails.

Survivorship learns nothing, so it has no fitted state to leak; its tests pin
each golden-record rule and the documented fact that a golden record reads
the whole cluster (which is why :class:`Deduplicator` drops ``leakage_safe``
when it merges).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.clean import (
    Canonicalizer,
    Deduplicator,
    Survivorship,
    fingerprint,
    fingerprint_clusters,
    golden_records,
    ngram_fingerprint,
    normalize_text,
)
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer


def _norm(values, **kw) -> list:
    frame = pl.DataFrame({"s": values}, schema={"s": pl.String})
    return frame.select(normalize_text("s", **kw)).to_series().to_list()


def _apply(expr, values) -> list:
    frame = pl.DataFrame({"s": values}, schema={"s": pl.String})
    return frame.select(expr).to_series().to_list()


# --------------------------------------------------------------------------- #
# normalize_text
# --------------------------------------------------------------------------- #
def test_normalize_text_default_steps():
    # NFC composes "e" + combining acute into one code point; case folds;
    # whitespace is trimmed and collapsed; punctuation is kept by default.
    got = _norm(["  Café  DU   Monde. ", None, ""])
    assert got == ["café du monde.", None, ""]


def test_normalize_text_unicode_forms():
    assert _norm(["ﬁne Ⅷ"], unicode="NFKC") == ["fine viii"]
    # NFC leaves compatibility characters alone.
    assert _norm(["ﬁne"], unicode="NFC") == ["ﬁne"]
    # unicode=None skips normalisation: the decomposed form survives.
    assert _norm(["é"], unicode=None) == ["é"]
    with pytest.raises(ValueError, match="unicode"):
        normalize_text("s", unicode="NFX")  # type: ignore[arg-type]


def test_normalize_text_optional_steps():
    assert _norm(["Société Générale"], strip_accents=True) == ["societe generale"]
    assert _norm(["ACME, Corp. (US) & co!"], strip_punctuation=True) == [
        "acme corp us co"
    ]
    assert _norm(["MiXeD"], casefold=False) == ["MiXeD"]
    assert _norm(["a   b"], collapse_whitespace=False) == ["a   b"]


def test_normalize_text_replacements_run_in_order():
    ordered = [(r"\bcorporation\b", "corp"), (r"\bcorp\b", "co")]
    assert _norm(["Acme Corporation"], replacements=ordered) == ["acme co"]
    # A mapping works too, and runs after case folding.
    assert _norm(["Acme INCORPORATED"], replacements={r"\bincorporated\b": "inc"}) == [
        "acme inc"
    ]


def test_normalize_text_accepts_expressions_and_keeps_the_name():
    frame = pl.DataFrame({"name": [" X "]})
    out = frame.select(normalize_text(pl.col("name")))
    assert out.columns == ["name"]
    assert out["name"].to_list() == ["x"]


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def test_fingerprint_is_order_case_punctuation_and_repeat_insensitive():
    values = ["Acme, Corp.", "corp ACME", "ACME   CORP ACME", "Ácme Corp"]
    assert _apply(fingerprint("s"), values) == ["acme corp"] * 4
    assert _apply(fingerprint("s"), [None, "", " "]) == [None, "", ""]
    assert _apply(fingerprint("s"), ["Beta"]) != _apply(fingerprint("s"), ["Acme"])


def test_ngram_fingerprint_values():
    # Whitespace and punctuation vanish, so spacing variants collide.
    got = _apply(ngram_fingerprint("s"), ["acme corp", "AcmeCorp", "acme-corp"])
    assert len(set(got)) == 1
    # Sorted distinct bigrams of "abab": {"ab", "ba"}.
    assert _apply(ngram_fingerprint("s", 2), ["abab"]) == ["abba"]
    # n = 1 is the sorted distinct characters.
    assert _apply(ngram_fingerprint("s", 1), ["banana"]) == ["abn"]
    # Strings shorter than n are returned normalised, not dropped.
    assert _apply(ngram_fingerprint("s", 5), ["ab", None, ""]) == ["ab", None, ""]
    with pytest.raises(ValueError, match="n"):
        ngram_fingerprint("s", 0)


def test_fingerprint_clusters_picks_the_most_frequent_spelling():
    df = pl.DataFrame(
        {
            "name": ["Acme Corp", "acme corp", "Corp ACME", "Acme Corp", None, "Beta"],
            "t": [3, 1, 2, 4, 0, 5],
        }
    )
    got = fingerprint_clusters(df, "name")
    assert got.columns == ["value", "key", "count", "canonical"]
    assert got.height == 4  # null is not a value
    acme = got.filter(pl.col("key") == "acme corp")
    assert set(acme["canonical"].to_list()) == {"Acme Corp"}
    assert acme["count"].to_list() == [2, 1, 1]
    assert got.filter(pl.col("key") == "beta")["canonical"].to_list() == ["Beta"]
    # LazyFrame input gives the same table.
    assert fingerprint_clusters(df.lazy(), "name").equals(got)
    with pytest.raises(ValueError, match="method"):
        fingerprint_clusters(df, "name", method="soundex")  # type: ignore[arg-type]


def test_fingerprint_clusters_ties_go_to_the_value_seen_first():
    df = pl.DataFrame({"name": ["corp acme", "acme corp"], "t": [2, 1]})
    # Input order: "corp acme" first.
    assert fingerprint_clusters(df, "name")["canonical"].unique().to_list() == [
        "corp acme"
    ]
    # Time order: "acme corp" (t=1) first.
    by_time = fingerprint_clusters(df, "name", order_by="t")
    assert by_time["canonical"].unique().to_list() == ["acme corp"]


def test_fingerprint_clusters_ngram_method():
    df = pl.DataFrame({"name": ["Acme Corp", "AcmeCorp", "acme corp", "Beta"]})
    got = fingerprint_clusters(df, "name", method="ngram", n=2)
    acme = got.filter(pl.col("value") != "Beta")
    assert acme["key"].n_unique() == 1
    assert acme["canonical"].unique().to_list() == ["Acme Corp"]


# --------------------------------------------------------------------------- #
# Canonicalizer: behaviour
# --------------------------------------------------------------------------- #
def _names_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "t": [1, 1, 2, 2, 3],
            "name": ["ACME  Corp.", "acme corp", "Acme Corp", "Corp. ACME", None],
            "country": ["U.S.", "united states", "US", "us", "UK"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_canonicalizer_declares_contract():
    assert issubclass(Canonicalizer, PanelTransformer)
    c = Canonicalizer()
    assert c.panel_safe is True and c.leakage_safe is True


def test_canonicalizer_rewriting_the_entity_key_is_not_panel_safe():
    df = pl.DataFrame({"firm": ["Acme", "ACME "], "t": [1, 2], "x": [1.0, 2.0]})
    c = Canonicalizer(columns="firm", entity="firm", time="t").fit(df)
    assert c.panel_safe is False
    assert c.transform(df).collect()["firm"].to_list() == ["acme", "acme"]
    c2 = Canonicalizer(columns="firm", suffix="_c", entity="firm", time="t").fit(df)
    assert c2.panel_safe is True


def test_canonicalizer_defaults_to_string_non_key_columns():
    df = _names_panel()
    c = Canonicalizer(entity="id", time="t").fit(df)
    assert c.feature_names_in_ == ["name", "country"]
    out = c.transform(df).collect()
    assert out["name"].to_list() == [
        "acme corp.",
        "acme corp",
        "acme corp",
        "corp. acme",
        None,
    ]
    assert out["x"].to_list() == df["x"].to_list()
    assert out.columns == df.columns
    assert c.vocab_ == {}


def test_canonicalizer_mapping_and_suffix():
    df = _names_panel()
    c = Canonicalizer(
        columns=["country"],
        strip_punctuation=True,
        mapping={"country": {"u s": "us", "united states": "us"}},
        suffix="_c",
        entity="id",
        time="t",
    )
    out = c.fit_transform(df).collect()
    assert out["country"].equals(df["country"])  # original kept
    assert out["country_c"].to_list() == ["us", "us", "us", "us", "uk"]


def test_canonicalizer_cluster_collapses_variants():
    df = _names_panel()
    c = Canonicalizer(
        columns="name",
        cluster="fingerprint",
        strip_punctuation=True,
        entity="id",
        time="t",
    )
    out = c.fit_transform(df).collect()
    assert out["name"].to_list() == ["acme corp"] * 4 + [None]
    vocab = c.vocab_["name"]
    assert vocab.columns == ["key", "canonical"]
    assert vocab.to_dicts() == [{"key": "acme corp", "canonical": "acme corp"}]


def test_canonicalizer_unseen_keys_pass_through_normalised():
    train = pl.DataFrame(
        {"id": [1, 2], "t": [1, 1], "name": ["corp acme", "Corp Acme"]}
    )
    test = pl.DataFrame({"id": [1, 3], "t": [2, 2], "name": ["ACME CORP", "Beta  Ltd"]})
    c = Canonicalizer(columns="name", cluster="fingerprint", entity="id", time="t")
    out = c.fit(train).transform(test).collect()
    assert out["name"].to_list() == ["corp acme", "beta ltd"]


def test_canonicalizer_preserves_row_order_and_nulls():
    rng = np.random.default_rng(3)
    variants = ["Acme Corp", "corp acme", "ACME, CORP", "Beta Ltd", "ltd beta", None]
    n = 400
    df = pl.DataFrame(
        {
            "id": rng.integers(0, 40, n),
            "t": rng.integers(0, 50, n),
            "name": [variants[i] for i in rng.integers(0, len(variants), n)],
            "x": rng.standard_normal(n),
        }
    )
    c = Canonicalizer(
        columns="name",
        cluster="fingerprint",
        strip_punctuation=True,
        entity="id",
        time="t",
    )
    out = c.fit_transform(df).collect()
    assert out.select("id", "t", "x").equals(df.select("id", "t", "x"))
    key = df.select(fingerprint("name")).to_series()
    for k, canon in zip(key.to_list(), out["name"].to_list(), strict=True):
        if k is None:
            assert canon is None
        else:
            assert canon in {"acme corp", "corp acme", "beta ltd", "ltd beta"}
            assert sorted(canon.split()) == k.split()
    # One spelling per key.
    assert (
        out.group_by(key.alias("k")).agg(pl.col("name").n_unique())["name"].max() == 1
    )


def test_canonicalizer_container_types():
    df = _names_panel()
    c = Canonicalizer(cluster="fingerprint", entity="id", time="t")
    want = c.fit_transform(df).collect()
    assert c.fit_transform(df.lazy()).collect().equals(want)
    pf = PanelFrame(df, entity="id", time="t")
    assert isinstance(c.fit_transform(pf), PanelFrame)


def test_canonicalizer_validation_errors():
    with pytest.raises(ValueError, match="unicode"):
        Canonicalizer(unicode="NFX")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cluster"):
        Canonicalizer(cluster="soundex")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ngram"):
        Canonicalizer(ngram=0)
    df = _names_panel()
    with pytest.raises(ValueError, match="not found"):
        Canonicalizer(columns="nope", entity="id", time="t").fit(df)
    with pytest.raises(RuntimeError, match="not fitted"):
        Canonicalizer(entity="id", time="t").transform(df)
    c = Canonicalizer(columns="name", entity="id", time="t").fit(df)
    with pytest.raises(ValueError, match="missing"):
        c.transform(df.drop("name")).collect()


# --------------------------------------------------------------------------- #
# Canonicalizer: leak safety of the learned vocabulary
# --------------------------------------------------------------------------- #
_CUT = 5


def _vocab_panel() -> pl.DataFrame:
    """Train (t < 5) prefers "corp acme"; the test fold repeats it too."""
    train = ["corp acme", "corp acme", "corp acme", "acme corp", "acme corp"]
    test = ["corp acme"] * 8
    names = train + test
    return pl.DataFrame(
        {
            "id": [f"e{i % 3}" for i in range(len(names))],
            "t": list(range(len(names))),
            "name": names,
            "x": np.arange(len(names), dtype=np.float64),
        }
    )


def _perturb_test_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Rewrite every test-fold name to another spelling of the same key."""
    return df.with_columns(
        pl.when(pl.col("t") >= _CUT)
        .then(pl.lit("acme, corp"))
        .otherwise(pl.col("name"))
        .alias("name")
    )


def _assert_train_rows_unchanged(op, df: pl.DataFrame) -> None:
    base = op(df).filter(pl.col("t") < _CUT)
    pert = op(_perturb_test_strings(df)).filter(pl.col("t") < _CUT)
    if not base.equals(pert):
        raise AssertionError("LEAK: rewriting the test fold changed train-fold output")


class _RefitsAtTransform(Canonicalizer):
    """Deliberately leaky: re-learns its vocabulary from the transform rows."""

    def _transform(self, panel):
        self._fit(panel)
        return super()._transform(panel)


def _kw() -> dict:
    return {"columns": "name", "cluster": "fingerprint", "entity": "id", "time": "t"}


def test_vocab_is_learned_from_train_only():
    df = _perturb_test_strings(_vocab_panel())
    train = df.filter(pl.col("t") < _CUT)
    fitted = Canonicalizer(**_kw()).fit(train)
    assert fitted.vocab_["name"]["canonical"].to_list() == ["corp acme"]
    # A full-sample fit is dominated by the test fold's spelling.
    full = Canonicalizer(**_kw()).fit(df)
    assert full.vocab_["name"]["canonical"].to_list() == ["acme, corp"]


def _check_frozen(cls, train: pl.DataFrame, test: pl.DataFrame) -> None:
    got = cls(**_kw()).fit(train).transform(test).collect()["name"].to_list()
    ref = Canonicalizer(**_kw()).fit(train)
    table = dict(ref.vocab_["name"].iter_rows())
    keys = test.select(fingerprint("name")).to_series().to_list()
    want = [table[k] for k in keys]
    if got != want:
        raise AssertionError("transform re-learned the vocabulary")


def test_transform_applies_the_frozen_vocabulary():
    df = _perturb_test_strings(_vocab_panel())
    train, test = df.filter(pl.col("t") < _CUT), df.filter(pl.col("t") >= _CUT)
    c = Canonicalizer(**_kw()).fit(train)
    before = c.vocab_["name"].clone()
    c.transform(test).collect()
    assert c.vocab_["name"].equals(before)
    _check_frozen(Canonicalizer, train, test)
    with pytest.raises(AssertionError, match="re-learned"):
        _check_frozen(_RefitsAtTransform, train, test)


def test_fit_on_train_passes_and_fit_on_all_rows_fails():
    df = _vocab_panel()

    def honest(frame):
        c = Canonicalizer(**_kw()).fit(frame.filter(pl.col("t") < _CUT))
        return c.transform(frame).collect()

    def leaky(frame):  # fit on every row, test fold included
        return Canonicalizer(**_kw()).fit(frame).transform(frame).collect()

    def refits(frame):  # fit on train, but re-learn at transform time
        c = _RefitsAtTransform(**_kw()).fit(frame.filter(pl.col("t") < _CUT))
        return c.transform(frame).collect()

    _assert_train_rows_unchanged(honest, df)
    with pytest.raises(AssertionError, match="LEAK"):
        _assert_train_rows_unchanged(leaky, df)
    with pytest.raises(AssertionError, match="LEAK"):
        _assert_train_rows_unchanged(refits, df)


def test_stateless_mode_is_row_local():
    """Without ``cluster`` nothing is learned: any row's output is the same
    whatever panel the transformer was fit on or applied with."""
    df = _names_panel()
    c = Canonicalizer(entity="id", time="t").fit(df.head(1))
    whole = c.transform(df).collect()
    for i in range(df.height):
        alone = c.transform(df.slice(i, 1)).collect()
        assert alone.row(0) == whole.row(i)


# --------------------------------------------------------------------------- #
# Survivorship
# --------------------------------------------------------------------------- #
def _cluster_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cluster": [1, 1, 1, 2, 2],
            "t": [1, 2, 3, 1, 2],
            "name": ["ACME", "Acme Corporation", None, "b", "bb"],
            "px": [None, 10.0, 11.0, 1.0, None],
            "src": ["x", "y", "z", "y", "x"],
        }
    )


@pytest.mark.parametrize(
    ("rule", "want"),
    [
        ("first", [(1, 1, "ACME", None, "x"), (2, 1, "b", 1.0, "y")]),
        ("last", [(1, 3, None, 11.0, "z"), (2, 2, "bb", None, "x")]),
        ("first_non_null", [(1, 1, "ACME", 10.0, "x"), (2, 1, "b", 1.0, "y")]),
        (
            "last_non_null",
            [(1, 3, "Acme Corporation", 11.0, "z"), (2, 2, "bb", 1.0, "x")],
        ),
        ("longest", [(1, 1, "Acme Corporation", 10.0, "x"), (2, 1, "bb", 1.0, "y")]),
        ("max", [(1, 3, "Acme Corporation", 11.0, "z"), (2, 2, "bb", 1.0, "y")]),
        ("min", [(1, 1, "ACME", 10.0, "x"), (2, 1, "b", 1.0, "x")]),
        # Row 2 of cluster 1 has four non-null fields; ties go to the first.
        (
            "most_complete",
            [(1, 2, "Acme Corporation", 10.0, "y"), (2, 1, "b", 1.0, "y")],
        ),
    ],
)
def test_survivorship_rules(rule, want):
    got = golden_records(_cluster_frame(), by="cluster", order_by="t", default=rule)
    assert got.rows() == want


def test_most_frequent_ignores_nulls_and_breaks_ties_by_first_seen():
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 1, 1, 2, 2],
            "v": [None, None, "b", "a", "a", "q", "p"],
        }
    )
    got = golden_records(df, by="g", default="most_frequent")
    assert got.rows() == [(1, "a"), (2, "q")]


def test_source_priority_prefers_trusted_non_null_values():
    df = _cluster_frame()
    got = golden_records(
        df,
        by="cluster",
        default="source_priority",
        source_col="src",
        source_priority=["x", "y"],  # "z" is unknown -> least trusted
    )
    # Cluster 1, px: x is null, so y's 10.0 wins over z's 11.0.
    assert got.rows() == [(1, 1, "ACME", 10.0, "x"), (2, 2, "bb", 1.0, "x")]


def test_survivorship_mixed_and_callable_rules():
    s = Survivorship(
        {
            "name": "longest",
            "px": "last_non_null",
            "src": lambda c: pl.col(c).str.join("|"),
        },
        default="first",
    )
    got = s.apply(_cluster_frame(), by="cluster", order_by="t")
    assert got.columns == ["cluster", "t", "name", "px", "src"]
    assert got.rows() == [
        (1, 1, "Acme Corporation", 11.0, "x|y|z"),
        (2, 1, "bb", 1.0, "y|x"),
    ]


def test_survivorship_order_and_container():
    df = _cluster_frame()
    newest = golden_records(df.lazy(), by="cluster", order_by="t", descending=True)
    assert newest.rows() == [
        (1, 3, "Acme Corporation", 11.0, "z"),
        (2, 2, "bb", 1.0, "x"),
    ]
    # Groups appear in order of first appearance, not sorted.
    flipped = pl.concat(
        [df.filter(pl.col("cluster") == 2), df.filter(pl.col("cluster") == 1)]
    )
    assert golden_records(flipped, by="cluster")["cluster"].to_list() == [2, 1]
    # No helper column leaks into the output.
    many = golden_records(
        df,
        by=["cluster"],
        rules={
            "name": "most_frequent",
            "px": "most_complete",
            "src": "source_priority",
        },
        source_col="src",
        source_priority=["x"],
    )
    assert many.columns == df.columns


def test_golden_records_equals_survivorship_apply():
    df = _cluster_frame()
    rules = {"name": "longest", "px": "max"}
    a = golden_records(df, by="cluster", rules=rules, order_by="t")
    b = Survivorship(rules).apply(df, by="cluster", order_by="t")
    assert a.equals(b)


def test_survivorship_validation_errors():
    with pytest.raises(ValueError, match="rules"):
        Survivorship({"x": "newest"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="default"):
        Survivorship(default="newest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_priority"):
        Survivorship({"x": "source_priority"})
    with pytest.raises(ValueError, match="source_priority"):
        Survivorship(default="source_priority", source_col="src", source_priority=[])


def test_golden_record_reads_the_whole_cluster():
    """Documented: merging is not point-in-time. Appending a later row to a
    cluster changes its golden record, which is exactly why Deduplicator stops
    claiming ``leakage_safe`` once a Survivorship merge is attached."""
    df = _cluster_frame()
    later = pl.DataFrame(
        {"cluster": [1], "t": [9], "name": ["Acme"], "px": [99.0], "src": ["x"]}
    )
    before = golden_records(df, by="cluster", order_by="t", default="last_non_null")
    after = golden_records(
        pl.concat([df, later]), by="cluster", order_by="t", default="last_non_null"
    )
    assert before.row(0) != after.row(0)
    d = Deduplicator(linkage="component", survivorship=Survivorship())
    assert d.leakage_safe is False


def test_deduplicator_survivorship_merge():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "t": [2, 1, 3, 0],
            "name": ["Acme Corporation"] * 3 + ["Other Co"],
            "px": [None, 10.0, 11.0, 5.0],
            "src": ["x", "y", "z", "x"],
        }
    )
    d = Deduplicator(
        method="exact",
        columns=["name"],
        linkage="component",
        survivorship=Survivorship({"px": "last_non_null", "src": "last"}),
        entity="id",
        time="t",
    )
    out = d.fit_transform(df).collect()
    # The merged record is keyed like the cluster's earliest row (b, t=1);
    # the other columns follow the rules in time order (b, a, c).
    assert out.columns == df.columns
    assert out.rows() == [
        ("b", 1, "Acme Corporation", 11.0, "z"),
        ("d", 0, "Other Co", 5.0, "x"),
    ]
