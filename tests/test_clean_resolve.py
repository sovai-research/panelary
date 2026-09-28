"""Tests for ``panelary.clean`` entity resolution (M5) and string similarity.

Covers the similarity kernels against textbook values and a brute-force
edit-distance oracle (plus ``rapidfuzz`` parity when the ``fuzzy`` extra is
installed), the blocking strategies, :func:`resolve_entities` against a
brute-force all-pairs oracle, and the :class:`EntityResolver` transformer.

The leak-safety contract for the resolver's fitted state (``mapping_`` and
``records_``), cashed out:

1. **Fit on train only.** A test-fold record that bridges two training ids
   merges them under a full-sample fit but not under a fit on train.
2. **Train outputs cannot see the test fold.** Fitting on train and
   transforming the whole panel passes ``assert_no_train_test_leak``; the
   variant that fits on all rows fails it.
3. **Transform never learns.** Ids unseen at fit are matched against the
   stored training records only and are never merged with each other; a
   deliberately leaky subclass that re-resolves at transform time fails.
"""

from __future__ import annotations

import itertools
import string

import numpy as np
import polars as pl
import pytest

from panelary.clean import (
    EntityResolver,
    LSHBlocking,
    connected_components,
    normalize_text,
    resolve_entities,
    string_similarity,
)
from panelary.clean._strsim import (
    _rapidfuzz_available,
    jaro_winkler,
    levenshtein_distance,
)
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.testing import assert_no_train_test_leak


def _lev_reference(a: str, b: str) -> int:
    """Textbook Wagner-Fischer dynamic programme."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- #
# String similarity
# --------------------------------------------------------------------------- #
def test_levenshtein_distance_known_values():
    a = ["kitten", "flaw", "", "abc", "martha", "", "été"]
    b = ["sitting", "lawn", "abc", "", "marhta", "", "ete"]
    got = levenshtein_distance(a, b)
    assert got.dtype == np.int64
    assert got.tolist() == [3, 2, 3, 3, 2, 0, 2]
    assert levenshtein_distance([], []).tolist() == []
    with pytest.raises(ValueError, match="length mismatch"):
        levenshtein_distance(["a"], [])


def test_levenshtein_distance_matches_brute_force():
    rng = np.random.default_rng(7)
    alphabet = list("abcde")

    def word() -> str:
        return "".join(rng.choice(alphabet, rng.integers(0, 12)))

    a = [word() for _ in range(300)]
    b = [word() for _ in range(300)]
    want = [_lev_reference(x, y) for x, y in zip(a, b, strict=True)]
    assert levenshtein_distance(a, b).tolist() == want


@pytest.mark.parametrize(
    ("s1", "s2", "jaro", "jw"),
    [
        ("MARTHA", "MARHTA", 0.944444, 0.961111),
        ("DWAYNE", "DUANE", 0.822222, 0.840000),
        ("DIXON", "DICKSONX", 0.766667, 0.813333),
        ("CRATE", "TRACE", 0.733333, 0.733333),
        ("JELLYFISH", "SMELLYFISH", 0.896296, 0.896296),
    ],
)
def test_jaro_winkler_textbook_values(s1, s2, jaro, jw):
    assert jaro_winkler(s1, s2, winkler=False) == pytest.approx(jaro, abs=1e-6)
    assert jaro_winkler(s1, s2) == pytest.approx(jw, abs=1e-6)
    assert jaro_winkler(s2, s1) == pytest.approx(jw, abs=1e-6)


def test_jaro_winkler_edges():
    assert jaro_winkler("", "") == 1.0
    assert jaro_winkler("abc", "") == 0.0
    assert jaro_winkler("abc", "xyz") == 0.0
    assert jaro_winkler("same", "same") == 1.0
    # The prefix boost only applies above the 0.7 Jaro threshold.
    j = jaro_winkler("ab", "ax", winkler=False)
    assert j <= 0.7
    assert jaro_winkler("ab", "ax") == j


def test_string_similarity_metrics_and_missing_values():
    a = ["kitten", "MARTHA", "acme corp", None, "", "same"]
    b = ["sitting", "MARHTA", "corp acme ltd", "x", "", "same"]
    lev = string_similarity(a, b, metric="levenshtein", backend="numpy")
    assert lev[0] == pytest.approx(1 - 3 / 7)
    assert np.isnan(lev[3])
    assert lev[4] == 1.0 and lev[5] == 1.0
    jw = string_similarity(a, b, metric="jaro_winkler", backend="numpy")
    assert jw[1] == pytest.approx(0.961111, abs=1e-6)
    jac = string_similarity(a, b, metric="token_jaccard")
    assert jac[2] == pytest.approx(2 / 3)
    assert jac[4] == 1.0  # two empty token sets
    ex = string_similarity(a, b, metric="exact")
    assert ex.tolist()[:3] == [0.0, 0.0, 0.0] and ex[5] == 1.0
    assert np.isnan(ex[3])
    # Series input (any dtype castable to String) gives the same answer.
    sa = pl.Series(a, dtype=pl.String).cast(pl.Categorical)
    same = string_similarity(sa, pl.Series(b), metric="levenshtein", backend="numpy")
    np.testing.assert_array_equal(same, lev)
    # Every score is in [0, 1].
    for m in ("levenshtein", "jaro", "jaro_winkler", "token_jaccard", "exact"):
        s = string_similarity(a, b, metric=m, backend="numpy")  # type: ignore[arg-type]
        assert np.all((s[~np.isnan(s)] >= 0) & (s[~np.isnan(s)] <= 1))


def test_string_similarity_validation():
    with pytest.raises(ValueError, match="metric"):
        string_similarity(["a"], ["b"], metric="cosine")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend"):
        string_similarity(["a"], ["b"], backend="c")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length mismatch"):
        string_similarity(["a", "b"], ["b"])
    assert np.isnan(string_similarity([None], [None])).all()


@pytest.mark.skipif(_rapidfuzz_available(), reason="rapidfuzz is installed")
def test_rapidfuzz_backend_without_the_extra_names_it():
    with pytest.raises(ImportError, match=r"panelary\[fuzzy\]"):
        string_similarity(["a"], ["b"], backend="rapidfuzz")
    # "auto" silently falls back to the dependency-free path.
    auto = string_similarity(["kitten"], ["sitting"], backend="auto")
    assert auto[0] == pytest.approx(1 - 3 / 7)


@pytest.mark.parametrize("metric", ["levenshtein", "jaro", "jaro_winkler"])
def test_rapidfuzz_parity(metric):
    pytest.importorskip("rapidfuzz")
    rng = np.random.default_rng(11)
    words = [
        "".join(rng.choice(list("abcdef"), rng.integers(0, 10))) for _ in range(400)
    ]
    a, b = words[:200], words[200:]
    fast = string_similarity(a, b, metric=metric, backend="rapidfuzz")
    slow = string_similarity(a, b, metric=metric, backend="numpy")
    np.testing.assert_allclose(fast, slow, atol=1e-9)


# --------------------------------------------------------------------------- #
# resolve_entities
# --------------------------------------------------------------------------- #
def _vendor_records() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["AAPL US", "APPLE", "aapl.o", "MSFT", "Microsoft Corp", "IBM"],
            "name": [
                "Apple Inc",
                "APPLE INC.",
                "Apple Inc",
                "Microsoft",
                "Microsoft Corp",
                "Intl Business Machines",
            ],
            "cap": [3.0e12, 3.01e12, 2.99e12, 2.5e12, 2.5e12, 1.5e11],
            "country": ["US", "US", "US", "US", "IE", "US"],
        }
    )


def test_resolve_entities_basic():
    got = resolve_entities(
        _vendor_records(), id_col="id", fields={"name": "jaro_winkler"}, threshold=0.9
    )
    assert got.columns == ["id", "resolved_id", "cluster_size"]
    assert got["id"].to_list() == _vendor_records()["id"].to_list()
    assert got["resolved_id"].to_list() == [
        "AAPL US",
        "AAPL US",
        "AAPL US",
        "MSFT",
        "MSFT",
        "IBM",
    ]
    assert got["cluster_size"].to_list() == [3, 3, 3, 2, 2, 1]
    # LazyFrame input gives the same answer.
    lazy = resolve_entities(
        _vendor_records().lazy(),
        id_col="id",
        fields={"name": "jaro_winkler"},
        threshold=0.9,
    )
    assert lazy.equals(got)


def test_resolve_entities_normalisation_matters():
    kw = {"id_col": "id", "fields": {"name": "exact"}, "threshold": 1.0}
    on = resolve_entities(_vendor_records(), **kw)  # type: ignore[arg-type]
    off = resolve_entities(_vendor_records(), normalize=False, **kw)  # type: ignore[arg-type]
    assert on.filter(pl.col("id") == "APPLE")["resolved_id"].item() == "AAPL US"
    assert off.filter(pl.col("id") == "APPLE")["resolved_id"].item() == "APPLE"
    assert off.filter(pl.col("id") == "aapl.o")["resolved_id"].item() == "AAPL US"


def test_resolve_entities_canonical_choices():
    recs = pl.DataFrame(
        {
            "id": ["zeta", "alpha", "zeta", "mid", "mid", "mid"],
            "t": [2, 1, 3, 4, 5, 6],
            "name": ["acme corp", "acme corp", "acme co", "acme corp", "acme", "ac"],
        }
    )
    kw = {"id_col": "id", "fields": {"name": "exact"}, "threshold": 1.0}
    first = resolve_entities(recs, **kw)  # type: ignore[arg-type]
    assert set(first["resolved_id"].to_list()) == {"zeta"}
    by_time = resolve_entities(recs, order_by="t", **kw)  # type: ignore[arg-type]
    assert by_time["id"].to_list() == ["alpha", "zeta", "mid"]
    assert set(by_time["resolved_id"].to_list()) == {"alpha"}
    smallest = resolve_entities(recs, canonical="min", **kw)  # type: ignore[arg-type]
    assert set(smallest["resolved_id"].to_list()) == {"alpha"}
    freq = resolve_entities(recs, canonical="most_frequent", **kw)  # type: ignore[arg-type]
    assert set(freq["resolved_id"].to_list()) == {"mid"}
    assert freq["cluster_size"].to_list() == [3, 3, 3]


def test_resolve_entities_weights_numeric_fields_and_missing_values():
    kw = {
        "id_col": "id",
        "fields": {"name": "exact", "cap": ("numeric", 3.0)},
        "numeric_tolerance": 0.05,
        "threshold": 0.8,
    }

    def resolved(caps: list) -> dict:
        recs = pl.DataFrame(
            {
                "id": list("abcd")[: len(caps)],
                "name": ["acme corp"] * len(caps),
                "cap": caps,
            }
        )
        got = resolve_entities(recs, **kw)  # type: ignore[arg-type]
        return dict(zip(got["id"].to_list(), got["resolved_id"].to_list(), strict=True))

    rid = resolved([100.0, 101.0, 200.0])
    # a ~ b: cap within 1%, score = (1 + 3 * 0.8) / 4 = 0.85.
    assert rid["b"] == "a"
    # c: cap doubles, score = (1 + 3 * 0) / 4 = 0.25.
    assert rid["c"] == "c"
    # d: cap unknown -> the field is skipped and the name alone scores 1.0,
    # so d links to c *and* to a; matching is transitive.
    rid = resolved([100.0, 101.0, 200.0, None])
    assert rid["d"] == "a" and rid["c"] == "a"


def test_exact_blocking_restricts_comparisons():
    recs = _vendor_records()
    kw = {"id_col": "id", "fields": {"name": "jaro_winkler"}, "threshold": 0.9}
    blocked = resolve_entities(recs, blocking=["country"], **kw)  # type: ignore[arg-type]
    # "Microsoft Corp" is in a different country block from "Microsoft".
    assert blocked.filter(pl.col("id") == "Microsoft Corp")["resolved_id"].item() == (
        "Microsoft Corp"
    )
    # OR-blocking with a name-prefix expression restores the pair.
    prefix = pl.col("name").str.slice(0, 4)
    both = resolve_entities(recs, blocking=["country", prefix], **kw)  # type: ignore[arg-type]
    assert both.filter(pl.col("id") == "Microsoft Corp")["resolved_id"].item() == "MSFT"


def test_sorted_neighbourhood_window():
    recs = pl.DataFrame(
        {"id": ["p", "q", "r"], "name": ["acme corp", "acme corp b", "acme corpx"]}
    )
    kw = {"id_col": "id", "fields": {"name": "exact"}, "threshold": 1.0}
    # Sorted: "acme corp", "acme corp b", "acme corpx" -- no exact matches.
    none = resolve_entities(recs, neighbourhood=("name", 2), **kw)  # type: ignore[arg-type]
    assert none["cluster_size"].to_list() == [1, 1, 1]
    recs2 = pl.DataFrame({"id": ["p", "q", "r"], "name": ["acme", "acme b", "acme"]})
    # "acme" (p), "acme" (r) sort adjacent: window 2 already catches them.
    got = resolve_entities(recs2, neighbourhood=("name", 2), **kw)  # type: ignore[arg-type]
    assert got["resolved_id"].to_list() == ["p", "q", "p"]
    recs3 = pl.DataFrame(
        {"id": ["p", "q", "r"], "name": ["acme", "acme b", "acme"]},
    ).with_columns(pl.Series("k", [1, 2, 3]))
    # Keyed by k, p and r are two apart: window 2 misses, window 3 finds.
    far = resolve_entities(recs3, neighbourhood=("k", 2), **kw)  # type: ignore[arg-type]
    assert far["resolved_id"].to_list() == ["p", "q", "r"]
    near = resolve_entities(recs3, neighbourhood=("k", 3), **kw)  # type: ignore[arg-type]
    assert near["resolved_id"].to_list() == ["p", "q", "p"]
    with pytest.raises(ValueError, match="window"):
        resolve_entities(recs, neighbourhood=("name", 1), **kw)  # type: ignore[arg-type]


def _typo_records(n_groups: int = 25, per: int = 4, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    letters = list(string.ascii_lowercase)
    ids, names = [], []
    for g in range(n_groups):
        base = "".join(rng.choice(letters, 14))
        for k in range(per):
            s = list(base)
            if k:  # one substitution per variant
                s[int(rng.integers(0, len(s)))] = str(rng.choice(letters))
            ids.append(f"g{g}_{k}")
            names.append("".join(s))
    return pl.DataFrame({"id": ids, "name": names})


def _brute_force_partition(recs, threshold, metric="levenshtein"):
    names = recs.select(
        normalize_text("name", strip_accents=True, strip_punctuation=True)
    )
    names = names.to_series().to_list()
    n = len(names)
    pairs = list(itertools.combinations(range(n), 2))
    a = [names[i] for i, _ in pairs]
    b = [names[j] for _, j in pairs]
    score = string_similarity(a, b, metric=metric, backend="numpy")
    hit = score >= threshold
    ei = np.array([p[0] for p in pairs], dtype=np.int64)[hit]
    ej = np.array([p[1] for p in pairs], dtype=np.int64)[hit]
    comp = connected_components(n, ei, ej)
    ids = recs["id"].to_list()
    return [ids[c] for c in comp.tolist()]


def test_resolve_entities_matches_brute_force_oracle():
    recs = _typo_records()
    for threshold in (0.85, 0.9, 0.95):
        got = resolve_entities(
            recs, id_col="id", fields={"name": "levenshtein"}, threshold=threshold
        )
        assert got["resolved_id"].to_list() == _brute_force_partition(recs, threshold)


def test_lsh_blocking_recovers_typo_matches():
    recs = _typo_records()
    full = resolve_entities(
        recs, id_col="id", fields={"name": "levenshtein"}, threshold=0.9
    )
    lsh = resolve_entities(
        recs,
        id_col="id",
        fields={"name": "levenshtein"},
        blocking=[LSHBlocking("name", threshold=0.3, num_perm=64, ngram=2)],
        threshold=0.9,
    )
    assert lsh.equals(full)
    # Deterministic under a fixed seed.
    again = resolve_entities(
        recs,
        id_col="id",
        fields={"name": "levenshtein"},
        blocking=[LSHBlocking("name", threshold=0.3, num_perm=64, ngram=2)],
        threshold=0.9,
    )
    assert again.equals(lsh)
    with pytest.raises(ValueError, match="threshold"):
        resolve_entities(
            recs,
            id_col="id",
            fields={"name": "levenshtein"},
            blocking=[LSHBlocking("name", threshold=0.0)],
        )


def test_max_pairs_guards():
    recs = _vendor_records()
    with pytest.raises(ValueError, match="without blocking"):
        resolve_entities(recs, id_col="id", fields={"name": "exact"}, max_pairs=3)
    with pytest.raises(ValueError, match="blocking produced"):
        resolve_entities(
            recs,
            id_col="id",
            fields={"name": "exact"},
            blocking=["country"],
            max_pairs=3,
        )


def test_resolve_entities_validation_errors():
    recs = _vendor_records()
    with pytest.raises(ValueError, match="threshold"):
        resolve_entities(recs, id_col="id", fields={"name": "exact"}, threshold=1.5)
    with pytest.raises(ValueError, match="canonical"):
        resolve_entities(
            recs,
            id_col="id",
            fields={"name": "exact"},
            canonical="last",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="fields"):
        resolve_entities(recs, id_col="id", fields={"name": "cosine"})
    with pytest.raises(ValueError, match="weight"):
        resolve_entities(recs, id_col="id", fields={"name": ("exact", 0.0)})
    with pytest.raises(ValueError, match="not found"):
        resolve_entities(recs, id_col="id", fields={"nope": "exact"})
    with pytest.raises(ValueError, match="exchange"):
        resolve_entities(
            recs, id_col="id", fields={"name": "exact"}, blocking=["exchange"]
        )


# --------------------------------------------------------------------------- #
# EntityResolver: behaviour
# --------------------------------------------------------------------------- #
def _vendor_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["AAPL US"] * 3
            + ["APPLE"] * 3
            + ["MSFT"] * 3
            + ["NEWCO"] * 2
            + ["APPLE INC NEW"] * 2,
            "t": [0, 1, 2, 3, 4, 5, 0, 1, 2, 6, 7, 6, 7],
            "name": ["Apple Inc"] * 3
            + ["APPLE INC."] * 3
            + ["Microsoft"] * 3
            + ["Newco"] * 2
            + ["Apple Inc"] * 2,
            "x": np.arange(13, dtype=np.float64),
        }
    )


def _er(**kw) -> EntityResolver:
    return EntityResolver(
        fields={"name": "jaro_winkler"}, threshold=0.9, entity="id", time="t", **kw
    )


def test_entity_resolver_declares_contract():
    assert issubclass(EntityResolver, PanelTransformer)
    er = _er()
    assert er.panel_safe is False and er.leakage_safe is True


def test_entity_resolver_fit_and_transform():
    df = _vendor_panel()
    train, test = df.filter(pl.col("t") < 6), df.filter(pl.col("t") >= 6)
    er = _er().fit(train)
    assert er.mapping_ is not None and er.records_ is not None
    rid = dict(er.mapping_.select("id", "resolved_id").iter_rows())
    assert rid == {"AAPL US": "AAPL US", "APPLE": "AAPL US", "MSFT": "MSFT"}
    out = er.transform(df).collect()
    # Row order and every non-entity column are untouched.
    assert out.drop("id").equals(df.drop("id"))
    assert (
        out["id"].to_list()
        == ["AAPL US"] * 6 + ["MSFT"] * 3 + ["NEWCO"] * 2 + ["AAPL US"] * 2
    )
    # The unseen "APPLE INC NEW" matched a training record; "NEWCO" kept its id.
    assert er.transform(test).collect()["id"].to_list() == [
        "NEWCO",
        "NEWCO",
        "AAPL US",
        "AAPL US",
    ]


def test_entity_resolver_output_column_and_containers():
    df = _vendor_panel()
    out = _er(output="rid").fit_transform(df).collect()
    assert out.select(df.columns).equals(df)
    assert out["rid"].to_list()[3:6] == ["AAPL US"] * 3
    lazy = _er(output="rid").fit_transform(df.lazy()).collect()
    assert lazy.equals(out)
    pf = PanelFrame(df, entity="id", time="t")
    assert isinstance(_er().fit_transform(pf), PanelFrame)


@pytest.mark.parametrize(
    "blocking",
    [
        # Blocking keys may read columns that are *not* compared fields.
        {"blocking": ["initial"]},
        {"blocking": [pl.col("initial").str.to_uppercase()]},
        {"blocking": [LSHBlocking("name", threshold=0.3, ngram=2)]},
        {"neighbourhood": ("initial", 3)},
    ],
)
def test_entity_resolver_blocked_unseen_matching(blocking):
    df = _vendor_panel().with_columns(
        pl.col("name").str.slice(0, 1).str.to_lowercase().alias("initial")
    )
    train, test = df.filter(pl.col("t") < 6), df.filter(pl.col("t") >= 6)
    er = EntityResolver(
        fields={"name": "jaro_winkler"},
        threshold=0.9,
        entity="id",
        time="t",
        **blocking,
    ).fit(train)
    assert er.mapping_ is not None
    assert er.mapping_["resolved_id"].to_list() == ["AAPL US", "MSFT", "AAPL US"]
    out = er.transform(test).collect()
    assert out.columns == test.columns
    assert out["id"].to_list() == ["NEWCO", "NEWCO", "AAPL US", "AAPL US"]


def test_unseen_matching_only_forms_reference_pairs():
    """With no blocking, transform compares n_ref x n_new pairs -- not all
    (n_ref + n_new) choose 2 -- so the max_pairs guard counts only those."""
    df = _vendor_panel()
    train, test = df.filter(pl.col("t") < 6), df.filter(pl.col("t") >= 6)
    # 3 training records x 2 unseen records = 6 pairs; (3 + 2) choose 2 = 10.
    er = _er(max_pairs=6).fit(train)
    assert er.transform(test).collect().height == test.height
    with pytest.raises(ValueError, match="without blocking"):
        _er(max_pairs=5).fit(train).transform(test).collect()


def test_entity_resolver_validation_errors():
    with pytest.raises(ValueError, match="threshold"):
        EntityResolver(fields={"name": "exact"}, threshold=0.0)
    with pytest.raises(ValueError, match="canonical"):
        EntityResolver(fields={"name": "exact"}, canonical="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend"):
        EntityResolver(fields={"name": "exact"}, backend="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fields"):
        EntityResolver(fields={"name": "cosine"})
    df = _vendor_panel()
    with pytest.raises(ValueError, match="not found"):
        EntityResolver(fields={"nope": "exact"}, entity="id", time="t").fit(df)
    with pytest.raises(RuntimeError, match="not fitted"):
        _er().transform(df)


# --------------------------------------------------------------------------- #
# EntityResolver: leak safety of the fitted map
# --------------------------------------------------------------------------- #
#: A ~ C and B ~ C at 0.85 (one edit in nine), but A !~ B (two edits).
_A, _B, _C = "northwind", "northwold", "northwond"
_CUT = 5


def _bridge_panel(test_name: str = _C) -> pl.DataFrame:
    rows = [("A", t, _A) for t in range(_CUT)] + [("B", t, _B) for t in range(_CUT)]
    rows += [("C", t, test_name) for t in range(_CUT, 2 * _CUT)]
    return pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "t": [r[1] for r in rows],
            "name": [r[2] for r in rows],
            "x": np.arange(len(rows), dtype=np.float64),
        }
    )


def _bridge_er(cls=EntityResolver) -> EntityResolver:
    return cls(
        fields={"name": "levenshtein"},
        threshold=0.85,
        output="rid",
        entity="id",
        time="t",
    )


def test_bridge_premise():
    sim = string_similarity([_A, _B, _A], [_C, _C, _B], metric="levenshtein")
    assert sim[0] >= 0.85 and sim[1] >= 0.85 and sim[2] < 0.85


def test_mapping_is_learned_from_train_only():
    df = _bridge_panel()
    train = df.filter(pl.col("t") < _CUT)
    honest = _bridge_er().fit(train)
    assert honest.mapping_ is not None
    assert dict(honest.mapping_.select("id", "resolved_id").iter_rows()) == {
        "A": "A",
        "B": "B",
    }
    # A full-sample fit lets the test-fold record C bridge A and B.
    full = _bridge_er().fit(df)
    assert full.mapping_ is not None
    assert set(full.mapping_["resolved_id"].to_list()) == {"A"}


def _assert_train_rows_unchanged(op) -> None:
    base = op(_bridge_panel()).filter(pl.col("t") < _CUT)
    pert = op(_bridge_panel(test_name="zzzzzzzzz")).filter(pl.col("t") < _CUT)
    if not base.equals(pert):
        raise AssertionError("LEAK: rewriting the test fold changed train-fold output")


class _ResolvesAtTransform(EntityResolver):
    """Deliberately leaky: re-resolves every id from the transform-time rows."""

    def _transform(self, panel):
        self._fit(panel)
        return super()._transform(panel)


def test_train_outputs_cannot_see_the_test_fold_strings():
    def honest(frame):
        er = _bridge_er().fit(frame.filter(pl.col("t") < _CUT))
        return er.transform(frame).collect()

    def leaky(frame):  # fit on every row, test fold included
        return _bridge_er().fit(frame).transform(frame).collect()

    def refits(frame):  # fit on train, re-resolve at transform
        er = _bridge_er(_ResolvesAtTransform).fit(frame.filter(pl.col("t") < _CUT))
        return er.transform(frame).collect()

    _assert_train_rows_unchanged(honest)
    with pytest.raises(AssertionError, match="LEAK"):
        _assert_train_rows_unchanged(leaky)
    with pytest.raises(AssertionError, match="LEAK"):
        _assert_train_rows_unchanged(refits)


def _cap_panel() -> pl.DataFrame:
    """A (cap 100) and B (cap 104) do not match; test-only C (cap 102) matches
    both, within 2.5% relative difference."""
    n = 10
    ids = ["A"] * n + ["B"] * n + ["C"] * (n // 2)
    t = list(range(n)) * 2 + list(range(n // 2, n))
    cap = [100.0] * n + [104.0] * n + [102.0] * (n // 2)
    return pl.DataFrame(
        {"id": ids, "t": t, "cap": cap, "x": np.arange(len(ids), dtype=np.float64)}
    )


def test_fit_on_train_passes_the_library_leak_harness():
    df = _cap_panel()
    cut = 5

    def make() -> EntityResolver:
        return EntityResolver(
            fields={"cap": "numeric"},
            numeric_tolerance=0.05,
            threshold=0.5,
            output="rid",
            entity="id",
            time="t",
        )

    def honest(frame):
        return make().fit(frame.filter(pl.col("t") < cut)).transform(frame)

    def leaky(frame):  # fit on all rows, test fold included
        return make().fit(frame).transform(frame)

    # Premise: under a full-sample fit C bridges A and B.
    full = make().fit(df).transform(df).collect()
    assert set(full["rid"].to_list()) == {"A"}
    split = (list(range(cut)), list(range(cut, 10)))
    assert_no_train_test_leak(honest, df, split, entity="id", time="t")
    with pytest.raises(AssertionError, match="LEAK"):
        assert_no_train_test_leak(leaky, df, split, entity="id", time="t")


def test_unseen_ids_are_never_merged_with_each_other():
    train = pl.DataFrame(
        {
            "id": ["A", "B"],
            "t": [0, 0],
            "name": ["alpha co", "beta co"],
            "x": [0.0, 1.0],
        }
    )
    test = pl.DataFrame(
        {
            "id": ["U1", "U2", "A"],
            "t": [1, 1, 1],
            "name": ["zeta holdings", "zeta holdings", "alpha co"],
            "x": [2.0, 3.0, 4.0],
        }
    )
    kw = {
        "fields": {"name": "levenshtein"},
        "threshold": 0.9,
        "entity": "id",
        "time": "t",
    }
    out = EntityResolver(**kw).fit(train).transform(test).collect()  # type: ignore[arg-type]
    assert out["id"].to_list() == ["U1", "U2", "A"]
    leaky = _ResolvesAtTransform(**kw).fit(train).transform(test).collect()  # type: ignore[arg-type]
    assert leaky["id"].to_list() == ["U1", "U1", "A"]  # what refitting would do
