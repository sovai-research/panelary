"""Canonicalisation: make equal things *look* equal before deduplication.

"ACME Corp.", "Acme  Corp" and "acme corp" are one company to a human and
three to ``==``. Canonicalisation runs **before** dedup (ordering doctrine,
stage 2) so variants collapse onto one key. Everything here is Polars ``.str``
expressions -- no Python loop over rows.

* :func:`normalize_text` -- Unicode normal form, case folding, accent
  stripping, punctuation removal, whitespace collapsing and ordered regex
  replacements, as one expression.
* :func:`fingerprint` / :func:`ngram_fingerprint` -- key-collision keys in the
  style popularised by OpenRefine (reimplemented from the published
  description): normalise, tokenise, de-duplicate and sort tokens (or
  character n-grams), so word order and repetition stop mattering.
* :func:`fingerprint_clusters` -- group a column's distinct values by key.
* :class:`Canonicalizer` -- a :class:`~panelary.core.protocol.PanelTransformer`
  applying all of the above; with ``cluster=`` it *learns* which spelling
  represents each key, from the training rows only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import polars as pl

from panelary.clean._common import check_choice, resolve_columns
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

__all__ = [
    "Canonicalizer",
    "fingerprint",
    "fingerprint_clusters",
    "ngram_fingerprint",
    "normalize_text",
]

UnicodeForm = Literal["NFC", "NFKC", "NFD", "NFKD"]
_FORMS: tuple[str, ...] = ("NFC", "NFKC", "NFD", "NFKD")
_PUNCT = r"[\p{P}\p{S}]"
_MARKS = r"\p{Mn}"


def normalize_text(
    expr: pl.Expr | str,
    *,
    unicode: UnicodeForm | None = "NFC",
    casefold: bool = True,
    strip_accents: bool = False,
    strip_punctuation: bool = False,
    collapse_whitespace: bool = True,
    replacements: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> pl.Expr:
    """Build one normalising expression over a string column.

    Steps run in this order: Unicode normal form -> accent stripping ->
    case folding -> ordered regex ``replacements`` -> punctuation removal ->
    whitespace collapsing (trim + runs of whitespace to one space).

    Parameters
    ----------
    expr : polars.Expr | str
        The string column (name or expression).
    unicode : {"NFC", "NFKC", "NFD", "NFKD"} or None, default="NFC"
        Unicode normal form; ``None`` skips it.
    casefold : bool, default=True
        Lower-case (Polars' Unicode-aware ``to_lowercase``; full case folding
        such as German sharp s -> "ss" is not applied).
    strip_accents : bool, default=False
        Decompose (NFKD) and drop combining marks: "Société" -> "Societe".
    strip_punctuation : bool, default=False
        Replace Unicode punctuation and symbols with a space.
    collapse_whitespace : bool, default=True
        Trim and collapse whitespace runs to a single space.
    replacements : mapping or sequence of (pattern, replacement), optional
        Regex substitutions applied in order, e.g.
        ``{r"\\bcorporation\\b": "corp"}``.

    Returns
    -------
    polars.Expr
        The normalised string expression (keeps the input's output name).
    """
    e = pl.col(expr) if isinstance(expr, str) else expr
    e = e.cast(pl.String)
    if unicode is not None:
        check_choice("unicode", unicode, _FORMS)
        e = e.str.normalize(unicode)
    if strip_accents:
        e = e.str.normalize("NFKD").str.replace_all(_MARKS, "")
    if casefold:
        e = e.str.to_lowercase()
    pairs = (
        list(replacements.items())
        if isinstance(replacements, Mapping)
        else list(replacements or [])
    )
    for pattern, repl in pairs:
        e = e.str.replace_all(pattern, repl)
    if strip_punctuation:
        e = e.str.replace_all(_PUNCT, " ")
    if collapse_whitespace:
        e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    return e


def fingerprint(expr: pl.Expr | str) -> pl.Expr:
    """Key-collision fingerprint: order- and repetition-insensitive tokens.

    Normalise (NFKD, accents stripped, lower-cased, punctuation removed),
    split on whitespace, de-duplicate, sort and re-join with single spaces.
    "Acme, Corp." / "corp ACME" / "ACME   CORP ACME" share the key
    ``"acme corp"``.

    Parameters
    ----------
    expr : polars.Expr | str
        A string column.

    Returns
    -------
    polars.Expr
    """
    base = normalize_text(
        expr, unicode="NFKC", strip_accents=True, strip_punctuation=True
    )
    return base.str.split(" ").list.unique().list.sort().list.join(" ")


def ngram_fingerprint(expr: pl.Expr | str, n: int = 2) -> pl.Expr:
    """Character n-gram fingerprint: robust to spacing and small typos.

    Normalise, drop all whitespace and punctuation, take the distinct
    character ``n``-grams, sort and concatenate. "Krzysztof" and
    "Krzystof" differ in one bigram, which is often enough to collide
    after the fuller spelling wins by frequency.

    Parameters
    ----------
    expr : polars.Expr | str
        A string column.
    n : int, default=2
        n-gram length (``n=1`` gives the sorted distinct characters).

    Returns
    -------
    polars.Expr
    """
    if n < 1:
        raise ValueError(f"`n` must be >= 1, got {n}.")
    base = normalize_text(
        expr, unicode="NFKC", strip_accents=True, strip_punctuation=True
    ).str.replace_all(r"\s+", "")
    chars = base.str.split("")
    if n == 1:
        grams = chars
    else:
        grams = chars.list.eval(
            pl.concat_str([pl.element().shift(-k) for k in range(n)]).drop_nulls()
        )
    short = base.str.len_chars() < n
    keyed = grams.list.unique().list.sort().list.join("")
    return pl.when(short).then(base).otherwise(keyed)


def _key_expr(col: str, method: str, n: int) -> pl.Expr:
    return fingerprint(col) if method == "fingerprint" else ngram_fingerprint(col, n)


def fingerprint_clusters(
    df: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    method: Literal["fingerprint", "ngram"] = "fingerprint",
    n: int = 2,
    order_by: str | None = None,
) -> pl.DataFrame:
    """Group the distinct values of ``column`` by fingerprint key.

    Parameters
    ----------
    df : polars.DataFrame | polars.LazyFrame
        Input records.
    column : str
        String column to cluster.
    method : {"fingerprint", "ngram"}, default="fingerprint"
        Which key to use.
    n : int, default=2
        n-gram length for ``method="ngram"``.
    order_by : str, optional
        Column giving first-appearance order for tie-breaks (e.g. time).

    Returns
    -------
    polars.DataFrame
        One row per distinct value: ``value``, ``key``, ``count`` and
        ``canonical`` (the key's most frequent value; ties go to the value
        seen first, then the lexicographically smaller). Sorted by key, then
        count descending.
    """
    check_choice("method", method, ("fingerprint", "ngram"))
    frame = df.collect() if isinstance(df, pl.LazyFrame) else df
    if order_by is not None:
        frame = frame.sort(order_by, maintain_order=True)
    vals = (
        frame.select(pl.col(column).cast(pl.String).alias("value"))
        .with_row_index("__pos")
        .filter(pl.col("value").is_not_null())
        .group_by("value")
        .agg(pl.len().alias("count"), pl.col("__pos").min().alias("__first"))
        .with_columns(_key_expr("value", method, n).alias("key"))
    )
    canon = (
        vals.sort(
            ["key", "count", "__first", "value"],
            descending=[False, True, False, False],
        )
        .group_by("key", maintain_order=True)
        .agg(pl.col("value").first().alias("canonical"))
    )
    return (
        vals.join(canon, on="key", how="left")
        .sort(["key", "count", "__first"], descending=[False, True, False])
        .select("value", "key", "count", "canonical")
    )


class Canonicalizer(PanelTransformer):
    """Normalise string columns, map crosswalks, and collapse spelling variants.

    Two layers:

    1. **Stateless normalisation** (always) -- :func:`normalize_text` plus an
       optional per-column ``mapping`` crosswalk (e.g. exchange codes,
       country names). Deterministic and row-local; it learns nothing.
    2. **Learned clustering** (``cluster="fingerprint"`` or ``"ngram"``) --
       at :meth:`fit` each column's values are grouped by fingerprint key and
       the most frequent spelling *in the training rows* becomes the key's
       canonical form. :meth:`transform` rewrites every value whose key was
       seen in training; unseen keys pass through normalised but unchanged.
       Nothing is ever learned from transform-time data.

    Parameters
    ----------
    columns : str | sequence of str, optional
        Columns to canonicalise. Default: every non-key ``String`` column.
    unicode, casefold, strip_accents, strip_punctuation, collapse_whitespace, replacements
        See :func:`normalize_text`.
    mapping : mapping of column -> mapping, optional
        Crosswalks applied after normalisation, e.g.
        ``{"country": {"u.s.": "us", "united states": "us"}}``. Keys are
        matched against the *normalised* value.
    cluster : {"fingerprint", "ngram"} or None, default=None
        Learn canonical spellings by key collision.
    ngram : int, default=2
        n-gram length for ``cluster="ngram"``.
    suffix : str, optional
        Write results to ``f"{col}{suffix}"`` instead of replacing in place.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        ``True`` -- row-local rewriting; no rows are combined. Set to
        ``False`` on the instance at :meth:`fit` when the entity column itself
        is canonicalised in place (no ``suffix``), since that can merge
        entities.
    leakage_safe : bool
        ``True`` -- the only learned state (the canonical spellings) comes
        from the fit panel.
    feature_names_in_ : list of str
        Columns canonicalised.
    vocab_ : dict of str -> polars.DataFrame
        Per column, the learned ``key`` -> ``canonical`` table (empty without
        ``cluster``).

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": [1, 2, 3, 4],
    ...         "t": [1, 1, 2, 2],
    ...         "name": ["ACME  Corp.", "acme corp", "Acme Corp", "Corp. ACME"],
    ...     }
    ... )
    >>> Canonicalizer(cluster="fingerprint", entity="id", time="t").fit_transform(
    ...     df
    ... ).collect()["name"].to_list()
    ['acme corp', 'acme corp', 'acme corp', 'acme corp']
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        columns: str | Sequence[str] | None = None,
        unicode: UnicodeForm | None = "NFC",
        casefold: bool = True,
        strip_accents: bool = False,
        strip_punctuation: bool = False,
        collapse_whitespace: bool = True,
        replacements: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        mapping: Mapping[str, Mapping[str, str]] | None = None,
        cluster: Literal["fingerprint", "ngram"] | None = None,
        ngram: int = 2,
        suffix: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if unicode is not None:
            check_choice("unicode", unicode, _FORMS)
        if cluster is not None:
            check_choice("cluster", cluster, ("fingerprint", "ngram"))
        if ngram < 1:
            raise ValueError(f"`ngram` must be >= 1, got {ngram}.")
        self.columns = columns
        self.unicode = unicode
        self.casefold = casefold
        self.strip_accents = strip_accents
        self.strip_punctuation = strip_punctuation
        self.collapse_whitespace = collapse_whitespace
        self.replacements = replacements
        self.mapping = {k: dict(v) for k, v in (mapping or {}).items()}
        self.cluster = cluster
        self.ngram = ngram
        self.suffix = suffix
        self.feature_names_in_: list[str] = []
        self.vocab_: dict[str, pl.DataFrame] = {}

    def _normalised(self, col: str) -> pl.Expr:
        e = normalize_text(
            col,
            unicode=self.unicode,
            casefold=self.casefold,
            strip_accents=self.strip_accents,
            strip_punctuation=self.strip_punctuation,
            collapse_whitespace=self.collapse_whitespace,
            replacements=self.replacements,
        )
        if col in self.mapping:
            e = e.replace(self.mapping[col])
        return e

    def _fit(self, panel: PanelFrame) -> None:
        if self.columns is None:
            schema = panel.schema
            cols = [
                c
                for c in panel.columns
                if c not in (panel.entity_col, panel.time_col)
                and schema[c] == pl.String
            ]
        else:
            cols = resolve_columns(
                panel.columns, "", "", self.columns, what="canonicalize"
            )
        self.feature_names_in_ = cols
        # Rewriting the entity key in place can merge two entities' rows, so
        # the instance stops claiming panel safety (writing to a suffix cannot).
        self.panel_safe = not (self.suffix is None and panel.entity_col in cols)
        self.vocab_ = {}
        if self.cluster is None or not cols:
            return
        frame = (
            panel.lazy()
            .select([pl.col(panel.time_col), *(self._normalised(c) for c in cols)])
            .collect()
        )
        for c in cols:
            table = fingerprint_clusters(
                frame, c, method=self.cluster, n=self.ngram, order_by=panel.time_col
            )
            self.vocab_[c] = table.select("key", "canonical").unique(
                "key", keep="first", maintain_order=True
            )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.feature_names_in_ if c not in panel]
        if missing:
            raise ValueError(
                f"Canonicalizer.transform: fitted column(s) {missing} are missing."
            )
        lf = panel.lazy()
        for c in self.feature_names_in_:
            out = c if self.suffix is None else f"{c}{self.suffix}"
            norm = self._normalised(c)
            vocab = self.vocab_.get(c)
            if vocab is None or vocab.height == 0:
                lf = lf.with_columns(norm.alias(out))
                continue
            key = _key_expr(
                "__panelary_norm", self.cluster or "fingerprint", self.ngram
            )
            lookup = vocab.lazy().rename(
                {"key": "__panelary_key", "canonical": "__panelary_canon"}
            )
            # Row order is restored explicitly: join order is not a contract.
            lf = (
                lf.with_row_index("__panelary_pos")
                .with_columns(norm.alias("__panelary_norm"))
                .with_columns(key.alias("__panelary_key"))
                .join(lookup, on="__panelary_key", how="left")
                .sort("__panelary_pos")
                .with_columns(
                    pl.coalesce("__panelary_canon", "__panelary_norm").alias(out)
                )
                .drop(
                    "__panelary_pos",
                    "__panelary_norm",
                    "__panelary_key",
                    "__panelary_canon",
                )
            )
        return PanelFrame(
            lf, entity=panel.entity_col, time=panel.time_col, validate=False
        )
