"""Calibration for the causal explosive-behaviour detectors: critical values that
do not look ahead.

The BSADF statistic itself is honest -- at endpoint ``t`` it reads only
``y[:t + 1]``.  The *conventional* calibration around it is not, and that is
where every published real-time bubble chronology quietly leaks.  This module
supplies the three calibrations that survive a strict prefix-invariance audit,
and refuses (or loudly gates) the ones that do not.

Three ways to get a threshold, in increasing order of cost
----------------------------------------------------------
1. :func:`training_max_cv` -- the critical value **is** an order statistic of
   the training sample.  Nothing is simulated and nothing after ``T*`` is
   touched.  :func:`training_max_alpha` gives the exact closed-form false
   positive rate over a monitoring horizon, so you know the size you bought.
2. :func:`kurozumi_boundary` -- an analytic detection boundary
   ``g(k/m) = q * (0.73 + 0.93 * log(0.90 + k/m))`` that depends only on
   *elapsed monitoring time* ``k`` relative to the frozen training length
   ``m``.  It contains no data at all, so there is nothing to leak.
3. :func:`mc_table` -- Monte Carlo critical values from the driftless
   random-walk null, tabulated **per endpoint** in one pass.

Leak-safety notes (read before choosing a calibration)
------------------------------------------------------
**Monte Carlo critical values are NOT a leak.**  The null path is
``numpy.cumsum(rng.standard_normal(n))``.  It contains none of your data.  The
resulting critical value is a deterministic function of
``(n, min_window, lag, grid, nrep, seed)`` only.  Simulating it is as innocent
as looking up a number in a printed table -- which is precisely what it is.

**Indexing that table by your own sample length IS a leak.**  The whole point
of :func:`mc_table` is that ``cv[:, t]`` is the critical value for a series of
length ``t + 1``, computed under a null of that same length.  If you instead
compare every endpoint against ``cv[:, T - 1]`` -- the value for your *final*
sample size -- then the threshold applied at an early date depends on how much
data eventually arrived, and the published chronology changes as the sample
grows.  Measured on this null (``min_window = 30``, ``lag = 0``, ``grid = 32``,
``nrep = 2000``, ``seed = 0``), the honest 95% value for a length-100 series is
**0.4331**.  Under the "use the last column" rule it becomes **0.6606** when
``T = 400`` -- a **0.2276** inflation at that date -- and then *moves again* to
**0.6906** when the sample later grows to ``T = 800``, silently revising a
signal that had already been published.  Always compare ``stat[t]`` against
``cv[:, t]``.  :func:`align_cv` exists to make the positional alignment the path
of least resistance.

**The wild bootstrap IS a leak.**  A wild (or residual) bootstrap fits the null
model on the *whole* sample and resamples those residuals, so the volatility of
the entire history -- including the future -- enters the threshold used at every
past date.  Measured: a critical value built from full-sample residuals came out
at **1.150** where the honest real-time value at ``t = 200`` was **0.914**, a
**26%** gap, and the gap is a monotone artefact of the realised volatility path
rather than noise.  This module ships no bootstrap.  If you add one, gate it
behind an explicit ``full_sample=True`` research flag documented as leaking, or
refit the null model on an expanding schedule so the threshold at ``t`` uses
only ``y[:t + 1]``.

**The PSY minimum-window rule is a leak.**  ``r0 = 0.01 + 1.8 / sqrt(T)``, i.e.
``min_window = floor(T * r0)``, is a function of the sample length, so it
mutates every time a new observation arrives.  Measured over 300 null paths
(``lag = 0``, ``grid = 32``, ``seed = 11``): growing ``T`` from 800 to 1600
moves the rule from 58 to 88 observations and revises **41.2%** of the
213 900 already-published (path, date) statistics by more than 0.05 t-units,
with mean ``|delta| = 0.123`` and a maximum of **2.515**.  Rerunning the same
comparison with ``min_window`` frozen at 30 gives a maximum revision of exactly
``0.0``.  ``min_window`` is therefore a **required absolute-integer
hyperparameter** everywhere in :mod:`panelary.detect`.
:func:`psy_min_window` is provided only for replicating published PSY numbers
and refuses to run without an explicit acknowledgement.

The same rule leaks into the calibration: with the PSY window, the 95% critical
value at the fixed calendar date ``t = 100`` is **0.3654** when ``T = 400`` and
**0.3027** when ``T = 600`` -- the threshold at a date in the past depends on
how long you eventually ran the sample.

Why the driftless null is not a shortcut
----------------------------------------
:func:`mc_table` simulates ``y = cumsum(randn)`` with **no drift**.  That is not
laziness, it is the mechanism that makes the one-pass table possible.

* On a null path the BSADF *sequence* is itself point-in-time, so its value at
  index ``n`` **is** the length-``n`` statistic.
* With no drift the paths nest exactly across ``n``: the first ``n`` points of a
  length-``t_max`` driftless random walk are distributed exactly as a
  length-``n`` driftless random walk.  So one simulation of ``nrep`` paths of
  length ``t_max`` yields ``cv[t]`` for **every** ``t`` at once.
* A drift term ``y_t = d * t^(-eta) + ...`` is exactly what destroys that
  nesting, because the drift's contribution at index ``n`` depends on the
  normalisation used for the full length.
* Nothing is lost: for the parameter range that matters here the limit law of
  the sup-ADF family is identical in the drifting and driftless cases, which is
  why PSY tabulate against the driftless null in the first place.

The same trick makes the table *extensible*: draws are generated time-major, so
``mc_table(t_max=800)[:, :400]`` is bitwise equal to ``mc_table(t_max=400)``.
The cache exploits this -- it is keyed on
``(min_window, lag, null, nrep, seed, grid, levels)`` and **never on T** --
and simply slices the widest table computed so far.

References
----------
Phillips, P. C. B., Shi, S. & Yu, J. (2015). "Testing for multiple bubbles:
historical episodes of exuberance and collapse in the S&P 500."
*International Economic Review* **56**(4), 1043-1078.

Kurozumi, E. (2023). "Monitoring for bubbles with a detection boundary."
*Econometrics and Statistics*, DOI 10.1016/j.ecosta.2023.06.007.

Astill, S., Harvey, D. I., Leybourne, S. J. & Taylor, A. M. R.
"Real-time monitoring for explosive financial bubbles" -- the training-sample
maximum whose exceedance probability has a closed form.

All code is clean-room from the published equations.
"""

from __future__ import annotations

import base64
import io
import warnings
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_TABLE_SPEC",
    "align_cv",
    "calibrate_kurozumi_q",
    "clear_table_cache",
    "default_table",
    "kurozumi_boundary",
    "mc_table",
    "psy_min_window",
    "training_max_alpha",
    "training_max_cv",
    "training_max_horizon",
    "verify_default_table",
]

#: The only null this module simulates.  See the module docstring for why a
#: drift term would break the nesting property the one-pass table relies on.
_SUPPORTED_NULLS = ("rw",)


# --------------------------------------------------------------------------- #
# Lazy hooks into Agent A's kernels
# --------------------------------------------------------------------------- #
def _bsadf_panel() -> Any:
    """Import :func:`panelary.detect._bsadf.bsadf_panel` on first use.

    Imported lazily and inside the call so this module stays importable (and
    :func:`kurozumi_boundary` / :func:`training_max_cv` stay usable) even when
    the BSADF kernel is unavailable.
    """
    from panelary.detect._bsadf import bsadf_panel  # noqa: PLC0415

    return bsadf_panel


def _bsadf_sequence() -> Any:
    """Lazily import :func:`panelary.detect._bsadf.bsadf_sequence`."""
    from panelary.detect._bsadf import bsadf_sequence  # noqa: PLC0415

    return bsadf_sequence


# --------------------------------------------------------------------------- #
# Null-path simulation
# --------------------------------------------------------------------------- #
def _null_paths(t_max: int, nrep: int, seed: int, null: str) -> np.ndarray:
    """``(nrep, t_max)`` driftless random-walk paths, extension-stable in ``t_max``.

    Draws are generated **time-major** -- ``standard_normal((t_max, nrep))`` --
    so that entry ``(t, i)`` is always the ``t * nrep + i``-th variate of the
    stream.  Growing ``t_max`` therefore only *appends* rows and leaves every
    existing entry untouched, which is what makes
    ``mc_table(t_max=B)[:, :A] == mc_table(t_max=A)`` hold bitwise.  Drawing
    ``(nrep, t_max)`` row-major would shift every row but the first.

    Parameters
    ----------
    t_max : int
        Path length.
    nrep : int
        Number of independent paths.
    seed : int
        Seed for :func:`numpy.random.default_rng`.
    null : str
        Must be ``"rw"``.

    Returns
    -------
    numpy.ndarray
        ``(nrep, t_max)`` float64 array of driftless random walks starting at
        the first innovation.
    """
    if null not in _SUPPORTED_NULLS:
        raise ValueError(
            f"`null` must be one of {_SUPPORTED_NULLS!r}, got {null!r}. A drift "
            "term destroys the nesting property the one-pass table relies on; "
            "see the module docstring."
        )
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((int(t_max), int(nrep)))
    return np.cumsum(noise, axis=0, dtype=np.float64).T


# --------------------------------------------------------------------------- #
# The one-pass nested Monte Carlo table
# --------------------------------------------------------------------------- #
#: Cache of simulated tables.  Key: ``(min_window, lag, null, nrep, seed, grid,
#: levels)``.  **Never keyed on T.**  Value: ``(t_max, table)`` for the widest
#: ``t_max`` computed so far; narrower requests are served by slicing, which is
#: exact because the draws are extension-stable (see :func:`_null_paths`).
_TABLE_CACHE: dict[tuple, tuple[int, np.ndarray]] = {}


def clear_table_cache() -> None:
    """Drop every cached Monte Carlo table (frees memory; changes no results)."""
    _TABLE_CACHE.clear()


def mc_table(
    *,
    min_window: int,
    lag: int,
    t_max: int,
    nrep: int = 2000,
    seed: int = 0,
    levels: Sequence[float] = (0.90, 0.95, 0.99),
    grid: int | None = 32,
    null: str = "rw",
    use_cache: bool = True,
    use_default_table: bool = True,
) -> np.ndarray:
    """Per-endpoint BSADF critical values from the driftless random-walk null.

    One simulation, every sample length.  ``nrep`` driftless random walks of
    length ``t_max`` are drawn, :func:`~panelary.detect._bsadf.bsadf_panel`
    is run on all of them at once, and the quantiles are taken **down the
    replication axis at each endpoint**.  Because the null path is driftless,
    its length-``n`` prefix is distributed exactly as a length-``n`` null path,
    so column ``t`` of the result is a legitimate length-``t + 1`` critical
    value -- not an approximation of one.

    Parameters
    ----------
    min_window : int
        Minimum window length, in observations, **as an absolute integer**.
        Never derive it from the sample length; see
        :func:`psy_min_window` for why.
    lag : int
        ADF augmentation order.  Must match the ``lag`` used for the statistic
        being calibrated.
    t_max : int
        Number of endpoints to tabulate.  Choose it once, generously, and reuse
        it -- the table is a property of the null, not of your data.
    nrep : int, default=2000
        Number of null paths.  At ``nrep = 199`` the critical value itself is
        seed-dependent noise: measured across 12 seeds at ``min_window = 30``,
        the length-200 95% value had a standard deviation of **0.093** t-units
        and a range of **0.324** -- the same order as the effects being tested.
        At ``nrep = 2000`` the standard deviation drops to **0.045**.  Do not go
        below the default unless you are prototyping.
    seed : int, default=0
        Seed for the null draws.  Explicit, as required by the module contract.
    levels : sequence of float, default=(0.90, 0.95, 0.99)
        Quantile levels.  Row ``i`` of the result corresponds to ``levels[i]``,
        in the order given.
    grid : int or None, default=32
        Window-length ladder passed to the BSADF kernel.  ``None`` means an
        exhaustive sup over all start points.  **Must match** the ``grid`` used
        for the statistic, otherwise the calibration is for a different
        estimator.
    null : str, default="rw"
        Only ``"rw"`` (driftless random walk) is supported.
    use_cache : bool, default=True
        Serve from / populate the module-level table cache.
    use_default_table : bool, default=True
        Allow the shipped precomputed table to satisfy the request when the
        settings match :data:`DEFAULT_TABLE_SPEC` exactly *and* the shipped
        table's fingerprint still matches the installed BSADF kernel.

    Returns
    -------
    numpy.ndarray
        ``(len(levels), t_max)`` float64.  ``cv[:, t]`` is the critical value
        for a series of length ``t + 1``, i.e. it lines up **positionally** with
        ``bsadf_sequence(y)[t]``.  Columns where the statistic is undefined
        (fewer than ``min_window`` observations) are ``nan``.

    Warnings
    --------
    Never index this table by your own sample length.  Comparing every endpoint
    against ``cv[:, T - 1]`` makes the threshold at an early date a function of
    how much data eventually arrived: measured on this null (``min_window =
    30``, ``lag = 0``, the defaults otherwise), the honest length-100 95% value
    is 0.4331, while the last column is 0.6606 at ``T = 400`` and 0.6906 at
    ``T = 800`` -- the threshold for the same date moves as data arrives.  Use
    ``cv[:, t]`` against ``stat[t]``, or :func:`align_cv`.

    See Also
    --------
    align_cv : positional alignment helper.
    kurozumi_boundary : analytic alternative with nothing to simulate.
    training_max_cv : order-statistic alternative with nothing to simulate.

    Notes
    -----
    Cost is one BSADF panel evaluation of shape ``(nrep, t_max)``.  Measured at
    ``nrep = 1500``, ``t_max = 800``, ``min_window = 30``, ``grid = 32``:
    about 7 s on a single core.
    """
    min_window = int(min_window)
    lag = int(lag)
    t_max = int(t_max)
    nrep = int(nrep)
    seed = int(seed)
    levels_t = tuple(float(x) for x in levels)

    if min_window < 3:
        raise ValueError(f"`min_window` must be >= 3, got {min_window}.")
    if t_max < 1:
        raise ValueError(f"`t_max` must be >= 1, got {t_max}.")
    if nrep < 2:
        raise ValueError(f"`nrep` must be >= 2, got {nrep}.")
    if lag < 0:
        raise ValueError(f"`lag` must be >= 0, got {lag}.")
    if not all(0.0 < q < 1.0 for q in levels_t):
        raise ValueError(f"`levels` must lie strictly in (0, 1), got {levels_t!r}.")
    if grid is not None and int(grid) < 2:
        raise ValueError(f"`grid` must be None or >= 2, got {grid!r}.")

    key = (min_window, lag, null, nrep, seed, grid, levels_t)

    if use_cache:
        cached = _TABLE_CACHE.get(key)
        if cached is not None and cached[0] >= t_max:
            return cached[1][:, :t_max].copy()

    if use_default_table:
        shipped = _shipped_for(key, t_max)
        if shipped is not None:
            if use_cache:
                _TABLE_CACHE[key] = (shipped.shape[1], shipped)
            return shipped[:, :t_max].copy()

    paths = _null_paths(t_max, nrep, seed, null)
    stats = np.asarray(
        _bsadf_panel()(paths, min_window=min_window, lag=lag, grid=grid),
        dtype=np.float64,
    )
    if stats.shape != (nrep, t_max):
        raise RuntimeError(
            f"bsadf_panel returned {stats.shape}, expected {(nrep, t_max)}."
        )

    table = _endpoint_quantiles(stats, levels_t)
    if use_cache:
        _TABLE_CACHE[key] = (t_max, table)
    return table.copy()


def _endpoint_quantiles(stats: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    """Per-column quantiles of a ``(nrep, t_max)`` null statistic matrix.

    All-``nan`` columns (endpoints before the statistic is defined) stay
    ``nan`` instead of raising.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        table = np.nanquantile(stats, list(levels), axis=0)
    table = np.atleast_2d(np.asarray(table, dtype=np.float64))
    dead = ~np.any(np.isfinite(stats), axis=0)
    table[:, dead] = np.nan
    return table


def align_cv(table: np.ndarray, n: int) -> np.ndarray:
    """Slice a Monte Carlo table so it lines up positionally with a statistic.

    ``align_cv(cv, len(y))[i, t]`` is the ``levels[i]`` critical value for the
    length-``t + 1`` statistic ``bsadf_sequence(y)[t]``.  This is the *only*
    correct way to use :func:`mc_table` in a chronology: it guarantees that the
    threshold applied at date ``t`` was computed for a null of exactly that
    length, so the published signal at ``t`` never changes when more data
    arrives.

    Parameters
    ----------
    table : numpy.ndarray
        ``(len(levels), t_max)`` table from :func:`mc_table`.
    n : int
        Length of the series being tested.

    Returns
    -------
    numpy.ndarray
        ``(len(levels), n)`` view-copy of the first ``n`` columns.

    Raises
    ------
    ValueError
        If ``n`` exceeds ``t_max``.  Regenerate the table at a larger ``t_max``
        rather than reusing the last column -- reusing it is the leak this
        module exists to prevent.

    Examples
    --------
    >>> import numpy as np
    >>> cv = np.arange(6.0).reshape(2, 3)
    >>> align_cv(cv, 2)
    array([[0., 1.],
           [3., 4.]])
    """
    table = np.asarray(table, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError(f"`table` must be 2-D, got shape {table.shape}.")
    n = int(n)
    if n < 0:
        raise ValueError(f"`n` must be >= 0, got {n}.")
    if n > table.shape[1]:
        raise ValueError(
            f"Table has {table.shape[1]} endpoints but the series has {n}. "
            "Regenerate with a larger `t_max`; do NOT reuse the last column, "
            "which would make the threshold at every date depend on the final "
            "sample size."
        )
    return table[:, :n].copy()


# --------------------------------------------------------------------------- #
# Shipped default table
# --------------------------------------------------------------------------- #
#: Settings the shipped table was generated under.  ``mc_table`` uses the
#: shipped constant only when every one of these matches the request.
DEFAULT_TABLE_SPEC: dict[str, Any] = {
    "min_window": 30,
    "lag": 0,
    "null": "rw",
    "nrep": 2000,
    "seed": 0,
    "grid": 32,
    "levels": (0.90, 0.95, 0.99),
    "t_max": 512,
}

#: Canonical fingerprint path: ``cumsum(default_rng(12345).standard_normal(64))``.
#: The shipped table is accepted only if the installed BSADF kernel reproduces
#: the stored ``fingerprint`` on this path, so a change in the estimator's
#: conventions can never silently reuse a stale calibration.
_FINGERPRINT_SEED = 12345
_FINGERPRINT_LEN = 64

#: Fixed-point scale of the shipped table: values are stored as
#: ``int16(round(cv * 10000))`` with ``-32768`` marking ``nan``, which is ~40%
#: smaller than float32 after compression and exact to 5e-5.
_TABLE_SCALE = 10_000.0
_TABLE_NAN = np.int16(-32768)

#: zlib-compressed ``.npz`` of ``{"cv_q": int16 (3, 512), "fingerprint": float64}``,
#: base64-encoded.  Regenerate with
#: ``python -m panelary.detect._critvals --regen``.
_DEFAULT_TABLE_B64 = """
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
"""

_DEFAULT_TABLE_CACHE: dict[str, np.ndarray] | None = None


def _decode_default() -> dict[str, np.ndarray]:
    """Decode the shipped table blob (once)."""
    global _DEFAULT_TABLE_CACHE
    if _DEFAULT_TABLE_CACHE is None:
        raw = zlib.decompress(base64.b64decode("".join(_DEFAULT_TABLE_B64.split())))
        with np.load(io.BytesIO(raw)) as npz:
            blob = {k: np.asarray(npz[k]) for k in npz.files}
        q = blob["cv_q"]
        cv = q.astype(np.float64) / _TABLE_SCALE
        cv[q == _TABLE_NAN] = np.nan
        _DEFAULT_TABLE_CACHE = {"cv": cv, "fingerprint": blob["fingerprint"]}
    return _DEFAULT_TABLE_CACHE


def default_table() -> np.ndarray:
    """The precomputed table shipped with the package.

    Returns
    -------
    numpy.ndarray
        ``(3, 512)`` float64 critical values for the settings in
        :data:`DEFAULT_TABLE_SPEC`, decoded from a ~4.5 kB base64 constant, so
        the common path needs no simulation at all.  Endpoints before
        ``min_window`` are ``nan``.
    """
    return _decode_default()["cv"].astype(np.float64)


def _fingerprint(bsadf_sequence: Any) -> np.ndarray:
    """BSADF sequence of the canonical fingerprint path, under the given kernel."""
    y = np.cumsum(
        np.random.default_rng(_FINGERPRINT_SEED).standard_normal(_FINGERPRINT_LEN)
    )
    return np.asarray(
        bsadf_sequence(
            y,
            min_window=DEFAULT_TABLE_SPEC["min_window"],
            lag=DEFAULT_TABLE_SPEC["lag"],
            grid=DEFAULT_TABLE_SPEC["grid"],
        ),
        dtype=np.float64,
    )


def _shipped_for(key: tuple, t_max: int) -> np.ndarray | None:
    """Return the shipped table if it is valid for ``key`` and wide enough."""
    spec = DEFAULT_TABLE_SPEC
    want = (
        spec["min_window"],
        spec["lag"],
        spec["null"],
        spec["nrep"],
        spec["seed"],
        spec["grid"],
        tuple(spec["levels"]),
    )
    if key != want or t_max > spec["t_max"]:
        return None
    try:
        blob = _decode_default()
    except Exception:  # pragma: no cover - corrupt/absent blob
        return None
    if "fingerprint" not in blob:
        return None
    try:
        got = _fingerprint(_bsadf_sequence())
    except Exception:  # pragma: no cover - kernel unavailable
        return None
    ref = np.asarray(blob["fingerprint"], dtype=np.float64)
    if got.shape != ref.shape:
        return None
    fin_g, fin_r = np.isfinite(got), np.isfinite(ref)
    if not np.array_equal(fin_g, fin_r):
        return None
    if not np.allclose(got[fin_g], ref[fin_r], rtol=1e-8, atol=1e-8):
        return None
    return blob["cv"].astype(np.float64)


def verify_default_table(*, atol: float = 5e-4) -> dict[str, float]:
    """Re-simulate the shipped table and report the deviation.

    Returns
    -------
    dict
        ``{"max_abs_dev": float, "mean_abs_dev": float, "n_compared": float}``.
        The shipped table is stored as int16 fixed point at 1e-4, so a deviation
        up to ``5e-5`` is pure quantisation.  Anything materially larger means
        the BSADF kernel changed and the blob must be regenerated.
    """
    spec = DEFAULT_TABLE_SPEC
    fresh = mc_table(
        min_window=spec["min_window"],
        lag=spec["lag"],
        t_max=spec["t_max"],
        nrep=spec["nrep"],
        seed=spec["seed"],
        levels=spec["levels"],
        grid=spec["grid"],
        null=spec["null"],
        use_cache=False,
        use_default_table=False,
    )
    ship = default_table()
    finite = np.isfinite(fresh) & np.isfinite(ship)
    dev = np.abs(fresh[finite] - ship[finite])
    out = {
        "max_abs_dev": float(dev.max()) if dev.size else float("nan"),
        "mean_abs_dev": float(dev.mean()) if dev.size else float("nan"),
        "n_compared": float(dev.size),
    }
    if dev.size and out["max_abs_dev"] > atol:
        warnings.warn(
            "The shipped default critical-value table disagrees with a fresh "
            f"simulation by {out['max_abs_dev']:.4g} (> {atol:g}). Regenerate "
            "it: python -m panelary.detect._critvals --regen",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def _encode_default_blob(cv: np.ndarray, fingerprint: np.ndarray) -> str:
    """Build the base64 blob for :data:`_DEFAULT_TABLE_B64` (regeneration only)."""
    arr = np.asarray(cv, dtype=np.float64)
    finite = np.isfinite(arr)
    q = np.where(
        finite, np.round(np.nan_to_num(arr) * _TABLE_SCALE), float(_TABLE_NAN)
    ).astype(np.int16)
    if np.any(np.abs(np.round(arr[finite] * _TABLE_SCALE)) >= 32767):
        raise ValueError("critical values overflow the int16 fixed-point encoding.")
    buf = io.BytesIO()
    np.savez_compressed(
        buf, cv_q=q, fingerprint=np.asarray(fingerprint, dtype=np.float64)
    )
    packed = zlib.compress(buf.getvalue(), 9)
    text = base64.b64encode(packed).decode("ascii")
    return "\n".join(text[i : i + 76] for i in range(0, len(text), 76))


# --------------------------------------------------------------------------- #
# Kurozumi (2023) analytic detection boundary
# --------------------------------------------------------------------------- #
def kurozumi_boundary(k_over_m: np.ndarray, q: float) -> np.ndarray:
    """Kurozumi (2023) monitoring boundary ``g(k/m) = q * (0.73 + 0.93*log(0.90 + k/m))``.

    The causal alternative to any simulated critical value.  ``k`` is the number
    of observations elapsed since monitoring began and ``m`` is the **frozen**
    training length, so the boundary is a function of elapsed monitoring time
    alone.  It reads no data whatsoever -- there is literally nothing in it that
    could leak -- and it is the same boundary whether you stop monitoring
    tomorrow or in ten years, so a signal issued at ``k`` is permanent.

    The shape is deliberately increasing: an unbounded monitoring scheme
    compared against a constant threshold rejects with probability one, so the
    boundary must grow to keep the *horizon* rejection probability at ``alpha``.
    ``q`` is the single scale that pins that down; obtain it from
    :func:`calibrate_kurozumi_q` for your ``(alpha, m, horizon, min_window,
    lag)``.

    Parameters
    ----------
    k_over_m : numpy.ndarray or float
        Elapsed monitoring time divided by the training length, ``k / m``.
        Must be ``> -0.90`` (the log argument); in practice ``k >= 0``.
    q : float
        Scale calibrated so that the rejection probability over the monitoring
        horizon equals ``alpha``.

    Returns
    -------
    numpy.ndarray
        The boundary, same shape as ``k_over_m``.

    Notes
    -----
    A minimum window fraction of ``h = 0.2`` is recommended for the detector
    this boundary calibrates; ``h = 0.1`` is materially size-distorted.  Note
    that deriving ``min_window = floor(h * m)`` from the *frozen training
    length* ``m`` is fine -- ``m`` is chosen once, ex ante, and never moves.
    Deriving it from the current or full sample length is the leak described in
    :func:`psy_min_window`.

    Examples
    --------
    >>> import numpy as np
    >>> float(np.round(kurozumi_boundary(0.0, 1.0), 6))
    0.632015
    >>> float(np.round(kurozumi_boundary(1.0, 1.0), 6))
    1.326924
    >>> bool(np.all(np.diff(kurozumi_boundary(np.linspace(0, 3, 50), 2.0)) > 0))
    True
    """
    x = np.asarray(k_over_m, dtype=np.float64)
    if np.any(x <= -0.90):
        raise ValueError("`k_over_m` must be > -0.90 (log argument 0.90 + k/m).")
    return float(q) * (0.73 + 0.93 * np.log(0.90 + x))


def calibrate_kurozumi_q(
    *,
    m: int,
    horizon: int,
    alpha: float = 0.05,
    min_window: int | None = None,
    h: float = 0.2,
    lag: int = 0,
    grid: int | None = 32,
    nrep: int = 2000,
    seed: int = 0,
    null: str = "rw",
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Find the ``q`` whose monitoring-horizon rejection probability equals ``alpha``.

    Bisects on ``q`` under the driftless random-walk null: a path "rejects" if
    the BSADF statistic crosses :func:`kurozumi_boundary` at **any** monitoring
    index ``k = 1 .. horizon``.  The rejection frequency is monotonically
    decreasing in ``q``, so bisection is exact up to ``tol``.

    Like :func:`mc_table` this is a pure null simulation -- ``cumsum(randn)``,
    no data -- so it is not a leak.  Unlike :func:`mc_table` the resulting ``q``
    is a single scalar that is valid for the whole monitoring run, which is why
    the boundary cannot be mis-indexed.

    Parameters
    ----------
    m : int
        Frozen training length, in observations.
    horizon : int
        Monitoring horizon, in observations after the training sample.
    alpha : float, default=0.05
        Target rejection probability over the whole horizon.
    min_window : int, optional
        Absolute minimum window for the BSADF statistic.  Defaults to
        ``floor(h * m)``, derived from the frozen training length only.
    h : float, default=0.2
        Minimum window fraction used when ``min_window`` is not given.
        ``h = 0.1`` is size-distorted; 0.2 is the recommended floor.
    lag, grid, nrep, seed, null
        As in :func:`mc_table`; must match the statistic being monitored.
    tol : float, default=1e-4
        Bisection tolerance on ``q``.
    max_iter : int, default=60
        Bisection iteration cap.

    Returns
    -------
    float
        The calibrated ``q``.
    """
    m = int(m)
    horizon = int(horizon)
    if m < 8 or horizon < 1:
        raise ValueError(f"Need m >= 8 and horizon >= 1, got m={m}, horizon={horizon}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must lie in (0, 1), got {alpha}.")
    if min_window is None:
        if not 0.0 < h < 1.0:
            raise ValueError(f"`h` must lie in (0, 1), got {h}.")
        min_window = int(np.floor(h * m))
    min_window = int(min_window)
    if min_window < 3:
        raise ValueError(f"Derived `min_window` = {min_window} is too small (< 3).")

    t_max = m + horizon
    paths = _null_paths(t_max, int(nrep), int(seed), null)
    stats = np.asarray(
        _bsadf_panel()(paths, min_window=min_window, lag=int(lag), grid=grid),
        dtype=np.float64,
    )[:, m : m + horizon]

    k = np.arange(1, horizon + 1, dtype=np.float64)
    shape = 0.73 + 0.93 * np.log(0.90 + k / float(m))

    def reject_freq(q: float) -> float:
        exceed = stats > (q * shape)[None, :]
        return float(np.mean(np.any(exceed, axis=1)))

    lo, hi = 0.05, 5.0
    for _ in range(40):
        if reject_freq(hi) <= alpha:
            break
        hi *= 2.0
    for _ in range(40):
        if reject_freq(lo) >= alpha:
            break
        lo *= 0.5
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if reject_freq(mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


# --------------------------------------------------------------------------- #
# Astill-style training-sample maximum
# --------------------------------------------------------------------------- #
def training_max_cv(train_stat: np.ndarray, *, alpha: float | None = None) -> float:
    """Critical value that **is** an order statistic of the training sample.

    Nothing is simulated, no distributional assumption is made beyond
    exchangeability of the statistics under the null, and -- decisively --
    nothing after the end of the training sample is touched.  The threshold is
    fixed the moment monitoring begins and never moves, so a signal issued in
    real time is exactly the signal that appears in the retrospective study.

    Parameters
    ----------
    train_stat : numpy.ndarray
        The detector's statistics over the **training** sample only, e.g.
        ``bsadf_sequence(y[:t_star], min_window=m)``.  Non-finite entries
        (endpoints before the statistic is defined) are dropped.
    alpha : float, optional
        If ``None`` (default), return the maximum -- the Astill et al. choice.
        Otherwise return the ``1 - alpha`` empirical quantile using
        ``method="higher"``, so the result is still a genuine order statistic of
        the training sample rather than an interpolation between two.

    Returns
    -------
    float
        The critical value.

    See Also
    --------
    training_max_alpha : exact false-positive rate over a monitoring horizon.
    training_max_horizon : the horizon that a target ``alpha`` buys you.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array([np.nan, 0.1, 2.0, 1.5, -0.3, 0.9])
    >>> training_max_cv(s)
    2.0
    >>> training_max_cv(s, alpha=0.25)
    1.5
    """
    arr = np.asarray(train_stat, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("`train_stat` contains no finite values.")
    if alpha is None:
        return float(arr.max())
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must lie in (0, 1), got {alpha}.")
    return float(np.quantile(arr, 1.0 - alpha, method="higher"))


def training_max_alpha(*, t_star: int, t_prime: int, m: int) -> float:
    """Closed-form false-positive rate of the training-sample maximum rule.

    Under exchangeability of the detector statistics, using the training-sample
    maximum as the threshold gives an exact horizon rejection probability

    .. math::

        \\alpha = \\frac{T' - T^{*} - m + 1}{T' - 2m + 1}

    -- the fraction of all statistics available up to ``T'`` that fall in the
    monitoring block.  It is a pure counting argument: nothing is simulated and
    nothing depends on the realised data, so you can quote the size of a
    real-time monitor before seeing a single monitoring observation.

    Parameters
    ----------
    t_star : int
        Last index of the training sample.
    t_prime : int
        End of the monitoring horizon, ``t_prime >= t_star``.
    m : int
        Window length of the detector.

    Returns
    -------
    float
        The horizon false-positive rate, clipped to ``[0, 1]``.

    Examples
    --------
    >>> round(training_max_alpha(t_star=200, t_prime=228.47, m=20), 4)
    0.05
    >>> training_max_alpha(t_star=200, t_prime=219, m=20)
    0.0
    """
    denom = float(t_prime) - 2.0 * float(m) + 1.0
    if denom <= 0.0:
        raise ValueError(
            f"Degenerate design: T' - 2m + 1 = {denom} must be positive "
            f"(t_prime={t_prime}, m={m})."
        )
    if float(t_prime) < float(t_star):
        raise ValueError(f"`t_prime` ({t_prime}) must be >= `t_star` ({t_star}).")
    num = float(t_prime) - float(t_star) - float(m) + 1.0
    return float(min(max(num / denom, 0.0), 1.0))


def training_max_horizon(*, t_star: int, alpha: float, m: int) -> float:
    """Invert :func:`training_max_alpha`: the horizon a target ``alpha`` buys.

    .. math::

        T' = \\frac{(T^{*} + m - 1) - \\alpha (2m - 1)}{1 - \\alpha}

    Parameters
    ----------
    t_star : int
        Last index of the training sample.
    alpha : float
        Target horizon false-positive rate, in ``[0, 1)``.
    m : int
        Window length of the detector.

    Returns
    -------
    float
        The monitoring horizon end ``T'``.  Beyond it, the training-maximum rule
        no longer holds size ``alpha``: re-anchor the training sample instead of
        letting the guarantee silently expire.

    Examples
    --------
    >>> round(training_max_horizon(t_star=200, alpha=0.05, m=20), 4)
    228.4737
    >>> round(training_max_alpha(t_star=200,
    ...                          t_prime=training_max_horizon(t_star=200,
    ...                                                       alpha=0.05, m=20),
    ...                          m=20), 10)
    0.05
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"`alpha` must lie in [0, 1), got {alpha}.")
    return float(
        (float(t_star) + float(m) - 1.0 - alpha * (2.0 * float(m) - 1.0))
        / (1.0 - alpha)
    )


# --------------------------------------------------------------------------- #
# The PSY minimum-window rule, quarantined
# --------------------------------------------------------------------------- #
def psy_min_window(nobs: int, *, acknowledge_leak: bool = False) -> int:
    """``floor(T * (0.01 + 1.8 / sqrt(T)))`` -- the PSY rule, which is a leak.

    Provided **only** to reproduce published PSY chronologies.  Do not use it in
    a pipeline.

    Because it is a function of the sample length, the minimum window changes
    every time an observation arrives, which retroactively changes the statistic
    at dates that were already published.  Measured on this null (300 paths,
    ``lag = 0``, ``grid = 32``): growing ``T`` from 800 to 1600 revised
    **41.2%** of already-published statistics by more than 0.05 t-units, with
    mean ``|delta| = 0.123`` and a maximum of ``2.515``.  A
    chronology built this way is not reproducible in real time and its backtest
    is not honest.

    ``min_window`` is a required absolute-integer hyperparameter throughout
    :mod:`panelary.detect`.  Pick it from the economics (the shortest
    episode you care to detect), freeze it, and never let it see ``len(y)``.

    Parameters
    ----------
    nobs : int
        Sample length ``T``.
    acknowledge_leak : bool, default=False
        Must be ``True``.  The flag exists so that the leak is impossible to
        introduce by accident.

    Returns
    -------
    int
        The PSY minimum window for this ``T``.

    Raises
    ------
    ValueError
        If ``acknowledge_leak`` is not ``True``.

    Examples
    --------
    >>> psy_min_window(400, acknowledge_leak=True)
    40
    >>> psy_min_window(800, acknowledge_leak=True)
    58
    >>> psy_min_window(1600, acknowledge_leak=True)
    88
    """
    if not acknowledge_leak:
        raise ValueError(
            "psy_min_window derives the minimum window from the sample length, "
            "which retroactively revises already-published dates (measured: "
            "41.2% of statistics moved by > 0.05 t-units when T grew 800 -> "
            "1600). Pass an "
            "absolute integer `min_window` instead. If you are replicating a "
            "published PSY table, call with acknowledge_leak=True."
        )
    n = int(nobs)
    if n < 4:
        raise ValueError(f"`nobs` must be >= 4, got {n}.")
    return int(np.floor(n * (0.01 + 1.8 / np.sqrt(n))))


# --------------------------------------------------------------------------- #
# Verification (Agent E owns the real test files; this is the smoke harness)
# --------------------------------------------------------------------------- #
def _verify(*, quick: bool = False) -> None:  # pragma: no cover - manual harness
    """Reproduce the PSY reference numbers and assert the calibration invariants."""
    import time

    from panelary.detect._moments import (  # noqa: PLC0415
        cumulative_moments,
        window_adf,
    )

    nrep = 400 if quick else 2000
    print(f"[verify] nrep={nrep}")

    # ---- 1. PSY (2015) finite-sample 95% GSADF cv, T=400, r0=0.10 ---------
    t_len, min_w = 400, 40
    paths = _null_paths(t_len, nrep, 0, "rw")
    t0 = time.perf_counter()
    stats = np.asarray(
        _bsadf_panel()(paths, min_window=min_w, lag=0, grid=None), dtype=np.float64
    )
    t_ex = time.perf_counter() - t0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gsadf = np.nanmax(stats, axis=1)
    g_cv = np.quantile(gsadf, [0.90, 0.95, 0.99])
    print(f"[verify] exhaustive BSADF panel {paths.shape}: {t_ex:.2f}s")
    print(f"[verify] GSADF cv 90/95/99 @ T=400,r0=0.10: {np.round(g_cv, 4)}")
    assert 2.15 <= g_cv[1] <= 2.30, f"PSY 95% GSADF cv off target: {g_cv[1]}"

    # ---- 2. GSADF > SADF > ADF at every level ----------------------------
    ends = np.arange(t_len)
    starts = np.zeros_like(ends)
    try:
        mom = cumulative_moments(paths, lag=0)
        sadf_seq = np.atleast_2d(
            np.asarray(window_adf(mom, starts, ends), dtype=np.float64)
        )
    except Exception:  # kernel may only accept 1-D series
        sadf_seq = np.stack(
            [
                np.asarray(
                    window_adf(cumulative_moments(row, lag=0), starts, ends),
                    dtype=np.float64,
                )
                for row in paths
            ]
        )
    sadf_seq[:, : min_w - 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sadf = np.nanmax(sadf_seq, axis=1)
    adf = sadf_seq[:, -1]
    s_cv = np.quantile(sadf, [0.90, 0.95, 0.99])
    a_cv = np.quantile(adf, [0.90, 0.95, 0.99])
    print(f"[verify] SADF  cv 90/95/99: {np.round(s_cv, 4)}")
    print(f"[verify] ADF   cv 90/95/99: {np.round(a_cv, 4)}")
    assert np.all(g_cv > s_cv), f"GSADF cv not above SADF: {g_cv} vs {s_cv}"
    assert np.all(s_cv > a_cv), f"SADF cv not above ADF: {s_cv} vs {a_cv}"

    # ---- 3. cv rises as the minimum window fraction falls -----------------
    prev = -np.inf
    for frac in (0.30, 0.20, 0.10, 0.05):
        mw = max(int(frac * t_len), 6)
        st = np.asarray(
            _bsadf_panel()(paths, min_window=mw, lag=0, grid=None), dtype=np.float64
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cv95 = float(np.quantile(np.nanmax(st, axis=1), 0.95))
        print(f"[verify] r0={frac:.2f} (min_window={mw:3d}) -> GSADF 95% cv {cv95:.4f}")
        assert cv95 > prev, f"cv not monotone in r0 at {frac}"
        prev = cv95

    # ---- 4. table nesting: wider t_max must not move earlier columns ------
    kw = {
        "min_window": 30,
        "lag": 0,
        "nrep": nrep,
        "seed": 0,
        "grid": 32,
        "use_cache": False,
    }
    small = mc_table(t_max=200, use_default_table=False, **kw)
    big = mc_table(t_max=400, use_default_table=False, **kw)
    a, b = small, big[:, :200]
    both = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    dev = float(np.abs(a[both] - b[both]).max())
    print(f"[verify] nesting max|cv(t_max=200) - cv(t_max=400)[:200]| = {dev:.3e}")
    assert dev == 0.0, "table is not extension-stable"

    # ---- 5. the last-column leak, quantified ------------------------------
    honest_100 = float(small[1, 99])
    naive_100 = float(big[1, 399])
    print(
        f"[verify] cv95 length-100 (honest) {honest_100:.4f} vs last column of a "
        f"T=400 table {naive_100:.4f} -> gap {naive_100 - honest_100:.4f}"
    )
    assert naive_100 > honest_100, "the last-column rule should inflate, not deflate"

    # ---- 5b. the PSY window rule revises already-published statistics -------
    long_paths = _null_paths(1600, min(nrep, 300), 11, "rw")
    mw8 = psy_min_window(800, acknowledge_leak=True)
    mw16 = psy_min_window(1600, acknowledge_leak=True)
    s8 = np.asarray(_bsadf_panel()(long_paths[:, :800], min_window=mw8, lag=0, grid=32))
    s16 = np.asarray(_bsadf_panel()(long_paths, min_window=mw16, lag=0, grid=32))[
        :, :800
    ]
    both = np.isfinite(s8) & np.isfinite(s16)
    delta = np.abs(s8[both] - s16[both])
    print(
        f"[verify] PSY rule T 800->1600 (min_window {mw8}->{mw16}): "
        f"{np.mean(delta > 0.05):.3f} of {both.sum()} published cells revised "
        f"by > 0.05, mean {delta.mean():.3f}, max {delta.max():.3f}"
    )
    f8 = np.asarray(_bsadf_panel()(long_paths[:, :800], min_window=30, lag=0, grid=32))
    f16 = np.asarray(_bsadf_panel()(long_paths, min_window=30, lag=0, grid=32))[:, :800]
    fb = np.isfinite(f8) & np.isfinite(f16)
    frozen_dev = float(np.abs(f8[fb] - f16[fb]).max())
    print(f"[verify] control, min_window frozen at 30: max revision {frozen_dev:.3e}")
    assert frozen_dev == 0.0, "frozen min_window must be exactly prefix-invariant"

    # ---- 6. Kurozumi boundary ---------------------------------------------
    q = calibrate_kurozumi_q(m=200, horizon=200, alpha=0.05, nrep=nrep, seed=1)
    kk = np.arange(1, 201) / 200.0
    bnd = kurozumi_boundary(kk, q)
    print(f"[verify] kurozumi q(alpha=0.05, m=200, horizon=200) = {q:.4f}")
    print(f"[verify] boundary g(0)={bnd[0]:.4f} -> g(1)={bnd[-1]:.4f}")
    assert np.all(np.diff(bnd) > 0)

    # ---- 7. training-max closed form ---------------------------------------
    tp = training_max_horizon(t_star=200, alpha=0.05, m=20)
    back = training_max_alpha(t_star=200, t_prime=tp, m=20)
    print(f"[verify] training-max: T'={tp:.4f} for alpha=0.05 (round trip {back:.6f})")
    assert abs(back - 0.05) < 1e-12

    print("[verify] all checks passed")


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--regen" in sys.argv:
        spec = DEFAULT_TABLE_SPEC
        cv = mc_table(
            min_window=spec["min_window"],
            lag=spec["lag"],
            t_max=spec["t_max"],
            nrep=spec["nrep"],
            seed=spec["seed"],
            levels=spec["levels"],
            grid=spec["grid"],
            null=spec["null"],
            use_cache=False,
            use_default_table=False,
        )
        fp = _fingerprint(_bsadf_sequence())
        print(_encode_default_blob(cv, fp))
    else:
        _verify(quick="--quick" in sys.argv)
