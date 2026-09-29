"""Causal detection of explosive regimes, bubbles and changepoints.

The fifth pillar of Panelary. Every statistic here is a function of data up to
``t`` and nothing after it: recomputing a feature at date ``t`` with more data
appended returns the **bitwise identical** value. That is asserted mechanically
in ``tests/test_detect_leak_safety.py``, not merely intended.

What's here
-----------
:mod:`~panelary.detect._moments` -- **the prefix-sum engine**
    :func:`cumulative_moments` accumulates the ADF regression's sufficient
    statistics once in ``O(T k^2)``; every window is then a difference of two
    cumulative sums, so any of the ``~T^2/2`` nested windows costs ``O(1)`` to
    assemble. The identity that removes the last ``O(n)`` term is ``SSR =
    s - beta'b`` (exact because ``A beta = b``), which no published
    implementation exploits -- the fastest reference kernel still forms
    residuals in full and is therefore ``O(T^3)``, not ``O(T^2)``.
    :func:`window_adf` evaluates the whole ``(end x start)`` surface with no
    Python loop; :func:`window_adf_qr` is the accurate fallback.

:mod:`~panelary.detect._bsadf` -- **recursive right-tailed unit root**
    :func:`bsadf_sequence` is the backward sup-ADF: at each endpoint, the
    supremum of the ADF t-statistic over all admissible start dates. Its
    window's right edge is pinned at ``t``, so every regression in the supremum
    terminates at ``t``. ``T=1000`` exhaustive (452,676 windows) in ~0.012 s of
    pure NumPy, against 0.78 s for the fastest published C++ implementation.

:mod:`~panelary.detect._critvals` -- **calibration without leakage**
    :func:`mc_table` exploits the fact that on a driftless random-walk null the
    BSADF sequence is *itself* point-in-time, so its value at index ``n`` **is**
    the length-``n`` statistic and paths nest exactly: one simulation yields
    ``cv[t]`` for every ``t`` at once. :func:`kurozumi_boundary` and
    :func:`training_max_cv` are analytic and training-sample alternatives that
    simulate nothing at all.

:mod:`~panelary.detect._monitors` -- **O(1) sequential detectors**
    :func:`page_cusum` in closed form (Lindley's recursion solves to a
    reflection, ``S - min(cummin(S), 0)``, so it is two cumulative aggregations
    and a subtraction -- see :func:`page_cusum_expr` for the Polars form);
    :func:`shiryaev_roberts` in log space; :func:`focus`, which is provably
    equivalent to running Page's CUSUM at *every* magnitude and *every* window
    length simultaneously with no tuning parameter, at ``O(log n)`` amortised;
    :func:`hb_cusum` with an analytic boundary; :func:`end_of_sample_S`.

:mod:`~panelary.detect._panel` -- **cross-sectional aggregation**
    :func:`breadth` -- the fraction of the cross-section exceeding its
    threshold -- rather than the conventional cross-sectional mean, which is
    hostage to a few extreme names because the statistic diverges
    exponentially under the alternative. :func:`residualise` projects out
    backward-looking factors with betas frozen at ``t``; without it the panel
    null is intractable (see below).

Leak-safety
-----------
Three rules govern the whole package, each with a measured consequence:

**No quantity may depend on the length of the data you happen to hold.** The
conventional minimum-window rule ``floor(T(0.01 + 1.8/sqrt(T)))`` revises 41.2%
of already-published statistics by more than 0.05 when ``T`` grows from 800 to
1600 (300 null paths; mean ``|delta|`` 0.123, max 2.515). ``min_window`` is
therefore a required absolute-integer hyperparameter with no default derived
from the sample.

**Simulated critical values are safe; bootstrapped ones are not.** The Monte
Carlo null is ``cumsum(randn(n))`` -- it contains no data and depends only on
``(n, min_window, lag)``. The wild bootstrap fits its null model on the whole
sample: a critical value from full-sample residuals measured 1.150 where the
value available in real time at ``t=200`` was 0.914.

**Anchor, do not centre.** Prefix sums are invariant to a level shift only if
the anchor is; ``y - y[0]`` is both accurate and constant as data arrives, while
the sample mean moves. Accumulators block-reset every ``block`` rows, which caps
error at ~1e-12 for any series length.

Deliberately refused
--------------------
Full-sample ``GSADF`` as a per-row feature (it is a supremum over the entire
sample by construction; the point-in-time analogue is ``cummax(bsadf)``), and
episode peak / end / duration, which are knowable only once an episode has
ended. :func:`gsadf` returns a scalar for a whole series and
:func:`psy_min_window` requires ``acknowledge_leak=True``.

What this can and cannot do
---------------------------
Detection delay, conditional on detection, is short: median 3-16 observations,
5-13% of a bubble's own duration. **Unconditional power is the binding
constraint** -- a 10-period bubble at ``delta=1.03`` is detected with
probability ~0.2, so short mild episodes are missed entirely.

Cross-sectional pooling is the one real escape, and it is contingent on
:func:`residualise`. With equicorrelated shocks and no bubble anywhere
(``N = 200``, ``T = 300``, family-wise entity threshold), the 95th percentile of
sup-over-``t`` breadth runs 0.010 at ``rho=0``, 0.096 at ``rho=0.6`` and
**0.251 at ``rho=0.9``** -- the effective sample size is roughly ``1/rho``, not
``N``. Project out the common factor first.

References
----------
Written clean-room from the published equations. Phillips, Wu & Yu (2011), *IER*
52(1); Phillips, Shi & Yu (2015), *IER* 56(4) and 56(4) 1079; Page (1954),
*Biometrika* 41; Chu, Stinchcombe & White (1996), *Econometrica* 64;
Homm & Breitung (2012), *JFEc* 10(1); Romano, Eckley, Fearnhead & Rigaill
(2023), *JMLR* 24(81); Kurozumi (2023), *Econometrics and Statistics*;
Astill, Harvey, Leybourne, Sollis & Taylor (2018), *JTSA* 39(6);
Pavlidis et al. (2016), *JREFE* 53(4).

Examples
--------
>>> import numpy as np
>>> from panelary.detect import bsadf_sequence, focus
>>> y = np.cumsum(np.random.default_rng(0).normal(size=400))
>>> stat = bsadf_sequence(y, min_window=50, lag=0)
>>> stat.shape
(400,)
"""

from __future__ import annotations

from panelary.detect._bsadf import (
    bsadf_panel,
    bsadf_sequence,
    gsadf,
    min_admissible_window,
)
from panelary.detect._critvals import (
    DEFAULT_TABLE_SPEC,
    align_cv,
    calibrate_kurozumi_q,
    clear_table_cache,
    default_table,
    kurozumi_boundary,
    mc_table,
    psy_min_window,
    training_max_alpha,
    training_max_cv,
    training_max_horizon,
    verify_default_table,
)
from panelary.detect._moments import (
    Moments,
    cholesky_batch,
    cumulative_moments,
    window_adf,
    window_adf_full,
    window_adf_qr,
)
from panelary.detect._monitors import (
    end_of_sample_S,
    focus,
    hb_cusum,
    page_cusum,
    page_cusum_expr,
    shiryaev_roberts,
    spot_variance,
    subsample_cv,
    volatility_rescale,
)
from panelary.detect._panel import (
    breadth,
    cross_sectional_rank,
    panel_features,
    panel_mean,
    residualise,
    sieve_bootstrap_cv,
)

__all__ = [
    # --- prefix-sum engine -------------------------------------------------
    "Moments",
    "cumulative_moments",
    "window_adf",
    "window_adf_full",
    "window_adf_qr",
    "cholesky_batch",
    # --- recursive right-tailed unit root ----------------------------------
    "bsadf_sequence",
    "bsadf_panel",
    "gsadf",
    "min_admissible_window",
    # --- calibration -------------------------------------------------------
    "mc_table",
    "default_table",
    "verify_default_table",
    "clear_table_cache",
    "DEFAULT_TABLE_SPEC",
    "align_cv",
    "kurozumi_boundary",
    "calibrate_kurozumi_q",
    "training_max_cv",
    "training_max_alpha",
    "training_max_horizon",
    "psy_min_window",
    # --- sequential monitors -----------------------------------------------
    "page_cusum",
    "page_cusum_expr",
    "shiryaev_roberts",
    "focus",
    "hb_cusum",
    "end_of_sample_S",
    "subsample_cv",
    "spot_variance",
    "volatility_rescale",
    # --- panel -------------------------------------------------------------
    "breadth",
    "panel_mean",
    "cross_sectional_rank",
    "residualise",
    "sieve_bootstrap_cv",
    "panel_features",
]
