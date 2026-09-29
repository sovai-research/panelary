"""Lazy numba dispatch for the optional ``fast`` extra.

Every numba kernel in panelary follows one pattern, and this module is it:

* numba is never imported at module scope. A kernel is compiled on its first
  use and the compiled function is cached for the process (``cache=True`` also
  caches it on disk across processes).
* Flags are fixed. ``error_model="numpy"`` is load-bearing: numba's default
  raises ``ZeroDivisionError`` on float division by zero, while the numpy twin
  relies on IEEE ``inf``/``nan``. ``fastmath`` is never set, because it permits
  reassociation and breaks bitwise parity with the twin.
* Every kernel has a numpy (or pure-Python) twin, and the two must agree
  **bitwise** unless the owning module documents a tolerance and the reason.
  :func:`assert_backend_parity` is the check.
* numba is a speed-up, never a feature: with numba missing, or inside
  :func:`force_numpy`, :meth:`LazyKernel.compiled` returns ``None`` and the
  caller runs its twin.

Usage::

    from panelary._internal._jit import lazy_njit

    @lazy_njit
    def _sweep(x, out):  # numba-compatible Python
        ...

    kernel = _sweep.compiled()  # numba function, or None
    if kernel is not None:
        kernel(x, out)
    else:
        _sweep_numpy(x, out)

Leaf module: standard library and numpy only.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from panelary._internal._deps import have

__all__ = [
    "LazyKernel",
    "assert_backend_parity",
    "force_numpy",
    "lazy_njit",
    "numba_available",
]

#: The fixed compilation flags. ``parallel`` is opt-in per kernel.
_BASE_OPTIONS: dict[str, Any] = {"cache": True, "error_model": "numpy"}

_forced_off = threading.local()
_have_numba: bool | None = None


def _is_forced_off() -> bool:
    return bool(getattr(_forced_off, "depth", 0))


def numba_available() -> bool:
    """True if numba is importable and not disabled by :func:`force_numpy`."""
    global _have_numba
    if _is_forced_off():
        return False
    if _have_numba is None:
        _have_numba = have("numba")
    return _have_numba


@contextmanager
def force_numpy() -> Iterator[None]:
    """Make every :class:`LazyKernel` report ``compiled() is None`` in this thread.

    For parity tests and reproducibility audits: code under the block runs its
    numpy twins even when numba is installed. Nests safely.
    """
    _forced_off.depth = getattr(_forced_off, "depth", 0) + 1
    try:
        yield
    finally:
        _forced_off.depth -= 1


class LazyKernel:
    """A numba-compatible Python function compiled on first use.

    Parameters
    ----------
    py_func : callable
        The kernel, written in the numba-supported subset of Python/numpy.
        Define it at module scope so ``cache=True`` can key it.
    parallel : bool, default False
        Compile with ``parallel=True`` (for kernels that use ``numba.prange``).
    **options
        Extra ``numba.njit`` options. ``fastmath`` is rejected.
    """

    def __init__(
        self, py_func: Callable[..., Any], *, parallel: bool = False, **options: Any
    ) -> None:
        if options.get("fastmath"):
            raise ValueError(
                "fastmath is not allowed: it breaks bitwise parity with the numpy twin."
            )
        self.py_func = py_func
        self._options = {**_BASE_OPTIONS, **options}
        if parallel:
            self._options["parallel"] = True
        self._compiled: Callable[..., Any] | None = None
        self._lock = threading.Lock()
        self.__name__ = getattr(py_func, "__name__", "kernel")
        self.__doc__ = py_func.__doc__

    def compiled(self) -> Callable[..., Any] | None:
        """The numba-compiled kernel, or ``None`` if numba is unavailable/disabled."""
        if not numba_available():
            return None
        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    import numba  # noqa: PLC0415  (lazy, optional-extra import)

                    self._compiled = numba.njit(**self._options)(self.py_func)
        return self._compiled

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the compiled kernel if available, else the Python function itself.

        Calling the Python loop is correct but slow; callers with a vectorised
        numpy twin should dispatch on :meth:`compiled` instead.
        """
        kernel = self.compiled()
        if kernel is not None:
            return kernel(*args, **kwargs)
        return self.py_func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"LazyKernel({self.__name__})"


def lazy_njit(
    func: Callable[..., Any] | None = None, *, parallel: bool = False, **options: Any
) -> Any:
    """Decorator form of :class:`LazyKernel` (usable bare or with options)."""
    if func is not None:
        return LazyKernel(func, parallel=parallel, **options)

    def wrap(f: Callable[..., Any]) -> LazyKernel:
        return LazyKernel(f, parallel=parallel, **options)

    return wrap


def assert_backend_parity(
    fast: Any, twin: Any, *, rtol: float = 0.0, atol: float = 0.0
) -> None:
    """Assert two backend outputs agree (bitwise by default).

    Accepts arrays, scalars, or tuples/lists of them. With ``rtol == atol == 0``
    the check is exact equality with matching dtypes, ``nan`` equal to ``nan``.
    """
    if isinstance(fast, (tuple, list)):
        if not isinstance(twin, (tuple, list)) or len(fast) != len(twin):
            raise AssertionError("backend outputs have different structure")
        for a, b in zip(fast, twin, strict=True):
            assert_backend_parity(a, b, rtol=rtol, atol=atol)
        return
    a = np.asarray(fast)
    b = np.asarray(twin)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if rtol == 0.0 and atol == 0.0:
        if a.dtype != b.dtype:
            raise AssertionError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
        is_float = a.dtype.kind in "fc"
        if not np.array_equal(a, b, equal_nan=is_float):
            same = (a == b) | (np.isnan(a) & np.isnan(b)) if is_float else (a == b)
            bad = int(np.sum(~same))
            raise AssertionError(f"backends differ bitwise at {bad} positions")
        return
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
