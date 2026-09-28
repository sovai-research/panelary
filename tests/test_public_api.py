"""The top-level public surface must stay complete.

`panelary.detect` shipped in 0.4.0 with 37 public symbols and a large
test suite, and was unreachable as ``pn.detect`` because nothing added it to
``panelary/__init__.py``. Three separate audits found it independently,
which is the signature of a gap that review does not catch. These tests make it
a build failure instead.

The rule enforced here: every public subpackage is reachable as an attribute of
``panelary``, and is either exported in ``__all__`` (light modules) or
declared in ``_LAZY_SUBMODULES`` (modules whose eager import would break the
light-core guarantee).
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

import panelary as pn

_ROOT = pathlib.Path(pn.__file__).parent

#: Packages that are deliberately not part of the public surface.
_PRIVATE = {
    # Structurally complete (9 modules, ~9.8k lines, an `__init__.py`
    # exporting 38 names, and its own tests) and it ships in the wheel, but it
    # is deliberately NOT wired into `panelary/__init__.py`: it has no entry in
    # `__all__` or `_LAZY_SUBMODULES`, and no docs. It is reachable only by a
    # deep import. Wiring it up is a public-API addition and needs an owner's
    # decision -- until then this opt-out is what keeps the surface honest.
    "evolve",
}

#: Internal plumbing: importable, but not public API and not advertised.
#: `conversion`, `ranges` and `type_aliases` used to live here; they now sit
#: under `panelary/_internal/` (and `forecasting/_conversion.py`), where the
#: underscore already excludes them from the checks below.
_INTERNAL_MODULES = {
    "registry",
}


def _public_subpackages() -> list[str]:
    return sorted(
        p.name
        for p in _ROOT.iterdir()
        if p.is_dir()
        and not p.name.startswith(("_", "."))
        and (p / "__init__.py").exists()
        and p.name not in _PRIVATE
    )


@pytest.mark.parametrize("name", _public_subpackages())
def test_subpackage_is_reachable(name: str) -> None:
    """``pn.<subpackage>`` is wired up: eager, or declared lazy.

    A lazy submodule may still raise on access when its extra is not
    installed (``pn.llm`` without ``[llm]`` is the normal case), so being
    declared in ``_LAZY_SUBMODULES`` is what counts as wired up here. The
    failure this guards against is a subpackage nobody connected at all.
    """
    # Order matters: `hasattr` propagates ImportError (it only swallows
    # AttributeError), so a lazy module whose extra is missing would raise
    # here rather than return False. Check the declaration first.
    assert name in pn._LAZY_SUBMODULES or hasattr(pn, name), (
        f"panelary.{name} is not reachable as pn.{name}. Add it to the "
        f"eager import block in panelary/__init__.py, or to "
        f"_LAZY_SUBMODULES if importing it pulls an optional dependency."
    )


@pytest.mark.parametrize("name", _public_subpackages())
def test_subpackage_is_advertised(name: str) -> None:
    """Every public subpackage is in ``__all__`` or declared lazy."""
    lazy = set(pn._LAZY_SUBMODULES)
    assert name in pn.__all__ or name in lazy, (
        f"panelary.{name} resolves but is invisible: it is neither in "
        f"__all__ nor in _LAZY_SUBMODULES."
    )


def test_lazy_submodules_are_real_and_importable() -> None:
    """Nothing rots in _LAZY_SUBMODULES: each name is a real module."""
    for name in pn._LAZY_SUBMODULES:
        assert (_ROOT / name).is_dir() or (_ROOT / f"{name}.py").exists(), (
            f"_LAZY_SUBMODULES lists {name!r}, which is not a module."
        )


def test_dir_includes_lazy_submodules() -> None:
    """``dir(pn)`` advertises the lazy names, so tab-completion finds them."""
    listed = set(dir(pn))
    assert set(pn._LAZY_SUBMODULES) <= listed


def test_detect_is_exported() -> None:
    """Regression: `detect` is 0.4.0's headline feature and was unreachable."""
    assert "detect" in pn.__all__
    assert pn.detect is importlib.import_module("panelary.detect")


def test_backtesting_imports_at_all() -> None:
    """Regression: a circular import made `backtesting` unimportable by any path.

    ``backtesting`` imports ``forecasting._reduction``; ``forecasting/__init__``
    imports ``elite``; ``elite`` imported ``backtesting`` at module scope. The
    cycle meant ``import panelary.backtesting`` raised ImportError no
    matter how it was reached.
    """
    pytest.importorskip("sklearn")  # backtesting -> forecasting needs the `ml` extra
    mod = importlib.import_module("panelary.backtesting")
    assert hasattr(mod, "backtest")


def test_public_modules_are_reachable() -> None:
    """Top-level public .py modules resolve too (excluding internal plumbing)."""
    # Skip the lazy ones: `hasattr` propagates ImportError, so probing
    # e.g. `plotting` without the `viz` extra raises instead of returning
    # False. test_subpackage_is_reachable already covers those by name.
    unreachable = [
        p.stem
        for p in _ROOT.glob("*.py")
        if not p.stem.startswith("_")
        and p.stem not in _INTERNAL_MODULES
        and p.stem not in pn._LAZY_SUBMODULES
        and not hasattr(pn, p.stem)
    ]
    assert not unreachable, f"unreachable public modules: {unreachable}"
