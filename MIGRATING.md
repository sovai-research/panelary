# Migrating to Panelary

As of **0.5.0** the project is called **Panelary**. The import path and the PyPI
distribution are both `panelary`.

| | Before | After |
|---|---|---|
| Install | `pip install polars-features` | `pip install panelary` |
| Import | `import polars_features` | `import panelary as pn` |
| Extras | `polars-features[recommended]` | `panelary[recommended]` |
| Attribution hook | `panelkit_shap_values` | `panelary_shap_values` |
| Env overrides | `PANELKIT_*` | `PANELARY_*` |

**There is no compatibility shim.** `import polars_features` will simply stop
working, and that is deliberate — see [Why no shim?](#why-no-shim) below.

Which section applies to you depends on where you are coming from.

---

## Read this first: what actually shipped where

Only the **0.1 line** was ever published to PyPI, and only under the old name:
`polars-features` 0.1.0, 0.1.5, 0.1.6 and 0.1.7. That release line was a
*maintained fork of functime* — modern packaging, current Polars, green CI — and
nothing more.

Versions **0.2.0, 0.3.0 and 0.4.0 were never published to PyPI under any name.**
Everything people think of as Panelary — `PanelFrame`, the leak-safe `Pipeline`,
CPCV, `detect`, `explain`, `validation`, `econ`, `reduce`, `cluster`, `select`,
`factor` — was built after 0.1.7 and reached the public for the first time as
`panelary` 0.5.0.

So there are exactly two migration stories.

---

## If you are on `polars-features` 0.1.7 (PyPI)

**This is not a drop-in swap.** Do not expect `pip install panelary` plus a
find-and-replace to leave your code working. 0.1.7's API surface was functime's;
0.5.0's is a panel-first library that happens to share that engine underneath.
Budget real time for this, and treat it as adopting a new library that you happen
to already trust.

What you should expect to deal with:

1. **The version number goes backwards in spirit and forwards in fact.** You are
   moving from `polars-features==0.1.7` to `panelary>=0.5.0`. If you pin in a
   requirements file, **rewrite the constraint by hand** — a blind
   `s/polars-features/panelary/` turns `polars-features>=0.1.7` into
   `panelary>=0.1.7`, which resolves to nothing useful and will silently pick up
   whatever the resolver likes.
2. **Dependencies got slim (0.4.0).** The hard dependency set is now exactly
   `numpy` and `polars`. `scikit-learn`, `scipy`, `flaml`, `holidays` and `tqdm`
   moved to extras. To get back the batteries-included behaviour you had on
   0.1.7:

   ```bash
   pip install 'panelary[recommended]'   # ml + scipy + seasonality + cafe
   pip install 'panelary[all]'           # everything optional
   ```

   Without an extra, code paths that need those libraries raise a single
   actionable `pip install 'panelary[<extra>]'` message at first use rather than
   at import.
3. **The Rust extension is gone (0.4.0).** Panelary ships one universal
   `py3-none-any` wheel. No compiler, no per-platform wheels. If you were pinning
   platform wheels or building from source with maturin, delete that machinery.
   The one kernel worth keeping — CUSUM — is pure Python with an optional
   `numba` fast path via `pip install 'panelary[fast]'`.
4. **New APIs, and the old ones still there.** The functime-derived engine
   (feature extractors, forecasting, preprocessing, seasonality, cross-validation,
   metrics, LLM analysis) is still present and importable from `panelary`. Your
   0.1.7 call sites are most likely to survive the move. What is *new* is
   everything layered on top, and the [CHANGELOG](CHANGELOG.md) entries for
   0.2.0 through 0.5.0 are the honest inventory of it.

The pragmatic path: change the install and the import, run your test suite, and
fix what breaks. Then read the changelog to find out what you now have.

---

## If you are tracking a git ref (0.2.x – 0.4.0)

This one *is* close to mechanical. The rename was the only breaking change; the
API is otherwise unchanged from the commit before it.

Four things changed that a rename tool needs to know about:

- `polars_features` → `panelary` (module path)
- `polars-features` → `panelary` (distribution name)
- `panelkit_shap_values` → `panelary_shap_values` (the public attribution hook)
- `PANELKIT_*` → `PANELARY_*` (test-guard env overrides:
  `PANELARY_IMPORT_BUDGET_MS`, `PANELARY_WHEEL_BUDGET_MB`,
  `PANELARY_DETECT_STRICT`)

### The one-liner

Run this from the root of *your* project (not this repo). It works on both GNU
and BSD/macOS `sed`, and leaves `.bak` files behind so you can back out.

```bash
grep -rl --include='*.py' --include='*.pyi' --include='*.ipynb' --include='*.md' \
     -e polars_features -e panelkit_shap_values -e PANELKIT_ . \
| xargs sed -i.bak \
    -e 's/polars_features/panelary/g' \
    -e 's/panelkit_shap_values/panelary_shap_values/g' \
    -e 's/PANELKIT_/PANELARY_/g'
```

Then check, and clean up:

```bash
grep -rn 'polars_features\|polars-features\|panelkit\|PANELKIT_' . --exclude='*.bak'
find . -name '*.bak' -delete
```

### Do the packaging files by hand

Deliberately **not** in the one-liner: `pyproject.toml`, `requirements*.txt`,
`setup.cfg`, `environment.yml`, lock files, Dockerfiles, CI workflows. A blind
substitution there produces broken version constraints (see point 1 above) and
mangles lock-file hashes. Edit the dependency name and the version constraint
together:

```diff
- polars-features>=0.4.0
+ panelary>=0.5.0
```

Then regenerate your lock file rather than editing it.

### If you implemented the attribution hook

If you wrote a custom model that exposes exact attributions to
`explain.TreeAttributor`, the method it looks for is now
`panelary_shap_values`. Rename it. Nothing warns you if you do not — the
attributor will simply fall back as though your model had no native path, and
you will get slower, approximate results with no error.

---

## Labels and purged CV: changes after 0.5.0

Three label/purge fixes change what you see. Folds from `PurgedKFold` and
`CombinatorialPurgedCV` are **byte-identical** to 0.5.0; only the speed of the
`t1=` purge changed.

- **`label.fixed_horizon` raises on an irregular time grid.** It now computes
  its forward return through `factor.forward_return`, the library's single
  audited negative-shift site, and inherits its gap guard: if the per-entity
  time step is not constant (a missing day, or weekends on a business-day
  `Date` panel) it raises `ValueError` instead of silently letting a
  `horizon`-row label span different amounts of time. Outputs on a regular grid
  are byte-identical. To keep the 0.5.0 behaviour on an irregular grid, pass
  `allow_gaps=True`; `t1` still records each label's true end time, which is
  what the purge uses.
- **`label.triple_barrier` gains a `censored` column** (Boolean, appended after
  `t1`). It is `True` for rows whose vertical barrier runs past the entity's
  last row without a touch: 0.5.0 reported those as resolved "no-trend" labels
  (the last row as `label=0, t1=t`), and they change as data arrives. `label`,
  `ret` and `t1` are unchanged; filter with `~pl.col("censored")` before
  training. If you select columns by position, or feed every non-key column to
  a model, account for the new column.
- **A null `t1` at a test time now warns.** The purge cannot test overlap
  against an unknown label span, so a training label overlapping it is kept
  (as in 0.5.0). When a surviving training time is at risk, the splitters emit
  a `UserWarning`; drop unresolved labels before splitting to silence it.

## Upcoming change: `preprocessing.resample` will stamp windows on the right

`resample(freq, agg_method, impute_method)` groups each entity's rows into
windows `[s, s + freq)` and has always stamped each window at its **left** edge
`s` (the polars `group_by_dynamic` default). A value stamped `s` therefore
aggregates data up to `s + freq`: used as a feature, that is a look-ahead of up
to one `freq`.

`resample` now takes `label=`:

| Call | Stamp | Leak-safe as a feature? |
|---|---|---|
| `resample(..., label="right")` | `s + freq`, when the window is complete | yes |
| `resample(..., label="left")` | `s` (the old behaviour) | no |
| `resample(...)` (no `label`) | `s`, **plus a `FutureWarning`** | no |

Nothing changes silently today: omitting `label` keeps the old output and warns.
A future release will make `label="right"` the default, which shifts every
stamped timestamp forward by one `freq` (the aggregated values themselves do not
change). To prepare:

- **Features** (anything a model sees at time `t`): pass `label="right"` now.
- **Code that joins on the old left-edge timestamps** (e.g. calendar labels,
  reports): pass `label="left"` explicitly to keep them and silence the warning.

---

## Why no shim?

A forwarding `polars_features` package that re-exported `panelary` would be cheap
to write, and it would be a lie.

The only thing ever installed from PyPI as `polars_features` was the 0.1.x
functime fork. A shim published today would forward that name to an API which
never existed under it, so `import polars_features` would start returning
something wholly unlike what the last person to type it received. Anyone whose
code happened to keep working would be relying on a compatibility promise nobody
ever made, and anyone whose code broke would get a confusing failure inside a
package they did not install.

A hard `ModuleNotFoundError` is the honest outcome. It tells you exactly what
happened, it happens at import rather than three call frames deep, and it takes
one line to fix.

`polars-features` 0.1.7 stays on PyPI, installable forever, for anyone who
genuinely wants the old fork.

---

## Checklist

- [ ] `pip uninstall polars-features && pip install 'panelary[recommended]'`
- [ ] Run the one-liner over your sources (git-ref users), or change the imports
      by hand (0.1.7 users)
- [ ] Update `pyproject.toml` / `requirements*.txt` by hand — name **and**
      constraint
- [ ] Regenerate lock files
- [ ] Rename `panelkit_shap_values` → `panelary_shap_values` if you implemented it
- [ ] Rename any `PANELKIT_*` environment variables in CI
- [ ] Grep for stragglers: `grep -rn 'polars_features\|polars-features\|panelkit'`
- [ ] Run your tests

## Where to look next

- [CHANGELOG.md](CHANGELOG.md) — what landed in each version, including a marker
  at the point where the names change
- `README.md` — the current quickstart
- The documentation site — the canonical alias is `import panelary as pn`
