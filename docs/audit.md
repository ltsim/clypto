# 2026 Full-Catalog Audit

All 243 classes in `clypto/native/collection/`, reviewed against three questions: is it bloated,
is it the published algorithm, and does it behave like `mealpy-lts` (the pinned reference fork).
Known per-class issues live in [issues](issues.md); this page records the audit and its outcome.

**Outcome: one engine-level bug, fixed.** `after_evolve()` overwrote `g_best` with the best agent of
the *current* population every epoch, with no comparison against the stored value. An algorithm whose
`evolve()` replaces the population without per-agent elitism — how most paper-faithful "Original"
variants are published — therefore lost results it had already found and reported the worse one
(`OriginalESO`: `0.0404` reported for a run that had reached `5e-34`). Both engines now ratchet
(`clypto/optimizer/native/legacy.pyx`, `clypto/optimizer/precompile/base.pyx`) and store `g_best` as
an independent snapshot of the best ever seen (`clone` shares the solution vector, so the vector is
copied: `OriginalHBO` and `OriginalBCO` mutate vectors in place). `g_worst` is deliberately untouched,
as in `mealpy-lts`, which also refreshes only `g_best`. See `CHANGELOG.md`, 2026-10.

Five reviewers working on five different taxonomies rediscovered it independently, with per-epoch
traces. The blast radius is **98 of 243 classes**, not the 15 first flagged: the pre-fix `g_best` was
a *live* population agent, so an algorithm that reads `g_best` back into its own math saw a global
best that other agents in the same epoch had already improved — `mealpy-lts` assigns its best only
after `evolve()` (`optimizer.py:266`). Measured, paired against `mealpy-lts` at 20 seeds
(`benchmarks/bench_correctness.py`): **17 `distribution_mismatch` flags cleared, 0 introduced**; 33
classes now report a better optimum on the golden problem.

## Other findings

+ **`@cy.optimizer` had the same defect twice** (it never calls `after_evolve`) **and passed a literal
  `0`** as its no-improvement counter, so `max_early_stop` could never fire. Both fixed with the
  ratchet.
+ **The six hard-coded `"min"` comparisons** (`OriginalIMODE`, `OriginalNGO`, `OriginalESO`, `C_PSO`,
  `OriginalBCO`) are faithful transcriptions of `mealpy-lts` calling `compare_target` without its
  `minmax` argument. Real maximization defects — in the reference — documented in `docs/issues.md`
  as inherited, **not** "fixed".
+ **Classes flagged because `mealpy-lts` crashes** at this budget (`pop_size=25`), not because clypto
  is wrong: `BaseGA`/`MultiGA`/`SingleGA`, `OriginalMA`, `OriginalDOA`, `OriginalBFO`, `OppoTWO`,
  `OriginalBCO`. Every one confirmed by running the failing `mealpy-lts` class.
+ **Three favorable default divergences** — `DevSARO`/`OriginalSARO` (`mu` 10 vs 15), `ImprovedTLO`
  (`n_teachers` 3 vs 5), `OriginalCEM` (`n_best` 10 vs 20): `mealpy-lts`'s own default falls outside
  its own validator range at `pop_size=25`.
+ **RNG-stream decoupling, not a logic bug**: four classes are still flagged after the fix
  (`OriginalABC`, `OriginalSHIO`, `OriginalTS`, `OriginalPSS`) and are term-for-term identical ports. A
  last-bit floating-point difference flips a data-dependent RNG-count branch and the two streams
  separate for the rest of the run (`OriginalABC` is bit-identical on sphere, measurably different on
  Rastrigin). Three others (`ABFO`, `DevBA`, `OriginalHC`) were misdiagnosed as this and were in fact
  cleared by the ratchet. Quality annotation, not a bug.
+ **Bloat: the catalog is lean.** No clypto-introduced speculative abstraction, reinvented stdlib or
  dead configuration surface across ~28k lines of `.pyx`. What remains is inherited duplication
  (`Agent`/`Population` boilerplate in 5 evolutionary-based files, the TWO boundary block, the AEO
  family, `THRO` scenarios) plus four small
  clypto-introduced items: `OriginalBCO`'s ~16 dead lines and 2 inert hyperparameters, two redundant
  `cls=cy.Population` kwargs (`OriginalEFO`, `OriginalNRO`), the duplicated `evolve()` of
  `EliteMultiGA`/`EliteSingleGA`, and `DevGCO`'s loop-invariant recomputes.
+ **Documentation gaps**: ~25 classes cite only their docstring, never `REFERENCES.md`; `INFO`'s entry
  is a copy-paste of `RUN`'s, `CircleSA` is filed under `CSA`, `BaseEFO`/`BaseHS`/`BaseGCO` name
  classes that do not exist.
+ **Tooling**: `bench_collection.py --mealpy-python` had been silently comparing against nothing since
  `NumberBounds` → `FloatVar`. Fixed.

## Not done

The `REFERENCES.md` gaps, the bloat cleanups above, and the `graphify` rebuild. `graphify-out/` is
blind to `.pyx` (every one of the 243 classes) and still keyed to the pre-`1622c20b` tree; do not use
it for architecture questions about this catalog.

The behavior pins (`tests/test_golden_baseline.py`, `tests/test_golden_scenarios.py`,
`tests/golden/`, `capture.py`) and the per-epoch ratchet test were removed at the maintainer's
direction, so nothing in `tests/` asserts that `g_best` stays monotonic.

## Reproducing

    PYTHONPATH=. uv run --no-sync python benchmarks/bench_correctness.py \
        --mealpy-python <venv-python> --seeds 20 --out correctness.json

`--names-file` scopes a run to one taxonomy. Pairing same-seed runs matters: comparing independent
means and variances buried a regression `OriginalWOA` showed on every seed. The paired t-statistic can
still miss a divergence — `OriginalWCA` matches exactly on 7/20 and 5/20 seeds yet stays under
`tol=3.0`; report the exact-match fraction alongside it.
