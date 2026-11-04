# Known Issues

This page collects everything you should know **before** selecting or
benchmarking an algorithm, organized by the same taxonomy used throughout the
catalog. Issues fall into four groups:

| Label | Meaning |
| --- | --- |
| **Plagiarism** | The method is suspected of duplicating another algorithm's mathematics under a new name. Avoid in comparative studies. |
| **Quality** | The original implementation is known to be weak or sensitive; a developed variant is usually preferred. |
| **Fixed** | A bug that was found and repaired during the 2026 Cython refactor. Kept here for traceability. |
| **Open** | A defect present in the current build that has not been fixed yet. |

Which algorithms evaluate a whole batch at once, and which are order-dependent
and only batch in `mode="swarm"`/`"parallel"`, is listed in
[Batch and parallel evaluation](parallel-evaluation.md).

The quality annotations are the maintainer's own notes; the plagiarism list
follows the warning in the upstream MEALPY project. Every flagged algorithm is
shipped by clypto — follow the module links to its category page.

---

## Swarm-based

### Suspected plagiarism

| Name | Module | Class | Year | Params |
| --- | --- | --- | --- | --- |
| Coati Optimization Algorithm | [`CoatiOA`](categories/swarm_based.md) | `OriginalCoatiOA` | 2023 | 2 |
| Fennec For Optimization | [`FFO`](categories/swarm_based.md) | `OriginalFFO` | 2022 | 2 |
| Northern Goshawk Optimization | [`NGO`](categories/swarm_based.md) | `OriginalNGO` | 2021 | 2 |
| Osprey Optimization Algorithm | [`OOA`](categories/swarm_based.md) | `OriginalOOA` | 2023 | 2 |
| Pelican Optimization Algorithm | [`POA`](categories/swarm_based.md) | `OriginalPOA` | 2023 | 2 |
| Serval Optimization Algorithm | [`ServalOA`](categories/swarm_based.md) | `OriginalServalOA` | 2022 | 2 |
| Siberian Tiger Optimization | [`STO`](categories/swarm_based.md) | `OriginalSTO` | 2022 | 2 |
| Tasmanian Devil Optimization | [`TDO`](categories/swarm_based.md) | `OriginalTDO` | 2022 | 2 |
| Walrus Optimization Algorithm | [`WaOA`](categories/swarm_based.md) | `OriginalWaOA` | 2022 | 2 |
| Zebra Optimization Algorithm | [`ZOA`](categories/swarm_based.md) | `OriginalZOA` | 2022 | 2 |

These algorithms are typically published under different names yet share the same
core equations and update rules, with only superficial changes in metaphor.
Several are flagged on [PubPeer](https://pubpeer.com/publications/1F5DCE5BC42BF2D77A1B0C281A5295).

### Quality annotations

- `OriginalMFO` — **weak**. Prefer `BaseMFO` when using Moth-Flame Optimization.
- `OriginalFOA`, `BaseFOA` — **weak**; `WhaleFOA` performs better.
- `OriginalCHIO` — **too weak**; `BaseCHIO` uses a mixed strategy.
- `HGSO`, `EHO`, `PFA`, `SSO` — sensitive to population ordering.
- `DS_GWO` — has no evolve step (not even in the pre-Cython sources); it
  raises `NotImplementedError`.

### Fixed in 2026

- `SSpiderA` — the candidate was appended only in sequential mode, so the
  batch modes failed with an `IndexError`. The sequential results are unchanged.
- `CSO` — `Validator.check_bool` list/tuple type mismatch (Cython enforces
  concrete types at compiled call boundaries).
- `SMO` — `local_leaders` built from the wrong return shape at four call sites.
- `GA` — infinite loop in roulette-wheel parent selection under floating-point
  underflow; replaced with a bounded uniform fallback (affects `MultiGA`,
  `SingleGA`, `EliteMultiGA`, `EliteSingleGA`).
- `BSA`, `CSO`, `DO` — mixed selection methods, reviewed against the references.

---

## Human-based

### Suspected plagiarism

| Name | Module | Class | Year | Params |
| --- | --- | --- | --- | --- |
| Teamwork Optimization Algorithm | [`TOA`](categories/human_based.md) | `OriginalTOA` | 2021 | 2 |

### Quality annotations

- `OriginalCHIO` — see the swarm note: **too weak** on its own.
- `LCO`, `GSKA` — update from neighbouring agents, so results are order
  sensitive.

### Fixed in 2026

- `DOA` — `idx`/`jdx` variable mixup caused out-of-bounds bound-array indexing.

---

## Bio-based

### Fixed in 2026

- `SBO` — `roulette_wheel_selection__` list-vs-`ndarray` mismatch; the
  annotation now accepts both.
- `BCO` — `n_chemotaxis` bound mismatch (the tuple bound is exclusive and the
  default sat exactly on the excluded boundary).
- `BFO` — odd-`pop_size` population-halving `IndexError` (integer division
  silently dropped a member).
- `BCO` — removed an unfinished "reproduction and elimination" block that
  referenced attributes and hyper-parameters never defined on the class.

---

## Physics-based

### Quality annotations

- `OriginalEFO` — changes only a single solution per generation and is
  **weak**; `BaseEFO` uses a swarm-inspired strategy.
- `OriginalTWO` — odd `pop_size` handling was fragile (see fixed bugs).

### Fixed in 2026

- `SOO` — the second phase appended to the first phase's candidate list, so
  `mode="swarm"`/`"parallel"` failed (`greedy` of two populations of different
  length). The sequential results are unchanged.
- `TWO` (`OppoTWO`) — odd-`pop_size` population-halving `IndexError`.
- `MSO` — `self.nfe_counter` typo (the real attribute is `self.nf_counter`).
- `MSO` — `nf_counter` now starts at 0 on every `solve()` (it started at 1 and
  accumulated across runs), so results changed. Its equations still use the
  evaluation count where the iteration count (`epoch`) looks intended.

---

## Evolutionary-based

### Quality annotations

- `DE` (`BaseDE`, `JADE`, `SADE`, `SHADE`, `L_SHADE`, `SAP_DE`), `GA`, `CRO`,
  `FPA` — selection picks random or roulette-wheel agents, so runs are more
  sensitive to population ordering.
- `OriginalGA` — has no evolve step (not even in the pre-Cython sources); it
  raises `NotImplementedError`. Use `BaseGA` (same parameters).

### Fixed in 2026

- `GA`, `MA` — `k_way` float/int coercion in
  `get_index_kway_tournament_selection` broke `numpy.random.Generator.choice()`
  under NumPy 2.5; fixed by truncating into a separate `int`-typed variable
  (affects `BaseGA`, `MultiGA`, `SingleGA`, `OriginalMA`).
- `MA` — off-by-one `IndexError` from eagerly-evaluated ternary indexing in
  offspring pairing.
- `GA` — off-by-one `IndexError` from population-size mismatch after crossover
  in survivor selection.

---

## Math-based

### Quality annotations

- `OriginalHC` — neighbourhood-size based; `BasicHC` reworks it as a
  swarm-based method.
- `SCA`, `AOA`, `CEM` — stable under reordering.

---

## System-based

### Quality annotations

- `AEO` (`BaseAEO`, `ImprovedAEO`, `ModifiedAEO`, `AugmentedAEO`,
  `EnhancedAEO`) — picks random agents but converges strongly at the end.
- `GCO`, `WCA` — selection depends on a best-agent list.

---

## Music-based

No known issues.

---

## Game-based

No known issues.

---

## SOTA-based

### Quality annotations

- **`OriginalIMODE` compares as if minimizing.** Its selection step (and the
  archive and operator-probability updates that follow it) uses a fixed
  `is_better(..., "min")` rather than the problem's sense, so on a
  **maximization** problem it keeps the *worse* trial vector. The behaviour is
  inherited, not clypto's: `mealpy-lts` calls `compare_target(agent.target,
  self.pop[idx].target)` and lets `minmax` fall back to its `"min"` default
  (`IMODE.py:277`, `optimizer.py:444`). See **Min-only comparisons** under
  Core framework. Use `OriginalLSHADEcnEpSin`, which passes the sense correctly.

---

## Core framework

These affect every optimizer rather than a single taxonomy group.

### Open issues

- **Min-only comparisons.** Six sites compare agents with a fixed `"min"` sense
  instead of `self.problem.sense`, so they select the worse agent on a
  maximization problem: `OriginalIMODE` (selection, archive and operator
  probabilities), `OriginalNGO` (Eq. 4 neighbour test), `OriginalESO` (survivor
  selection), `C_PSO` (chaotic local-search acceptance) and `OriginalBCO`
  (both neighbour and global-best exchange). Every one of them transcribes the
  same call in `mealpy-lts`, where `compare_target(...)` is invoked *without*
  its `minmax` argument and therefore defaults to `"min"`
  (`optimizer.py:444`; `IMODE.py:277`, `NGO.py:74`, `ESO.py:159`, `PSO.py:600`,
  `BCO.py:150,154`). The behaviour is kept for fidelity — "fixing" it would
  move clypto away from the pinned reference, as with the dead personal-best
  comparisons in `ABFO`/`ModifiedSLO`. Avoid these five on maximization
  problems.
- **`g_worst` is the agent that was worst at initialization.** It is set once in
  `after_initialization` and never refreshed, which is deliberately what
  `mealpy-lts` does: its loop assigns only `self.g_best`
  (`optimizer.py:266`), leaving `self.g_worst` on the initialization object. The
  classes that read it (`OriginalBBOA`, `OriginalEVO`) are bit-identical for
  that reason, and `OriginalDO`, `OriginalHGS`, `OriginalSSA`, the `JA` family
  and `OriginalGBO` set their own current worst inside `evolve()` and are
  unaffected either way.
- **Evaluation of a vector other than the agent's solution.** A few MEALPY
  algorithms give an agent the fitness of a vector it does not hold:
  `OriginalWOA` and `OriginalSBO` evaluate the new position into the *current*
  agent, `OriginalArchOA`, `OppoTWO`, `LevyTWO` and `EnhancedTWO` evaluate the
  position before its last correction, and the FOA variants evaluate the
  position rather than the smell vector the fly stores. The behaviour is kept
  and each site is marked in the source
  (`agent.update_solution(candidate, agent.solution)`). It was earlier
  documented here as leaving *results* unchanged versus MEALPY — true only for
  the variants that give themselves per-agent elitism (`OriginalFOA` and
  `WhaleFOA` call `cy.get_better_agent` right after and stay bit-identical). For
  `OriginalWOA`, `OriginalSBO` and `OriginalArchOA` the reported result differed
  until the `g_best` ratchet below landed; the quirk itself is byte-for-byte
  identical on both sides.

### Fixed in 2026

- **`g_best` was the best of the current population, not the best ever found.**
  `LegacyOptimizer.after_evolve()` overwrote `self.g_best` with
  `population.sort()[0]` every epoch, with no comparison against the previously
  stored value, and `DecoratedOptimizer.solve()` (`@cy.optimizer`) did the same
  with `self.population.best`. An algorithm whose `evolve()` replaces the
  population without per-agent elitism — which is how most "Original",
  paper-faithful variants are published — therefore lost results it had already
  found and *reported* the worse one: `OriginalESO` returned `0.0404` for a run
  that had reached `5e-34`. Both engines now ratchet: `g_best` is replaced only
  when the candidate is better, and is stored as an independent snapshot (a
  clone whose solution vector is copied, so a later in-place mutation of the
  population cannot corrupt it retroactively — `OriginalHBO` and `OriginalBCO`
  mutate solution vectors in place). This also restores `mealpy-lts`'s own
  semantics, where `g_best` is the `get_better_agent(current, historical)`
  clone taken *after* `evolve()`: the pre-fix aliasing meant the PSO family read
  a global best that other agents in the *same* epoch had already improved,
  which `mealpy-lts` does not do. The change is catalog-wide, not limited to the
  non-elitist classes (the reported-value bug was, but the trajectory effect was
  not): 98 of 243 classes changed their result across the two golden captures, and paired
  against `mealpy-lts` at 20 seeds the same set went from 17 `distribution_mismatch` flags to
  none (re-measure with `benchmarks/bench_correctness.py`). There is no automated guard: the
  golden captures and the per-epoch ratchet test were removed, so assert monotonicity by hand
  with `debug=True, after_iteration=...` over `epoch=1..N` at a fixed seed.
- **`max_early_stop` was ignored by `@cy.optimizer`.** The decorated engine
  passed a literal `0` as the no-improvement counter to `Termination.should_terminate`,
  so early stopping could never fire there; it now counts epochs in which
  `g_best` did not improve, as the legacy engine does.
- **Memory corruption from `wraparound: False`.** The codebase uses negative
  list indexing (`pop[-1]`) across 55+ files; disabling wraparound corrupted
  memory once compiled. Set back to `True`.
- **`Termination` and `debug=True` tracking.** The per-epoch early-stopping
  check no longer reads an uninitialized `__history` buffer, the dict form no
  longer reads nonexistent `Problem.log_to` / `log_file` attributes, and
  `debug=True` now records per-epoch metrics (and, with
  `track_population=True`, full population snapshots) into the Zarr-backed
  `model.tracker`.
