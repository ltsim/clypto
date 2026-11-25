# Changelog

All notable changes to Clypto are documented in this file. The project follows [Semantic / Calendar Versioning](https://semver.org/).

## Clypto

### [Unreleased]

### [2026b2]

#### Fixed
- **The SHADE family runs with a population below 10**: `OriginalSHADE` sampled `p ~ U[2/NP, 0.2]`, an inverted interval for `NP < 10`, which raised `ValueError: high - low < 0` on the first epoch; the `2/NP` floor now falls back to the `0.2` cap when it would exceed it. `OriginalLSHADEcnEpSin` set `NP_min = pop_size_min` unconditionally, so the default `pop_size_min = 10` with a smaller population grew `NP` past `len(population)` and the mutation index ran off the end (`IndexError`); `NP_min` is now clamped to `NP_init`. Results for `pop_size >= 10` are bit-for-bit unchanged.
- **The `n_objs` probe no longer consumes a bounds draw**: `Problem.n_objs` evaluates one random solution on first access, and `Agent.evaluate` reads `obj_weights` right after each evaluation, so the probe fired inside initialisation and shifted the whole initial population by one draw (draws `[1, 3, 4, 5, 6]` instead of `[1, 2, 3, 4, 5]`). The probe now saves and restores every bounds block's `bit_generator` state, so the run's stream is left untouched.
- **`OriginalPSS` samples its Latin hypercube from the optimizer stream**: `scipy.stats.qmc.LatinHypercube` was built without a seed, so it drew from its own uncontrolled generator and `solve(seed=...)` did not reproduce `OriginalPSS` on the `LHS` branch (the `MC` branch already used `self.generator`). It now receives `rng=self.generator`.
- **Duck-typed agents no longer crash `Population`**: `evaluate`, `snapshot`, `fitness` and `solutions` cast the public `agents` list to `Agent` without checking, so a decorator-API agent sitting next to a native one raised `TypeError`, segfaulted, or silently returned the wrong fitness/solutions. Each site now checks or converts explicitly; covered by subprocess regression tests.
- **Bounds fall back when the span overflows**: a `NumberBounds` span that overflows to `inf` (very wide `low`/`up`, or `±sys.float_info.max`) produced `NaN` positions. Such spans now draw through `Generator.uniform`, which handles them.
- **`scipy.stats.*.rvs` draws come from the optimizer stream**: `JADE`, `OriginalSHADE`, `L_SHADE` and `LSHADEcnEpSin` drew from scipy's unseeded global generator, so `solve(seed=...)` did not reproduce them. They now draw from `self.generator`.

#### Changed
- **Ranking breaks fitness ties by population order**: `Population.argsort` and `argsort_agents` use a stable `argsort`, so inside a tie group the population order decides, matching the `sorted()` the mealpy port is compared against (and `kway_tournament`, which was already stable). Results without ties are unchanged.
- **Decorator compilation no longer uses `pyximport`**: `@cy.optimizer(..., compile=True)` builds the extension through `clypto.compiler` and caches it by source hash, deleting the `pyximport` engine (`precompile/_builder.pyx`, `precompile/_runtime.pyx`).
- **Seeded runs (BREAKING-SEEDED)**: seeded runs draw different populations than 2026b2; bit compatibility with earlier releases is lost. The same seed on the same version still replays exactly, but fitness and solutions recorded on 2026b2 or earlier change bit for bit and cannot be diffed against a new capture — re-run baselines instead. `problem.n_evals` now counts the `n_objs` probe too, so a fresh `Problem`'s counter reads one higher than in 2026b2, and a `max_fe` budget hit exactly by a population step stops one epoch earlier.
- **A seeded `Problem` seeds the whole run**: `Problem.coerce` used to overwrite `problem.seed` with the `solve()` argument unconditionally, so `solve(problem)` with no seed discarded `Problem(seed=...)` and produced a different result on every call. `None` now means "no seed given": the Problem's seed is kept and drives the engine's own generators, so `Problem(seed=1)` plus `solve(problem)` replays exactly like `solve(problem, seed=1)`. Pass an explicit seed, or build an unseeded `Problem`, for a fresh random run.

#### Performance
- **`Population` hot paths**: `fill()` writes `fitness` and `solutions` through C pointers instead of boxing a float per agent, and a pointer reset plus a single `clip` replaced per-agent work in the evaluation loop; `argsort`/`sort` and the tournament helper use a stable sort.
- **PSO `update_velocity`** walks the population through pointers instead of Python indexing.
- **`Bounds`** inlines the float `uniform` draw instead of building a per-agent helper object.
- **DE families** (`BaseDE`, `JADE`, `SADE`, `SHADE`, `L_SHADE`, `SAP_DE`) build their crossover pools as lists, dropping `set` differences per agent.
- **Loop-invariant aggregates hoisted** out of the per-agent loop across the collection (evolutionary, GCO, and the shared swarm/human/physics modules); the redundant `cls=cy.Population` argument was dropped from the catalog.
- **Decorator engine**: classes are built by `clypto.compiler` with a source-hash cache, and the redundant compiler/CLI test suite was removed.

### [2026b1]

#### Added
- **`AgentFactory` & Agent Lifecycle**: Delegated agent creation, generation, and initialization to `AgentFactory` (`Population.fill()`, `add()`, `remove_at()`). Added `cy.population_with_factory(size, factory)`.
- **Copy Protocol for Agent Subclasses**: Subclassable `Agent` (unsealed `@cython.final`) with virtual copy hooks `_new_instance` and `_copy_state_to` for algorithms requiring extra per-agent state. Added `tools/scaffold_agent.py` helper.
- **In-Memory Tracking Buffers**: Structured `MetricBuffer` (pre-allocated float64 metrics matrix) and `SnapshotBuffer` (population frames buffer).
- **Custom Per-Epoch Metrics (`@cy.metrics`)**: Class decorator that validates metric methods at definition time (float return annotation, one or two `positions`/`fits` arguments) into a `MetricSchema` of pre-resolved column indices; `DefaultMetrics` keeps the canonical `mealpy-lts` continuous metric set and can be subclassed to extend it. Passing a metrics class or instance to `solve(problem, metrics=...)` enables tracking on its own and allocates `(epoch, n_metrics)` with zero dictionary lookups in the per-epoch loop; read the result through `tracker["<metric_name>"]`, `tracker.metric_names` or `"name" in tracker`. `cy.metrics`, `cy.DefaultMetrics` and `cy.Tracker` join the flat public API.
- **Problem Decorator & JIT Compilation (`@cy.problem`)**: Added function and class decorator `@cy.problem` with opt-in Cython JIT compilation (`compile=True`). Supports automatic decoding (`decode=True`) for combinatorial, categorical, and mixed engineering problems, flexible target binding, standardized `evaluate` methods, and vectorized evaluations.
- **Execution Lifecycle Callbacks**: Direct `before_iteration` and `after_iteration` hooks in `solve()`.
- **Local Git Hooks & Release Workflow**: Husky pre-commit (Ruff/mypy) and pre-push (pytest) hooks; isolated release branches (`release/v<year><edition><version>`).

#### Changed
- **In-Memory `Tracker`**: Replaced Zarr storage with in-memory NumPy arrays (`tracker[key]`, typed fields such as `tracker.n_epochs`, `tracker.optimizer_name`).
- **Unified `Agent` & `Population`**: Consolidated catalog agent variants into a single `Agent` class and a final `Population` class with a standardized batch evaluation pipeline (`best`, `top(k)`, `bottom(k)`).
- **OpenMP Thread Management**: Thread team configured once per run via `solve(..., threads=0)` and evaluated through `prange`.
- **Module Relocation**: Moved tracker into `clypto.optimizer.history` (`tracker.pyx`).
- **Citation Metadata**: Updated `CITATION.cff` for Clypto.

#### Fixed
- **Algorithm Repairs & Implementations**:
  - **`OriginalGA`**: Repaired `OriginalGA` by inheriting from `BaseGA`, implementing the complete genetic algorithm lifecycle and evolution loop with support for configurable multi-point and single-point mutation strategies; removed `NotImplementedError`.
  - **`DS_GWO`**: Implemented the missing `evolve()` step following Jiang et al. (2022) with group-stage competition in the exploration phase and 3-leader convergence in the exploitation phase; removed `NotImplementedError`.
- **Performance & Algorithm Optimizations**:
  - **`OriginalASO`**: Vectorized Lennard-Jones (LJ) potential and acceleration calculations, inlined potential evaluations in C, and eliminated per-atom quadratic memory allocations and loop overhead in the atom force calculation loop.
  - **`AAO`**: Optimized `evolve()` by hoisting epoch-level invariants (`sigmoid_factor`, exploration probability, population mean position `x_mean`) outside the agent loop; replaced unseeded global `np.random.rand()` with `self.generator.random()`; eliminated unused random index choice overwrite.
  - **`EliteMultiGA` / `MultiGA`**: Optimized swap mutation by directly swapping two elements via `generator.integers` (eliminating `set` difference and list conversions), and vectorized flip mutation via `generator.random`.
  - **`DevEPC`**: Optimized penguin attractiveness and spiral movement calculation; precomputed trigonometric constants (`cos_theta`, `sin_theta`); inlined C-level math and vector operations; eliminated dynamic rotation matrix allocations and `np.allclose` comparisons in inner loops.
  - **`ImprovedBSO`**: Clamped `m_clusters` dynamically to `max_clusters = max(2, int(pop_size / 5))` when `pop_size < 25`, avoiding validator range exceptions on small population budgets.
- **Budget-Dependent Parameter Defaults Adaptation**:
  - Dynamically adapted hyperparameter defaults that scale with `epoch` or `pop_size` to prevent validator out-of-bounds exceptions on small budgets:
    - **`CHIO`** (`OriginalCHIO`, `DevCHIO`): `max_age` default adapted to upper bound `[1, max(1, 1 + int(epoch / 5))]`.
    - **`HI_WOA`**: `feedback_max` default adapted to upper bound `[2, max(2, 2 + int(epoch / 2))]`.
    - **`OCRO`**: `restart_count` default adapted to upper bound `[2, max(2, int(epoch / 2))]`.
    - **`OriginalEHO`**: `n_clans` default adapted to upper bound `[2, max(2, int(pop_size / 5))]`.
    - **`OriginalGSKA`**: `kg` default adapted to upper bound `[1, max(1, 1 + int(epoch / 2))]`.

#### Removed / Breaking Changes
- Removed `zarr>=3` dependency, `tracker.meta`, `history_path`, and on-disk tracking stores.
- Removed `tracker.before`, `tracker.after`, `on_before`, and `on_after` (use `solve(before_iteration=..., after_iteration=...)`).
- Custom `Population` subclasses are no longer accepted in `cy.population(size)`.

---

### [2026b]

#### Added
- **Decorator-Based Optimizer API**: Support for writing custom optimizers via `@cy.optimizer`, `@cy.agent`, and `@cy.legacy`, declaring hyper-parameters and state via `cy.Argument` and `cy.Attribute`.
- **`cy.Population`**: Clean population container exposing solutions matrix, read-only fitness, and manipulation methods.
- **Opt-in Cython Compilation**: Compile decorated classes to native extensions at import time via `compile=True` / `precompile=True`.
- **Documentation**: New custom optimizer tutorials and migration guide.

#### Changed
- Renamed base `Optimizer` to `LegacyOptimizer` (kept as a backward-compatible alias).

---

### [2026]

#### Added
- **Complete Cython Native Core**: Reimplemented core classes, algorithms, and populations in native Cython with C-level evaluation pipelines and OpenMP multithreading (`prange`).
- **Monotonic Global Best (`g_best` Ratchet)**: Guaranteed that `g_best` always tracks the best solution found throughout the entire run using independent snapshots.
- **Unified `cy` Namespace**: Consolidated the collection under `clypto.native.collection.<category>` and exposed core utilities via `cimport clypto.core as cy`.
- **C-Level Bounds & Repair Routines**: Added `Bounds` system (`NumberBounds`, `TransferBounds`, `StringBounds`, etc.) and native repair functions (`cy.correct_solution`, `cy.reset_solution`, `cy.opposite_solution`).
- **Fast Batch Pipeline**: Batched evaluation pipeline for all catalog algorithms with vectorization and OpenMP execution.
- **Modern Packaging with `uv`**: Switched build/test workflow to `uv` with cross-platform wheel generation (`cibuildwheel`).

#### Fixed
- Fixed memory corruption caused by `wraparound: False` on negative list indexing.
- Fixed strict type coercion issues with NumPy 2.x and Cython call boundaries.
- Fixed infinite loops in roulette-wheel selection caused by floating-point underflow.
- Resolved boundary condition bugs, indexing mismatches, and latent errors inherited from legacy implementations across 90+ algorithm variants.

#### Removed
- Removed legacy vectorization tree duplicate engines (`VectorizeOptimizer`, `NativePopulation`, `AgentListOptimizer`).
- Permanently stripped all GUI, plotting, external file logging, and unneeded benchmark bloat.
- Dropped runtime dependencies on `Cython`, `flake8`, and legacy packaging scripts.

---

## MEALPY (Upstream Archive Summary)

*Clypto originated as a high-performance Cython rewrite and fork of [MEALPY](https://github.com/thieu1995/mealpy) (v1.0.0 through v3.0.4.1 by Thieu Nguyen).*

### Historical Milestones

- **v3.0.x (2023 - 2024)**:
  - Added new algorithm families: state-of-the-art CEC competition variants (`sota_based`), `game-based`, chaotic and fuzzy modules.
  - Expanded swarm, physics, and human-based catalogs (SOO, EAO, SFOA, DOA, MSO, THRO, MShOA).
  - Switched license from GNU GPLv3 to MIT.
- **v2.0.x - v2.5.x (2021 - 2023)**:
  - Restructured architecture into clean object-oriented modules (`Problem`, `Optimizer`, `Agent`).
  - Added multi-objective optimization support, termination conditions, and variable search spaces (`*Var`).
  - Introduced parallel/multiprocessing modes and expanded the algorithm catalog beyond 150+ metaheuristics.
- **v1.0.x - v1.2.x (2020 - 2021)**:
  - Standardized algorithm interface and modularized legacy procedural implementations.
  - Introduced base classes for evolutionary, swarm-based, and physics-based metaheuristics.
- **v0.1.0 - v0.8.x (2019 - 2020)**:
  - Initial open-source releases implementing classic metaheuristics (GA, PSO, DE, GWO, BFO).

*For the complete granular historical changelog prior to the Clypto fork, refer to the [MEALPY repository](https://github.com/thieu1995/mealpy).*
