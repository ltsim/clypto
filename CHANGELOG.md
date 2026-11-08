# Changelog

All notable changes to Clypto are documented in this file. The project follows [Semantic / Calendar Versioning](https://semver.org/).

## Clypto

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
