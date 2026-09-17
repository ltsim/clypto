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

### Fixed in 2026

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

- `TWO` (`OppoTWO`) — odd-`pop_size` population-halving `IndexError`.
- `MSO` — `self.nfe_counter` typo (the real attribute is `self.nf_counter`).

---

## Evolutionary-based

### Quality annotations

- `DE` (`BaseDE`, `JADE`, `SADE`, `SHADE`, `L_SHADE`, `SAP_DE`), `GA`, `CRO`,
  `FPA` — selection picks random or roulette-wheel agents, so runs are more
  sensitive to population ordering.

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

No known issues.

---

## Core framework

These affect every optimizer rather than a single taxonomy group.

### Open issues

No known issues.

### Fixed in 2026

- **Memory corruption from `wraparound: False`.** The codebase uses negative
  list indexing (`pop[-1]`) across 55+ files; disabling wraparound corrupted
  memory once compiled. Set back to `True`.
- **`Termination` and `debug=True` tracking.** The per-epoch early-stopping
  check no longer reads an uninitialized `__history` buffer, the dict form no
  longer reads nonexistent `Problem.log_to` / `log_file` attributes, and
  `debug=True` now records per-epoch metrics (and, with
  `track_population=True`, full population snapshots) into the Zarr-backed
  `model.tracker`.
