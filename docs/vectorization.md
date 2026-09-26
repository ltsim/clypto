# Legacy and vectorized collections

clypto ships every algorithm twice, both Cythonized:

| Tree | Package | What it is |
| --- | --- | --- |
| `vectorize` (default) | `clypto.native.collection.vectorize` | whole-population NumPy/C code on the native engine (`NativePopulation`) |
| `legacy` | `clypto.native.collection.legacy` | the classic per-agent algorithms on `_LegacyOptimizer`, frozen |

`clypto.collection.<category>.<Module>` and `cy.get_all_optimizers()` return the vectorized classes. The classic ones are
reachable with `cy.get_all_optimizers(engine="legacy")`, `cy.get_optimizer_by_name("PSO", engine="legacy")` or
`from clypto.native.collection.legacy.swarm_based import PSO`. Class names are identical in both trees.

## What "vectorized" means here

* **Block random numbers.** The draws of one epoch come from `self.generator` as arrays (`(n, d)`, `(n, k, d)`), on the main
  thread. The same seed always gives the same result; no OpenMP kernel draws random numbers.
* **Synchronous phases.** Classic code updates agents one at a time and lets later agents read the rows already updated.
  The vectorized code computes all candidates of a phase from the same population, evaluates them in **one** batch
  (`evolve` calls `ops.step`, `ops.replace` or `ops.scatter`) and keeps the better rows.
* **One objective call per phase** with `Problem(vectorized=True)`.

Because of the synchronous update the results are *statistically* equivalent to legacy, not bit-identical. The classes
that are not bit-identical are listed in `tests/golden/vectorize_exactness.json`; `benchmarks/compare_engines.py`
checks them against legacy (30 seeds, sphere / rastrigin / ackley / rosenbrock, Mann-Whitney U with Holm correction, a
median guard, and the number of function evaluations must not exceed legacy's).

## Results (d=30, pop=50, epoch=100, sphere, best of 3, one core)

| | vectorize vs legacy | vectorize vs mealpy-lts | batch objective (`vectorized=True`) |
| --- | --- | --- | --- |
| median speedup (243 classes) | **x5.9** | x7.1 | a further x2.0 |

Before this work the whole collection ran at a median x1.4 of legacy. Distribution of the vectorize/legacy speedup:
<0.9x: 10, 0.9-1.5x: 48, 1.5-3x: 24, 3-10x: 104, >=10x: 57. Best: OriginalGSKA x137, OriginalSCSO x103, OriginalSMA x83, OriginalSCA x62, DevFOA x40, OriginalPSS x39.
Slower than legacy (9 classes, all sequential by nature or object-based): OCRO x0.32, GaussianSA x0.39, OriginalEVO x0.63, OriginalICA x0.64, OriginalCRO x0.72, OriginalHBO x0.77, OppoQSA x0.81, ImprovedSFO x0.82, QleSCA x0.90.

Reproduce with `python benchmarks/bench_engines.py --epoch 100 --repeats 3 --mealpy-python <venv>/bin/python`.
Statistical equivalence (`benchmarks/compare_engines.py`, 12 seeds, d=10): 195 of the 243 classes PASS in the first full sweep, 28 only trip
the median guard (`WARN`), 11 are significantly *better* than legacy (`INVESTIGATE`: they mostly fix legacy quirks), and 6 are
slightly worse (`FAIL`: DevEPC, MultiGA, OriginalServalOA, OriginalDMOA, LevyTWO, OriginalDO; the synchronous update loses the
chained improvements of the classic loop).

## Helpers (`clypto.optimizer.native.ops`)

| Helper | Use |
| --- | --- |
| `ops.step(opt, pos, stop=, start=)` | bound `pos`, evaluate the block once, keep the better rows |
| `ops.replace(opt, pos)` | every agent moves to its candidate |
| `ops.scatter(opt, cand, targets)` | rows of `cand` compete with rows `targets` of the population (best candidate per target wins) |
| `ops.others / two_others / k_others` | random other agents, never the agent itself |
| `ops.roulette / better / best_row / neighbors / exclude / pick_range` | selection and index helpers |

## What stays sequential

37 classes still run on `AgentListOptimizer`, the compatibility layer that keeps the classic list-of-agents code on the
native engine (roughly the speed of legacy). Their update rules cannot be batched without changing the algorithm:
single-trajectory searches and state machines whose next step depends on the outcome of the previous agent (QSA family,
HBO, BRO, SARO, CHIO), per-agent sub-populations and groups (CSO, ESOA, ICA, SRSR, SSpiderA/O, SFO family, BSO, BFO/ABFO,
SquirrelSA, FFA), adaptive or shrinking populations (IMODE, LSHADEcnEpSin, CMA-ES, MA) and `OriginalGA`, which has no
`evolve` in the classic code either.

`ABFO`, `CMA_ES`, `DevBRO`, `DevCHIO`, `DevQSA`, `DevSARO`, `DevSMO`, `ImprovedBSO`, `ImprovedQSA`, `ImprovedSFO`, `ImprovedSLO`, `ImprovedTLO`, `LevyQSA`, `ModifiedSLO`, `OppoQSA`, `OriginalBFO`, `OriginalBRO`, `OriginalBSA`, `OriginalBSO`, `OriginalCHIO`, `OriginalCSO`, `OriginalESOA`, `OriginalFFA`, `OriginalGA`, `OriginalHBO`, `OriginalICA`, `OriginalIMODE`, `OriginalLSHADEcnEpSin`, `OriginalMA`, `OriginalQSA`, `OriginalSARO`, `OriginalSFO`, `OriginalSRSR`, `OriginalSSpiderA`, `OriginalSSpiderO`, `OriginalSquirrelSA`, `WMQIMRFO`
