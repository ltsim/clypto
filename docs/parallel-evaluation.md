# Batch and parallel evaluation

Every algorithm of the collection runs on one engine (`cy.Optimizer`). The `mode` argument decides how
a batch of candidate agents is evaluated:

| `mode` | Evaluation |
| --- | --- |
| `"sequential"` (default) | agent by agent, in the order of the classic algorithm |
| `"swarm"` | the whole batch at once, in one thread |
| `"parallel"` | the whole batch at once, on OpenMP threads when the problem has a nogil `evaluator` (single objective) |

Whenever an algorithm evaluates a batch, a problem built with `vectorized=True` receives it as one `(n, n_dims)` matrix: one
objective call per batch instead of one per agent.

## The batch pipeline

An algorithm that can evaluate a whole batch writes one pipeline for all modes:

```cython
cdef cy.Population n_population = cy.empty_snapshot(self.population)
for idx, agent in enumerate(self.population.toarray()):
    ...                                                   # the transition rule of the algorithm
    n_population.append(self.population.create_agent(cy.correct_solution(self.problem, x)))
self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)
```

`Population.evaluate` is the batch step. In sequential mode it only evaluates the agents that are not
evaluated yet, so an algorithm may still evaluate some agents one by one. `greedy(..., self.mode)` keeps the
tie rule of the one-by-one selection in sequential mode (`get_better_agent`: when maximizing, a tie keeps the
candidate), so sequential results are the ones of the classic loop, bit for bit.

## Which algorithms use it

An algorithm can use the pipeline only when every candidate of a phase depends on the population as it was
at the start of the phase. When agent *i* reads agents already replaced in the same epoch (a random other
agent, a teacher, the global best moved in place...) evaluating the batch at once is a different algorithm.
The split below was measured, not assumed: every class ran in sequential and in batch mode with the same
seed, and a class moved to the pipeline only when both gave the same bits (`tests/test_golden_*.py` hold
that line).

**Batch pipeline in every mode (73).** Parallel-capable whatever the mode:
`AAO`, `BaseGA`, `CMA_ES`, `ChaoticGWO`, `DevALO`, `DevBA`, `DevCHIO`, `DevGCO`, `DevHS`, `DevQSA`, `DevSARO`, `DevSCA`, `DevSOA`, `ER_GWO`, `EliteMultiGA`, `EliteSingleGA`, `ExGWO`, `FuzzyGWO`, `GWO_WOA`, `HI_WOA`, `IGWO`, `ImprovedQSA`, `IncrementalGWO`, `JADE`, `L_SHADE`, `LevyEP`, `LevyES`, `LevyQSA`, `Matlab101GTO`, `MultiGA`, `OGWO`, `OppoQSA`, `OriginalACOR`, `OriginalALO`, `OriginalAOA`, `OriginalBeesA`, `OriginalCDO`, `OriginalCEM`, `OriginalCHIO`, `OriginalCOA`, `OriginalCSA`, `OriginalEHO`, `OriginalEO`, `OriginalEP`, `OriginalES`, `OriginalFA`, `OriginalGCO`, `OriginalGTO`, `OriginalGWO`, `OriginalHBA`, `OriginalHCO`, `OriginalHGSO`, `OriginalHS`, `OriginalIWO`, `OriginalMShOA`, `OriginalPFA`, `OriginalQSA`, `OriginalRIME`, `OriginalSARO`, `OriginalSCA`, `OriginalSFOA`, `OriginalSHADE`, `OriginalSHIO`, `OriginalSeaHO`, `OriginalTS`, `OriginalWCA`, `OriginalWDO`, `ProbBeesA`, `QleSCA`, `SADE`, `Simple_CMA_ES`, `SingleGA`, `SwarmSA`.

**Order-dependent (124).** They keep the classic one-by-one step when `mode="sequential"`
(`if self.mode == "sequential": ...`) and evaluate their batch at once only in `"swarm"`/`"parallel"`,
where they are a (documented, MEALPY) batch variant of the algorithm:
`AdaptiveBA`, `AdaptiveEO`, `AugmentedAEO`, `CL_PSO`, `CleverBookBeesA`, `DevBBO`, `DevEFO`, `DevFBIO`, `DevFOA`, `DevFOX`, `DevGSKA`, `DevJA`, `DevLCO`, `DevMVO`, `DevSBO`, `DevSMA`, `DevSPBO`, `DevSSA`, `DevTLO`, `DevTPO`, `DevVCS`, `DevWOA`, `EnhancedAEO`, `EnhancedTWO`, `IARO`, `ImprovedAEO`, `ImprovedBSO`, `ImprovedLCO`, `ImprovedNMRA`, `ImprovedSFO`, `ImprovedSLO`, `ImprovedTLO`, `LARO`, `LevyJA`, `LevyTWO`, `MGTO`, `Matlab102GTO`, `ModifiedAEO`, `ModifiedEO`, `ModifiedSLO`, `OCRO`, `OppoTWO`, `OriginalAEO`, `OriginalAFT`, `OriginalAGTO`, `OriginalAO`, `OriginalARO`, `OriginalASO`, `OriginalAVOA`, `OriginalArchOA`, `OriginalBA`, `OriginalBBO`, `OriginalBBOA`, `OriginalBCO`, `OriginalBES`, `OriginalBMO`, `OriginalBSA`, `OriginalBSO`, `OriginalBWO`, `OriginalCDDO`, `OriginalCRO`, `OriginalCSO`, `OriginalCircleSA`, `OriginalDE`, `OriginalDO`, `OriginalDOA`, `OriginalEFO`, `OriginalEOA`, `OriginalEVO`, `OriginalFBIO`, `OriginalFLA`, `OriginalFOA`, `OriginalFOX`, `OriginalFPA`, `OriginalGBO`, `OriginalGJO`, `OriginalGOA`, `OriginalGSKA`, `OriginalHC`, `OriginalHGS`, `OriginalHHO`, `OriginalICA`, `OriginalINFO`, `OriginalJA`, `OriginalLCO`, `OriginalMA`, `OriginalMFO`, `OriginalMGO`, `OriginalMPA`, `OriginalMRFO`, `OriginalMSA`, `OriginalMVO`, `OriginalNMRA`, `OriginalNRO`, `OriginalPSS`, `OriginalSBO`, `OriginalSCSO`, `OriginalSFO`, `OriginalSHO`, `OriginalSLO`, `OriginalSMA`, `OriginalSOA`, `OriginalSOO`, `OriginalSPBO`, `OriginalSRSR`, `OriginalSSA`, `OriginalSSDO`, `OriginalSSO`, `OriginalSSpiderA`, `OriginalSSpiderO`, `OriginalSquirrelSA`, `OriginalTLO`, `OriginalTSA`, `OriginalTSO`, `OriginalTWO`, `OriginalVCS`, `OriginalWHO`, `OriginalWOA`, `OriginalZOA`, `RW_GWO`, `SAP_DE`, `SwarmHC`, `WMQIMRFO`, `WhaleFOA`.

**One agent at a time (46).** No batch step at all; the mode does not change them. The PSO
family is one of them: `g_best` is one of the particles and moves in place, so the particles after it
follow the new best within the same epoch:
`ABFO`, `AIW_PSO`, `CG_GWO`, `C_PSO`, `DS_GWO`, `DevBRO`, `DevDMOA`, `DevEPC`, `DevSMO`, `GaussianSA`, `HPSO_TVAC`, `IOBL_GWO`, `LDW_PSO`, `OriginalABC`, `OriginalBFO`, `OriginalBRO`, `OriginalCA`, `OriginalCGO`, `OriginalCoatiOA`, `OriginalDMOA`, `OriginalEAO`, `OriginalESO`, `OriginalESOA`, `OriginalFDO`, `OriginalFFA`, `OriginalFFO`, `OriginalGA`, `OriginalHBO`, `OriginalIMODE`, `OriginalLSHADEcnEpSin`, `OriginalMSO`, `OriginalNGO`, `OriginalOOA`, `OriginalPOA`, `OriginalPSO`, `OriginalRUN`, `OriginalSA`, `OriginalSOS`, `OriginalSTO`, `OriginalServalOA`, `OriginalTDO`, `OriginalTHRO`, `OriginalTOA`, `OriginalWaOA`, `OriginalWarSO`, `P_PSO`.

## History

Until 2026-09 clypto also shipped a second, vectorized copy of the collection (`engine="vectorize"`,
`VectorizeOptimizer` on a `NativePopulation` buffer). It was faster (median x5.9 on sphere) but only
statistically equivalent to the classic algorithms: it drew its random numbers in blocks and updated every
phase synchronously. It was removed so that one collection holds the reference behaviour; it can be recovered
from commit `0d839435`.
