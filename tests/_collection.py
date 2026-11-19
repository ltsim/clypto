"""Shared facts about the collection for the collection-wide tests."""

# No evolve step, not even in the pre-Cython sources (docs/issues.md): they raise NotImplementedError
# instead of silently returning the best random initial agent.
NO_EVOLVE: set[str] = set()

# Behavior changes made on purpose after the baseline was captured.
CHANGED = {
    "OriginalGA": "implemented evolve step by inheriting from BaseGA",
    "DS_GWO": "implemented evolve step (Jiang et al. 2022) with group-stage competition",
    "OriginalMSO": "reads nf_counter, which now starts at 0 on every solve() instead of 1 and accumulating",
    # scipy's `rvs` defaulted to the global numpy RNG, so `solve(seed=...)` did not reproduce;
    # these four now draw from the optimizer's own (seeded) stream.
    "JADE": "draws cauchy from the optimizer stream instead of the global numpy RNG",
    "OriginalSHADE": "draws cauchy from the optimizer stream instead of the global numpy RNG",
    "L_SHADE": "draws cauchy from the optimizer stream instead of the global numpy RNG",
    "OriginalLSHADEcnEpSin": "draws cauchy/norm from the optimizer stream instead of the global numpy RNG",
}
