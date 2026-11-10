"""Shared facts about the collection for the collection-wide tests."""

# No evolve step, not even in the pre-Cython sources (docs/issues.md): they raise NotImplementedError
# instead of silently returning the best random initial agent.
NO_EVOLVE = set()

# Behavior changes made on purpose after the baseline was captured.
CHANGED = {
    "OriginalGA": "implemented evolve step by inheriting from BaseGA",
    "DS_GWO": "implemented evolve step (Jiang et al. 2022) with group-stage competition",
    "OriginalMSO": "reads nf_counter, which now starts at 0 on every solve() instead of 1 and accumulating",
}
