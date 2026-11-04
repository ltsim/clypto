"""Shared facts about the collection for the collection-wide tests."""

# No evolve step, not even in the pre-Cython sources (docs/issues.md): they raise NotImplementedError
# instead of silently returning the best random initial agent.
NO_EVOLVE = {"OriginalGA", "DS_GWO"}

# Behavior changes made on purpose after the baseline was captured.
CHANGED = {
    "OriginalGA": "never had an evolve step; now raises NotImplementedError",
    "DS_GWO": "never had an evolve step; now raises NotImplementedError",
    "OriginalMSO": "reads nf_counter, which now starts at 0 on every solve() instead of 1 and accumulating",
}
