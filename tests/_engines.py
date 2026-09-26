"""Shared helpers to run collection tests on both engines (``legacy`` and ``vectorize``)."""
import json
from pathlib import Path

import pytest

ENGINES = ("legacy", "vectorize")
EXACTNESS_PATH = Path(__file__).parent / "golden" / "vectorize_exactness.json"

# No evolve step, not even in the pre-Cython sources (docs/issues.md): they raise NotImplementedError
# instead of silently returning the best random initial agent.
NO_EVOLVE = {"OriginalGA", "DS_GWO"}

# Behavior changes made on purpose after the baseline was captured.
CHANGED = {
    "OriginalGA": "never had an evolve step; now raises NotImplementedError",
    "DS_GWO": "never had an evolve step; now raises NotImplementedError",
    "OriginalMSO": "reads nf_counter, which now starts at 0 on every solve() instead of 1 and accumulating",
}


def statistical_only():
    """Class names whose vectorize version is only statistically equivalent to legacy."""
    return set(json.loads(EXACTNESS_PATH.read_text())["statistical"])


def skip_if_not_bit_identical(engine, name):
    if engine == "vectorize" and name in statistical_only():
        pytest.skip("vectorize version is statistically equivalent only (see tests/test_native_statistical.py)")
