#!/usr/bin/env python
"""Behavior-preservation check against the pre-migration golden baseline.

The baseline was captured from the pure-Python implementation before the
collection was migrated to native Cython. A handful of optimizers are excluded
because they call scipy's global RNG (``cauchy.rvs``) and are non-deterministic
even with a fixed seed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import clypto as cy
from tests._engines import ENGINES, skip_if_not_bit_identical

N_DIMS = 5
BASELINE_PATH = Path(__file__).parent / "golden" / "baseline_2026a0.json"

NONDETERMINISTIC = {
    "AAO",
    "JADE",
    "L_SHADE",
    "OriginalLSHADEcnEpSin",
    "OriginalPSS",
    "OriginalSHADE",
}

pytestmark = pytest.mark.skipif(
    not BASELINE_PATH.exists(), reason="golden baseline not available"
)


def _baseline():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=lambda s: float(np.sum(s**2)),
        bounds=cy.FloatVar(lb=[-1.0] * N_DIMS, ub=[1.0] * N_DIMS),
        minmax="min",
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_no_new_optimizers_missing(engine):
    baseline = _baseline()
    discovered = cy.get_all_optimizers(engine=engine)
    missing = sorted(set(baseline) - set(discovered))
    assert missing == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "name",
    sorted(n for n in _baseline() if n not in NONDETERMINISTIC),
)
def test_optimizer_matches_baseline(engine, name, problem):
    expected = _baseline()[name]
    if "error" in expected:
        pytest.skip("baseline recorded an error for this optimizer")
    skip_if_not_bit_identical(engine, name)

    cls = cy.get_all_optimizers(engine=engine)[name]
    g_best = cls(epoch=50, pop_size=25).solve(problem, seed=1)

    fitness = float(g_best.target.fitness)
    digest = hashlib.sha1(
        np.asarray(g_best.solution, dtype=np.float64).tobytes()
    ).hexdigest()

    assert fitness == pytest.approx(expected["fitness"], abs=1e-9)
    assert digest == expected["solution_sha1"]
