#!/usr/bin/env python
"""Vectorized collection versus the frozen legacy collection.

``test_same_seed_same_result`` (default CI) checks that a run is reproducible for every optimizer.
The statistical comparison (``benchmarks/compare_engines.py``: 30 seeds, Mann-Whitney U with Holm correction,
median guard) is slow; ``-m slow`` runs a small subset here, the full sweep is the script.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import clypto as cy
from tests._engines import NO_EVOLVE, statistical_only

VECTORIZE = cy.get_all_optimizers(engine="vectorize")
# Still on scipy's global RNG (nondeterministic like legacy); remove a name from this set once its vectorize
# version draws from ``self.generator``.
NONDETERMINISTIC_TODO = {"OriginalLSHADEcnEpSin"}


def _sphere(x):
    return float(np.sum(x**2))


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.xfail(reason="uses scipy's global RNG", strict=False)) if n in NONDETERMINISTIC_TODO else n
     for n in sorted(VECTORIZE.keys() - NO_EVOLVE)],
)
def test_same_seed_same_result(name):
    problem = cy.Problem(obj_func=_sphere, bounds=cy.NumberBounds(float, low=[-3.0] * 6, up=[3.0] * 6), sense="min")
    runs = [VECTORIZE[name](epoch=50, pop_size=25).solve(problem, seed=7) for _ in range(2)]
    assert runs[0].target.fitness == runs[1].target.fitness
    np.testing.assert_array_equal(runs[0].solution, runs[1].solution)


def _load_script():
    path = Path(__file__).parents[1] / "benchmarks" / "compare_engines.py"
    spec = importlib.util.spec_from_file_location("compare_engines", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_engines"] = module
    spec.loader.exec_module(module)
    return module


SUBSET = ["OriginalPSO", "OriginalGWO", "OriginalMFO", "OriginalDE", "OriginalHHO"]


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n in SUBSET if n in statistical_only() or n in VECTORIZE])
def test_statistically_equivalent_to_legacy(name):
    compare = _load_script()
    rc = compare.main(["--only", f"^{name}$", "--seeds", "12", "--dims", "10", "--epoch", "60", "--pop", "25"])
    assert rc == 0
