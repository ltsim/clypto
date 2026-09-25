#!/usr/bin/env python
"""Vectorized native engine: thread-count independence and vectorized objectives."""
import os
import subprocess
import sys

import numpy as np
import pytest

import clypto as cy
from clypto.collection.swarm_based import PSO

_RUN = """
import hashlib, numpy as np, clypto as cy
from clypto.collection.swarm_based import PSO
problem = cy.Problem(obj_func=lambda s: float(np.sum(s**2)),
                     bounds=cy.FloatVar(lb=[-5.0] * 60, ub=[5.0] * 60))
g = PSO.{name}(epoch=5, pop_size=500).solve(problem, seed=3)
print(g.target.fitness, hashlib.sha1(g.solution.tobytes()).hexdigest())
"""


@pytest.mark.parametrize("name", ["OriginalPSO", "LDW_PSO"])
def test_result_does_not_depend_on_openmp_threads(name):
    # 500 x 60 is above the threshold where the move kernel uses OpenMP threads.
    def run(threads):
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        out = subprocess.run([sys.executable, "-c", _RUN.format(name=name)],
                             env=env, capture_output=True, text=True, check=True)
        return out.stdout.strip()

    assert run(1) == run(4)


def _problems():
    bounds = cy.FloatVar(lb=[-3.0] * 8, ub=[3.0] * 8)
    scalar = cy.Problem(obj_func=lambda x: float(np.sum(x**2)), bounds=bounds)
    vector = cy.Problem(obj_func=lambda X: np.sum(X**2, axis=1), bounds=bounds, vectorized=True)
    return scalar, vector


@pytest.mark.parametrize("name", ["OriginalPSO", "AIW_PSO", "C_PSO", "CL_PSO"])
def test_vectorized_objective_matches_scalar(name):
    scalar, vector = _problems()
    a = getattr(PSO, name)(epoch=20, pop_size=20).solve(scalar, seed=5)
    b = getattr(PSO, name)(epoch=20, pop_size=20).solve(vector, seed=5)
    assert a.target.fitness == b.target.fitness
    np.testing.assert_array_equal(a.solution, b.solution)


def test_vectorized_objective_on_agent_list_engine():
    # Algorithms kept on the agent-list compatibility layer evaluate one row at a time.
    from clypto.collection.swarm_based import GTO

    scalar, vector = _problems()
    a = GTO.OriginalGTO(epoch=10, pop_size=10).solve(scalar, seed=2)
    b = GTO.OriginalGTO(epoch=10, pop_size=10).solve(vector, seed=2)
    assert a.target.fitness == b.target.fitness


def test_population_is_a_buffer():
    model = PSO.OriginalPSO(epoch=3, pop_size=6)
    model.solve(_problems()[0], seed=1)
    pop = model.pop
    assert pop.buf.shape == (6, pop.width) and pop.buf.flags.c_contiguous
    assert np.array_equal(pop.F, np.sum(pop.X**2, axis=1))
    assert pop[0].target.fitness == pop.F[0]
