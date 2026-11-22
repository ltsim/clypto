#!/usr/bin/env python
"""Tests for user-defined algorithms in both Python and Cython (.pyx)."""

import os
import numpy as np
import pytest

import clypto as cy

N_DIMS = 3


def sphere(solution):
    return float(np.sum(solution**2))


@pytest.fixture
def problem():
    return cy.Problem(
        obj_func=sphere,
        bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS),
        sense="min",
    )


def test_get_include():
    include_dir = cy.get_include()
    assert os.path.isdir(include_dir)
    core_pxd = os.path.join(include_dir, "clypto", "core.pxd")
    assert os.path.isfile(core_pxd)


def test_python_subclass_optimizer(problem):
    class StepDecayOptimizer(cy.Optimizer):
        def __init__(self, epoch=25, pop_size=12, decay=0.8, **kwargs):
            super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
            self.decay = decay
            self.init_called = False

        def initialize(self):
            self.init_called = True

        def evolve(self, epoch):
            for agent in self.population:
                agent.solution = agent.solution * self.decay
                agent.evaluate(self.problem)

    optimizer = StepDecayOptimizer(epoch=15, pop_size=10, decay=0.5)

    assert optimizer.epoch == 15
    assert optimizer.pop_size == 10
    assert optimizer.parameters["epoch"] == 15
    assert optimizer.parameters["pop_size"] == 10
    assert optimizer.parameters["decay"] == 0.5
    assert optimizer.bounds is None

    best = optimizer.solve(problem, seed=42)

    assert optimizer.init_called is True
    assert optimizer.bounds is not None
    assert best.solution.shape == (N_DIMS,)
    assert np.isfinite(best.fitness)
    assert best.fitness < 1.0


def test_python_subclass_custom_agent_state(problem):
    class TaggedSwarmOptimizer(cy.Optimizer):
        def generate(self, agent, solution):
            agent.weight = 99.0
            return agent

        def evolve(self, epoch):
            for agent in self.population:
                agent.solution = agent.solution * 0.9

    optimizer = TaggedSwarmOptimizer(epoch=5, pop_size=8)
    best = optimizer.solve(problem, seed=42)

    assert all(agent.weight == 99.0 for agent in optimizer.population)
    assert np.isfinite(best.fitness)


def test_cython_pyx_user_algorithm(problem):
    pytest.importorskip("Cython")

    pyx_content = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class CythonUserAlgo(cy.Optimizer):
    cdef public double factor
    cdef public bint init_called

    def __init__(self, epoch=20, pop_size=10, factor=0.5, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")
        self.init_called = False

    def initialize(self):
        self.init_called = True

    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * self.factor
            agent.evaluate(self.problem)
"""

    cls = cy.compile(pyx_content, name="CythonUserAlgo")

    algo = cls(epoch=10, pop_size=8, factor=0.3)
    assert algo.epoch == 10
    assert algo.pop_size == 8
    assert algo.parameters["epoch"] == 10
    assert algo.parameters["pop_size"] == 8
    assert algo.parameters["factor"] == 0.3

    best = algo.solve(problem, seed=123)
    assert algo.init_called is True
    assert best.solution.shape == (N_DIMS,)
    assert np.isfinite(best.fitness)
    assert best.fitness < 1.0
