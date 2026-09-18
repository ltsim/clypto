#!/usr/bin/env python
"""Unit tests for ``@cy.legacy`` (classic MEALPY-style API via decorator)."""

import numpy as np
import pytest

import clypto as cy

N_DIMS = 3


def objective(solution):
    return np.sum(solution**2)


@cy.legacy(precompile=False)
class RandomSearch:
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True
        self.is_parallelizable = False

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(self.problem.generate_solution(encoded=True))
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.get_target(pos_new)
            self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.FloatVar(lb=[-5.0] * N_DIMS, ub=[5.0] * N_DIMS),
        minmax="min",
    )


def test_legacy_decorator_injects_the_base():
    assert issubclass(RandomSearch, cy.LegacyOptimizer)
    assert issubclass(RandomSearch, cy.Optimizer)


def test_legacy_decorator_keeps_super_working():
    optimizer = RandomSearch(epoch=25, pop_size=12)

    assert optimizer.get_parameters() == {"epoch": 25, "pop_size": 12}
    assert optimizer.epoch == 25
    assert optimizer.pop_size == 12
    assert optimizer.validator is not None


def test_legacy_decorator_solves(problem):
    optimizer = RandomSearch(epoch=30, pop_size=15)
    g_best = optimizer.solve(problem, seed=1)

    assert g_best.solution.shape == (N_DIMS,)
    assert np.all(np.isfinite(g_best.solution))
    assert np.isfinite(g_best.target.fitness)


def test_legacy_decorator_respects_is_parallelizable(problem):
    optimizer = RandomSearch(epoch=5, pop_size=10)
    optimizer.solve(problem, seed=1)

    assert optimizer.is_parallelizable is False
