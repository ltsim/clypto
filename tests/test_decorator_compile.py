#!/usr/bin/env python
"""Cython JIT compilation of the decorator API (needs the ``compile`` extra)."""

import importlib.machinery
import sys

import numpy as np
import pytest

import clypto as cy

pytest.importorskip("Cython")

N_DIMS = 3


def objective(solution):
    return np.sum(solution**2)


@cy.legacy(precompile=True)
class LegacyRS:
    def __init__(self, epoch=50, pop_size=25, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(self.problem.generate_solution(encoded=True))
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.get_target(pos_new)
            self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)


@cy.agent(compile=True)
class FastAgent:
    v: cy.Attribute(float, (0.0, 1.0), 0.5)


@cy.optimizer(agent=FastAgent, compile=True)
class FastSearch:
    alpha: cy.Argument(float, (0.0, 1.0), 0.5)

    def evolve(self, epoch):
        for idx in range(len(self.population)):
            self.population[idx].solution = self.population[idx].solution * (1 - self.alpha)
            self.population[idx].v = self.alpha


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.FloatVar(lb=[-5.0] * N_DIMS, ub=[5.0] * N_DIMS),
        minmax="min",
    )


@pytest.mark.parametrize("cls", [LegacyRS, FastAgent, FastSearch])
def test_classes_are_compiled(cls):
    assert cls.__clypto_precompiled__ is True
    assert cls.__module__.startswith("_clypto_")

    extension = sys.modules[cls.__module__]
    assert extension.__file__.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))


def test_decorated_methods_are_native():
    assert type(LegacyRS.evolve).__name__ == "cython_function_or_method"
    assert type(FastSearch.evolve).__name__ == "cython_function_or_method"


def test_compiled_legacy_decorator_solves(problem):
    g_best = LegacyRS(epoch=30, pop_size=15).solve(problem, seed=1)

    assert g_best.solution.shape == (N_DIMS,)
    assert np.isfinite(g_best.target.fitness)


def test_compiled_optimizer_decorator_solves(problem):
    optimizer = FastSearch(epoch=10, pop_size=8)
    g_best = optimizer.solve(problem, seed=1)

    assert g_best.solution.shape == (N_DIMS,)
    assert np.isfinite(g_best.fitness)
    assert all(agent.v == pytest.approx(0.5) for agent in optimizer.population)
