#!/usr/bin/env python
"""Cython JIT compilation of the decorator API (needs the ``compile`` extra)."""

import importlib.machinery
import sys

import numpy as np
import pytest

import clypto as cy
from clypto import Argument

pytest.importorskip("Cython")

N_DIMS = 3


def objective(solution):
    return np.sum(solution**2)


@cy.agent(compile=True)
class FastAgent:
    v: cy.Attribute[float, (0.0, 1.0), 0.5]


@cy.optimizer(agent=FastAgent, compile=True)
class FastSearch:
    alpha: cy.Argument[float, (0.0, 1.0), 0.5]

    def generate(self, agent, solution):
        agent.v = 0.25
        return agent

    def evolve(self, epoch):
        for idx in range(len(self.population)):
            self.population[idx].solution = self.population[idx].solution * (1 - self.alpha)


@cy.optimizer(compile=True)
class DirectImportSearch:
    rank: Argument[int, (0, 10), 1]

    def evolve(self, epoch):
        pass


@cy.optimizer(compile=True)
class ExplicitBaseSearch(cy.DecoratedOptimizer):
    """An explicitly-inheriting class must compile without base injection."""

    rank: cy.Argument[int, (0, 10), 1]

    def evolve(self, epoch):
        pass


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS),
        sense="min",
    )


@pytest.mark.parametrize(
    "cls", [FastAgent, FastSearch, DirectImportSearch, ExplicitBaseSearch]
)
def test_classes_are_compiled(cls):
    assert cls.__clypto_precompiled__ is True
    assert cls.__module__.startswith("_clypto_")

    extension = sys.modules[cls.__module__]
    assert extension.__file__.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))


def test_decorated_methods_are_native():
    assert type(FastSearch.evolve).__name__ == "cython_function_or_method"


def test_compiled_directly_imported_declaration():
    optimizer = DirectImportSearch()

    assert optimizer.parameters["rank"] == 1
    assert type(DirectImportSearch.evolve).__name__ == "cython_function_or_method"


def test_compiled_optimizer_decorator_solves(problem):
    optimizer = FastSearch(epoch=10, pop_size=8)
    g_best = optimizer.solve(problem, seed=1)

    assert g_best.solution.shape == (N_DIMS,)
    assert np.isfinite(g_best.fitness)
    assert all(agent.v == pytest.approx(0.25) for agent in optimizer.population)


def test_compiled_explicit_base_solves(problem):
    optimizer = ExplicitBaseSearch(epoch=5, pop_size=6)
    g_best = optimizer.solve(problem, seed=1)

    assert g_best.solution.shape == (N_DIMS,)
    assert np.isfinite(g_best.fitness)
