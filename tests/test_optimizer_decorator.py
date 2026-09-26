#!/usr/bin/env python
"""Unit tests for ``@cy.optimizer`` and ``cy.Argument``."""

import numpy as np
import pytest

import clypto as cy

N_DIMS = 3


def objective(solution):
    return np.sum(solution**2)


@cy.optimizer
class RandomSearch:
    alpha: cy.Argument[float, (0.0, 1.0), 0.5]

    def evolve(self, epoch):
        for idx in range(len(self.population)):
            candidate = self.generate_agent()
            if candidate.fitness < self.population[idx].fitness:
                self.population[idx].solution = candidate.solution


@cy.optimizer
class CountingSearch:
    calls = 0

    def evolve(self, epoch):
        self.calls += 1


@cy.optimizer
class CenteredSearch:
    def initialize(self):
        self.population.solutions = np.zeros((len(self.population), self.bounds.n_dims))

    def evolve(self, epoch):
        pass


@cy.agent
class TaggedAgent:
    tag: cy.Attribute[int, (0, 100), 0]


@cy.optimizer(agent=TaggedAgent)
class TaggedSearch:
    def evolve(self, epoch):
        pass

    def generate(self, agent, solution):
        agent.tag = 99
        return agent


@cy.optimizer
class ArraySearch:
    weights: cy.Argument[np.ndarray]

    def evolve(self, epoch):
        pass


@cy.optimizer
class FunctionArraySearch:
    weights: cy.Argument[np.array]

    def evolve(self, epoch):
        pass


@cy.optimizer
class DefaultedSearch:
    rate: cy.Argument[float, ..., 0.25]

    def evolve(self, epoch):
        pass


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS),
        sense="min",
    )


def test_default_arguments():
    optimizer = RandomSearch()

    assert optimizer.parameters == {"epoch": 100, "pop_size": 30, "alpha": 0.5}


def test_arguments_are_overridable():
    optimizer = RandomSearch(epoch=25, pop_size=12, alpha=0.75)

    assert optimizer.epoch == 25
    assert optimizer.pop_size == 12
    assert optimizer.alpha == 0.75


def test_argument_bound_is_validated():
    with pytest.raises(TypeError):
        RandomSearch(alpha=2.0)


def test_unknown_argument_raises():
    with pytest.raises(TypeError):
        RandomSearch(beta=1)


def test_missing_evolve_raises():
    with pytest.raises(TypeError):

        @cy.optimizer
        class Broken:
            def initialize(self):
                pass


def test_solve_returns_best_agent(problem):
    optimizer = RandomSearch(epoch=20, pop_size=10)
    g_best = optimizer.solve(problem, seed=1)

    assert g_best is optimizer.g_best
    assert g_best is optimizer.population.best
    assert g_best.solution.shape == (N_DIMS,)
    assert np.all(np.isfinite(g_best.solution))
    assert np.isfinite(g_best.fitness)
    assert optimizer._nfe > optimizer.pop_size


def test_solve_is_reproducible(problem):
    first = RandomSearch(epoch=20, pop_size=10).solve(problem, seed=7)
    second = RandomSearch(epoch=20, pop_size=10).solve(problem, seed=7)

    assert np.allclose(first.solution, second.solution)
    assert first.fitness == pytest.approx(second.fitness)


def test_bounds_view(problem):
    optimizer = RandomSearch(epoch=1, pop_size=5)
    optimizer._bind_problem(problem, seed=1)

    assert optimizer.bounds.n_dims == N_DIMS
    assert np.allclose(optimizer.bounds.low, [-5.0] * N_DIMS)
    assert np.allclose(optimizer.bounds.up, [5.0] * N_DIMS)


def test_initialize_override_is_used(problem):
    optimizer = CenteredSearch(epoch=3, pop_size=8)
    g_best = optimizer.solve(problem, seed=1)

    assert np.allclose(optimizer.population.solutions, 0.0)
    assert g_best.fitness == pytest.approx(0.0)


def test_generate_override(problem):
    optimizer = TaggedSearch(epoch=3, pop_size=8)
    g_best = optimizer.solve(problem, seed=1)

    assert all(agent.tag == 99 for agent in optimizer.population)
    assert g_best.tag == 99


def test_termination_stops_early(problem):
    optimizer = CountingSearch(epoch=1000, pop_size=5)
    optimizer.solve(problem, termination={"max_epoch": 4}, seed=1)

    assert optimizer.calls == 4


def test_array_argument_accepts_numpy_types():
    weights = np.array([0.1, 0.2])

    assert ArraySearch(weights=weights).weights is weights
    assert FunctionArraySearch(weights=weights).weights is weights
    assert ArraySearch().weights is None

    with pytest.raises(TypeError):
        ArraySearch(weights=[0.1, 0.2])


def test_ellipsis_skips_bound_and_keeps_default():
    assert DefaultedSearch().rate == 0.25
    assert DefaultedSearch(rate=0.5).rate == 0.5


def test_public_decorator_api_is_exported():
    for name in ("agent", "optimizer", "legacy", "Attribute", "Argument", "Population", "LegacyOptimizer"):
        assert hasattr(cy, name)

    assert cy.Optimizer is cy.LegacyOptimizer


def test_solve_respects_maximization():
    problem = cy.Problem(
        obj_func=objective,
        bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS),
        sense="max",
    )
    optimizer = RandomSearch(epoch=20, pop_size=10)
    g_best = optimizer.solve(problem, seed=1)

    assert g_best.fitness == pytest.approx(max(optimizer.population.fitness))
