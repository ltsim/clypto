#!/usr/bin/env python
"""Unit tests for the decorator API's :class:`clypto.Population` container."""

import numpy as np
import pytest

import clypto as cy

N_DIMS = 3


def objective(solution):
    return np.sum(solution**2)


@cy.agent
class CountingAgent:
    v: cy.Attribute[int, (1, 100), 5]


@cy.optimizer(agent=CountingAgent)
class Search:
    """A trivial optimizer used only to exercise the Population container."""

    def evolve(self, epoch):
        pass

    def generate_agent(self, solution=None):
        agent = super().generate_agent(solution)
        agent.v = 1
        return agent


def build_population(problem, pop_size=6, minmax="min"):
    optimizer = Search(epoch=1, pop_size=pop_size)
    optimizer._bind_problem(problem, seed=1)
    optimizer.rng = np.random.default_rng(1)
    return cy.Population(pop_size, optimizer.bounds.ndim, optimizer, minmax)


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.FloatVar(lb=[-5.0] * N_DIMS, ub=[5.0] * N_DIMS),
        minmax="min",
    )


def test_population_is_created_and_evaluated(problem):
    population = build_population(problem, pop_size=6)

    assert len(population) == 6
    assert population.solutions.shape == (6, N_DIMS)
    assert population.fitness.shape == (6,)
    assert np.isfinite(population.fitness).all()
    assert all(agent.fitness is not None for agent in population)


def test_population_indexing_and_iteration(problem):
    population = build_population(problem)

    assert population[0] is list(population)[0]
    assert population[-1] is list(population)[-1]


def test_population_solutions_setter_reevaluates(problem):
    population = build_population(problem)
    new_solutions = np.zeros((len(population), N_DIMS))

    population.solutions = new_solutions

    assert np.allclose(population.solutions, 0.0)
    assert np.allclose(population.fitness, 0.0)


def test_population_solutions_shape_is_validated(problem):
    population = build_population(problem)

    with pytest.raises(ValueError):
        population.solutions = np.zeros((len(population) + 1, N_DIMS))


def test_agent_solution_assignment_updates_fitness(problem):
    population = build_population(problem)
    agent = population[0]

    agent.solution = np.array([3.0, 4.0, 0.0])

    assert agent.fitness == pytest.approx(25.0)
    assert population.fitness[0] == pytest.approx(25.0)


def test_agent_fitness_is_read_only(problem):
    population = build_population(problem)

    with pytest.raises(AttributeError):
        population[0].fitness = 100.0


def test_in_place_solution_operator_updates_fitness(problem):
    population = build_population(problem)
    agent = population[0]
    agent.solution = np.array([4.0, 4.0, 4.0])

    population[0].solution /= 2.0

    assert agent.fitness == pytest.approx(np.sum(np.array([2.0, 2.0, 2.0]) ** 2))


def test_best_and_worst_follow_minmax(problem):
    min_population = build_population(problem, minmax="min")
    min_population.solutions = np.array(
        [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert min_population.best.fitness == pytest.approx(0.0)
    assert min_population.worst.fitness == pytest.approx(25.0)

    max_population = build_population(problem, minmax="max")
    max_population.solutions = np.array(
        [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert max_population.best.fitness == pytest.approx(25.0)
    assert max_population.worst.fitness == pytest.approx(0.0)


def test_remove_and_append(problem):
    population = build_population(problem)
    worst = population.worst
    original_size = len(population)

    removed = population.remove(worst.id)

    assert removed is worst
    assert len(population) == original_size - 1
    assert all(agent.id != worst.id for agent in population)

    generated = population.generate()
    population.append(generated)

    assert len(population) == original_size
    assert generated.id is not None
    assert generated.fitness is not None


def test_remove_unknown_id_raises(problem):
    population = build_population(problem)

    with pytest.raises(KeyError):
        population.remove(10_000)


def test_custom_agent_attributes(problem):
    population = build_population(problem)

    assert all(agent.v == 1 for agent in population)
    population[0].v = 42
    assert population[0].v == 42
