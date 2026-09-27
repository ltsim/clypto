#!/usr/bin/env python
"""Unit tests for :class:`clypto.Population`."""

import numpy as np
import pytest

import clypto as cy

N_DIMS = 3


def objective(solution):
    return np.sum(solution**2)


def agents(*fitness):
    return [cy.Agent(np.full(N_DIMS, f), f) for f in fitness]


def bound(items=(), sense="min", size=None):
    """A population of ``items`` bound to a small problem of the given sense."""
    problem = cy.Problem(bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS), sense=sense, obj_func=objective)
    population = cy.population(size or max(len(items), 5))
    population.bind(problem, np.random.default_rng(1))
    population.extend(items)
    return population


@cy.agent
class CountingAgent:
    v: cy.Attribute[int, (1, 100), 5]


@cy.optimizer(agent=CountingAgent)
class Search:
    """A trivial optimizer used only to build a decorator-API population."""

    def evolve(self, epoch):
        pass

    def generate(self, agent, solution):
        agent.v = 1
        return agent


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS),
        sense="min",
    )


def runtime_population(problem, pop_size=6):
    optimizer = Search(epoch=1, pop_size=pop_size)
    optimizer._bind_problem(problem, seed=1)
    optimizer.rng = np.random.default_rng(1)
    return optimizer, bound([optimizer.generate_agent() for _ in range(pop_size)])


def test_generic_alias_builds_a_population():
    population = cy.Population[cy.Agent](10, agents(3.0, 1.0))

    assert isinstance(population, cy.Population)
    assert population.sense == "min"  # unbound
    assert len(population) == 2 and population.size() == 10


def test_population_validates_its_size_and_binds_later():
    population = cy.population(30, range=[5, 100])

    assert population.size() == 30 and len(population) == 0
    with pytest.raises(TypeError):
        cy.population(3)
    population.bind(bound().problem, np.random.default_rng(1))
    generated = population.generate()
    assert len(generated) == 30 and generated.size() == 30 and population.problem.n_evals == 30


def test_subclass_state_survives_slices_and_sort():
    class Tagged(cy.Population):
        pass

    population = Tagged(4, agents(3.0, 1.0))
    population.tag = "kept"
    for derived in (population[:1], population.sort(), population + agents(2.0), population.copy()):
        assert type(derived) is Tagged and derived.tag == "kept" and derived.size() == 4


def test_indexing_and_slices():
    items = agents(3.0, 1.0, 2.0, 5.0)
    population = bound(items)

    assert population[0] is items[0] and population[-1] is items[-1]
    part = population[1:3]
    assert isinstance(part, cy.Population) and list(part) == items[1:3]


def test_mutable_sequence_operations():
    a, b, c, d = agents(3.0, 1.0, 2.0, 5.0)
    population = bound([a, b])

    population.append(c)
    population += [d]
    assert list(population) == [a, b, c, d]
    assert population.popleft() is a
    population.remove(c)
    assert list(population) == [b, d]
    assert list([a] + population) == [a, b, d]
    assert isinstance(population + [a], cy.Population)


def test_wrapping_a_list_shares_it():
    items = agents(3.0, 1.0)
    population = cy.Population(5, items)

    population.append(agents(2.0)[0])
    assert len(items) == 3


@pytest.mark.parametrize("sense, best, worst", [("min", 1.0, 5.0), ("max", 5.0, 1.0)])
def test_best_worst_and_sort_follow_sense(sense, best, worst):
    population = bound(agents(3.0, 1.0, 2.0, 5.0), sense)

    assert population.best.fitness == best
    assert population.worst.fitness == worst
    ranked = population.sort()
    assert ranked[0].fitness == best
    assert [population[i] for i in ranked.idx] == list(ranked)
    assert ranked.idx == population.argsort()


def test_argsort_matches_numpy():
    population = bound(agents(3.0, 1.0, 1.0, 5.0), "max")

    assert population.argsort() == np.argsort([3.0, 1.0, 1.0, 5.0]).tolist()[::-1]


@pytest.mark.parametrize("sense, expected", [("min", [1.0, 2.0, 2.0]), ("max", [3.0, 4.0, 2.0])])
def test_greedy_keeps_strict_improvements(sense, expected):
    population = bound(agents(3.0, 2.0, 2.0), sense)

    kept = population.greedy(agents(1.0, 4.0, 2.0))
    assert kept.fitness.tolist() == expected


def test_arrays():
    population = bound(agents(3.0, 1.0))

    assert population.fitness.tolist() == [3.0, 1.0]
    assert population.solutions.shape == (2, N_DIMS)


def test_copy_is_shallow_duplicate_is_deep():
    population = bound(agents(3.0, 1.0))

    shallow, deep = population.copy(), population.duplicate()
    assert shallow[0] is population[0]
    assert deep[0] is not population[0] and deep[0].fitness == 3.0
    shallow.append(agents(2.0)[0])
    assert len(population) == 2


def test_solutions_setter_reevaluates_runtime_agents(problem):
    _, population = runtime_population(problem)

    population.solutions = np.zeros((len(population), N_DIMS))
    assert np.allclose(population.solutions, 0.0)
    assert np.allclose(population.fitness, 0.0)
    with pytest.raises(ValueError):
        population.solutions = np.zeros((len(population) + 1, N_DIMS))


def test_runtime_agents_keep_their_attributes(problem):
    optimizer, population = runtime_population(problem)

    assert all(agent.v == 1 for agent in population)
    population.remove(population.worst)
    population.append(optimizer.generate_agent())
    assert len(population) == 6
    assert np.isfinite(population.fitness).all()


def test_agent_factory_repair_and_evaluation():
    population = bound()
    agent = population.generate_agent(np.array([1.0, 2.0, 0.0]))

    assert agent.fitness == 5.0 and agent.objectives.tolist() == [5.0]
    assert population.amend_solution(np.array([9.0, -9.0, 0.0])).tolist() == [5.0, -5.0, 0.0]
    candidate = population.evaluate_solution(np.zeros(N_DIMS))
    assert type(candidate) is cy.Agent and candidate.fitness == 0.0
    with pytest.raises(AttributeError):
        agent.fitness = 1.0
    agent.update_solution(candidate)
    assert agent.fitness == 0.0 and agent.solution is not candidate.solution


def test_reset_population_redraws_out_of_bounds_values():
    population = cy.population(5, range=[5, 10000], cls=cy.ResetPopulation)
    population.bind(bound().problem, np.random.default_rng(3))

    fixed = population.amend_solution(np.array([9.0, 1.0, -9.0]))
    assert fixed[1] == 1.0 and -5.0 <= fixed[0] <= 5.0 and fixed[0] != 5.0
