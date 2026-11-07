"""``Population`` owns the agent lifecycle: the factory, ``fill``, ``add``, ``remove_at``."""
import numpy as np
import pytest

import clypto as cy
from clypto.optimizer.native.factory import AgentFactory


@pytest.fixture
def problem():
    return cy.Problem(
        obj_func=lambda s: float(np.sum(np.asarray(s) ** 2)),
        bounds=cy.NumberBounds(float, low=[-1.0] * 3, up=[1.0] * 3),
        sense="min",
    )


def bind(pop, problem):
    pop.bind(problem, np.random.default_rng(1))
    return pop


def test_default_factory_generates_evaluated_agents(problem):
    pop = bind(cy.population(5, [1, 100]), problem)
    filled = pop.fill()
    assert len(filled) == pop.size() == 5
    assert type(pop[0]) is cy.Agent
    assert all(np.isfinite(a.fitness) for a in filled)


def test_derived_populations_carry_the_factory(problem):
    pop = bind(cy.population_with_factory(5, AgentFactory()), problem)
    for derived in (pop.empty(), pop.spawn([]), pop.snapshot(), pop.copy()):
        assert derived.factory is pop.factory  # the base factory is shared


def test_assigning_the_factory_rebinds_the_problem(problem):
    pop = cy.population(5, [1, 100])
    pop.bind(problem, np.random.default_rng(2))
    pop.factory = AgentFactory()  # attached after bind: the setter must re-bind
    assert all(np.isfinite(pop.generate_agent().fitness) for _ in range(3))


def test_population_with_factory_binds_positionally(problem):
    pop = bind(cy.population_with_factory(5, AgentFactory()), problem)
    assert len(pop.fill()) == pop.size() == 5


def test_add_and_remove_at_roundtrip(problem):
    pop = bind(cy.population_with_factory(5, AgentFactory()), problem).fill()
    grown = pop.add()
    assert len(pop) == 6 and np.isfinite(grown.fitness)
    removed = pop.remove_at(0)
    assert len(pop) == 5 and np.isfinite(removed.fitness)


def test_fill_takes_starting_solutions(problem):
    pop = bind(cy.population(2, [1, 10]), problem)
    starting = [np.zeros(3), np.ones(3)]
    pop.fill(starting)
    assert np.allclose([a.solution for a in pop], starting, atol=1e-12)
    assert all(np.isfinite(a.fitness) for a in pop)


def test_factory_is_the_single_entry_point(problem):
    pop = bind(cy.population_with_factory(2, AgentFactory(), [1, 10]), problem)
    by_fill = pop.fill()[0]
    default = cy.population(1, [1, 10])
    default.bind(problem, np.random.default_rng(3))
    assert type(by_fill) is type(default.generate_agent()) is cy.Agent
    assert __import__("clypto").population_with_factory.__qualname__.startswith("population")
