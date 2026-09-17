#!/usr/bin/env python
"""Unit tests for ``@cy.agent`` and ``cy.Attribute``."""

import numpy as np
import pytest

import clypto as cy
from clypto.utils.target import Target


@cy.agent
class CustomAgent:
    v: cy.Attribute(int, (1, 100), 5)
    ratio: cy.Attribute(float, (0.0, 1.0), 0.25)
    enabled: cy.Attribute(bool, [True, False], True)


def evaluate(solution):
    return Target(objectives=float(np.sum(solution**2)))


def test_decorator_collects_declared_attributes():
    assert set(CustomAgent._attributes) == {"v", "ratio", "enabled"}


def test_declared_defaults_are_used():
    agent = CustomAgent()

    assert agent.v == 5
    assert agent.ratio == 0.25
    assert agent.enabled is True


def test_attributes_can_be_passed_or_assigned():
    agent = CustomAgent(v=9, ratio=0.5)

    assert agent.v == 9
    assert agent.ratio == 0.5

    agent.v = 12
    assert agent.v == 12


def test_fitness_is_none_until_a_solution_is_evaluated():
    agent = CustomAgent()

    assert agent.solution is None
    assert agent.target is None
    assert agent.fitness is None


def test_solution_assignment_evaluates_and_fitness_is_read_only():
    agent = CustomAgent()
    agent._evaluator = evaluate

    agent.solution = np.array([1.0, 2.0, 0.0])

    assert agent.fitness == pytest.approx(5.0)

    agent.solution = np.array([0.0, 0.0, 0.0])
    assert agent.fitness == pytest.approx(0.0)

    with pytest.raises(AttributeError):
        agent.fitness = 123.0


def test_copy_preserves_solution_target_and_attributes():
    agent = CustomAgent(v=42)
    agent._evaluator = evaluate
    agent.solution = np.array([1.0, 1.0, 1.0])

    clone = agent.copy()

    assert clone is not agent
    assert clone.v == 42
    assert np.allclose(clone.solution, agent.solution)
    assert clone.fitness == pytest.approx(agent.fitness)


def test_decorated_agent_is_a_runtime_agent():
    from clypto.agents.api import RuntimeAgent

    assert issubclass(CustomAgent, RuntimeAgent)
    assert isinstance(CustomAgent(), RuntimeAgent)
