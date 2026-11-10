import numpy as np
import typing
from _typeshed import Incomplete
from clypto.hints.array import NDArrayType
from clypto.optimizer.agents.runtime import RuntimeAgent
from clypto.optimizer.bounds import Bounds
from clypto.optimizer.native.agent import Agent
from clypto.optimizer.native.population import Population
from clypto.optimizer.native.problem import Problem
from clypto.optimizer.termination import Termination

__all__ = ['DecoratedOptimizer']

class DecoratedOptimizer:
    """Base class injected by ``@cy.optimizer``.

    Subclasses implement ``initialize`` (optional) and ``evolve``. All other
    lifecycle methods — problem binding, population creation, the epoch loop,
    termination and returning the best agent — live here.
    """
    agent_class: typing.Any
    epoch: Incomplete
    pop_size: Incomplete
    problem: Problem
    rng: np.random.Generator
    population: Population
    g_best: RuntimeAgent
    bounds: Bounds
    termination: Termination | None
    parameters: dict[str, typing.Any]
    def __init__(self, **kwargs: typing.Any) -> None: ...
    def initialize(self) -> None:
        """Hook the algorithm may override to set up ``self.population``.

        The population is already generated (randomly, within bounds) before
        this runs, so a default implementation is not required.
        """
    def evolve(self, epoch: int) -> None: ...
    def solve(self, problem: dict | Problem, termination: Termination | dict | None = None, seed: int | None = None) -> RuntimeAgent:
        """Run the optimizer and return the best agent found."""
    def evaluate_agent(self, solution: NDArrayType) -> Agent:
        """Evaluate one solution (counted by the problem)."""
    def generate(self, agent: RuntimeAgent, solution: NDArrayType) -> RuntimeAgent:
        """Seed extra state on a freshly created agent before it is evaluated.

        Override this to initialise algorithm-specific attributes (velocity,
        memory, a tag, ...). The agent already carries the declared ``@cy.agent``
        defaults and is not yet evaluated; return the agent to keep, which the
        base then evaluates by assigning its solution.
        """
    def generate_agent(self, solution: NDArrayType | None = None) -> RuntimeAgent:
        """Create a fully evaluated agent through the ``generate`` hook."""
