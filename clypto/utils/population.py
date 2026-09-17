#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The population container used by ``@cy.optimizer``.

A :class:`Population` owns a list of runtime agents and exposes the views an
algorithm's ``initialize``/``evolve`` methods work with: the whole solution
matrix (``population.solutions``), the fitness vector (``population.fitness``),
the best/worst agents, and structural operations (``append``/``remove``/
``generate``).
"""

import typing

import numpy as np
from clypto.agents.api import RuntimeAgent
from clypto.hints.array import NDArrayType

__all__ = ["Population"]


class Population:
    """A mutable ordered collection of solution agents.

    Args:
        n_pop: Number of agents to create up front.
        ndim: Number of decision variables per solution.
        optimizer: The owning optimizer, used to generate and evaluate agents.
        minmax: ``"min"`` or ``"max"``; controls which agent is ``best``/``worst``.
    """

    def __init__(
        self,
        n_pop: int,
        ndim: int,
        optimizer: typing.Any,
        minmax: str = "min",
    ) -> None:
        self._optimizer = optimizer
        self._ndim = int(ndim)
        self._minmax = minmax
        self._next_id = 0
        self._agents: list[RuntimeAgent] = []

        for _ in range(int(n_pop)):
            self.append(optimizer.generate_agent())

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def minmax(self) -> str:
        return self._minmax

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> typing.Iterator[RuntimeAgent]:
        return iter(self._agents)

    def __getitem__(self, index: typing.Any) -> typing.Any:
        return self._agents[index]

    @property
    def solutions(self) -> NDArrayType:
        """The whole population as a 2-D ``(n_pop, ndim)`` matrix."""
        if not self._agents:
            return np.empty((0, self._ndim))
        return np.array([agent.solution for agent in self._agents], dtype=float)

    @solutions.setter
    def solutions(self, values: typing.Any) -> None:
        # Assigning the matrix routes every row through the agent's solution
        # setter, so each agent is re-evaluated and its fitness refreshed.
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[0] != len(self._agents):
            raise ValueError(
                f"Expected {len(self._agents)} solutions, got {values.shape[0]}."
            )
        for agent, row in zip(self._agents, values, strict=True):
            agent.solution = row

    @property
    def fitness(self) -> NDArrayType:
        """The fitness of every agent, aligned with :attr:`solutions`."""
        return np.array([agent.fitness for agent in self._agents], dtype=float)

    @property
    def best(self) -> RuntimeAgent:
        """The best agent according to the problem's ``minmax`` sense."""
        return self._agents[self._extreme_index(best=True)]

    @property
    def worst(self) -> RuntimeAgent:
        """The worst agent according to the problem's ``minmax`` sense."""
        return self._agents[self._extreme_index(best=False)]

    def append(self, agent: RuntimeAgent) -> RuntimeAgent:
        """Append an agent, binding it to this population's evaluator and ids."""
        if getattr(agent, "_evaluator", None) is None:
            object.__setattr__(agent, "_evaluator", self._optimizer.evaluate_agent)
        if agent.id is None:
            agent.id = self._next_id
            self._next_id += 1
        self._agents.append(agent)
        return agent

    def remove(self, agent_id: typing.Any) -> RuntimeAgent:
        """Remove and return the agent with ``agent_id``."""
        for idx, agent in enumerate(self._agents):
            if agent.id == agent_id:
                return self._agents.pop(idx)
        raise KeyError(agent_id)

    def generate(self) -> RuntimeAgent:
        """Generate (but do not append) a fresh, evaluated agent."""
        return self._optimizer.generate_agent()

    def _extreme_index(self, best: bool) -> int:
        if not self._agents:
            raise ValueError("Population is empty.")
        fitness = self.fitness
        if self._minmax == "min":
            return int(np.argmin(fitness)) if best else int(np.argmax(fitness))
        return int(np.argmax(fitness)) if best else int(np.argmin(fitness))
