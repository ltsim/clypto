#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The base class injected by ``@cy.optimizer``."""

import time
import typing

import numpy as np
from clypto.optimizer.agents.runtime import RuntimeAgent
from clypto.hints.array import NDArrayType
from clypto.optimizer.bounds import Bounds
from clypto.optimizer.native.population import Population
from clypto.optimizer.native.problem import Problem
from clypto.optimizer.native.target import NativeTarget as Target
from clypto.optimizer.precompile.declaration import Argument
from clypto.optimizer.precompile.decoration import collect_declarations
from clypto.optimizer.termination import Termination
from clypto.optimizer.validator import Validator

__all__ = ["DecoratedOptimizer"]


def _coerce_argument(name: str, declaration: Argument, value: typing.Any) -> typing.Any:
    dtype = declaration.type

    if value is None or dtype is None:
        return value

    bound = declaration.bound

    if dtype in (int, np.integer):
        return Validator.check_int(name, value, bound)
    if dtype in (float, np.floating):
        return Validator.check_float(name, value, bound)
    if dtype is str:
        return Validator.check_str(name, value, bound)
    if dtype is bool:
        return Validator.check_bool(name, value, bound)

    expected = np.ndarray if dtype in (np.ndarray, np.array) else dtype
    if isinstance(value, expected):
        return value
    raise TypeError(f"'{name}' should be an instance of {getattr(expected, '__name__', expected)}.")


class DecoratedOptimizer:
    """Base class injected by ``@cy.optimizer``.

    Subclasses implement ``initialize`` (optional) and ``evolve``. All other
    lifecycle methods — problem binding, population creation, the epoch loop,
    termination and returning the best agent — live here.
    """

    agent_class: typing.Any = RuntimeAgent
    # ``Argument[...]`` is a runtime declaration, not a type annotation; mypy
    # would otherwise parse the subscript as a (non-generic) type application.
    epoch = Argument[int, [1, 1000000], 100]  # type: ignore[type-arg, valid-type]
    pop_size = Argument[int, [5, 10000], 30]  # type: ignore[type-arg, valid-type]

    # Declared (not assigned) so typed ``initialize``/``evolve`` bodies see the
    # concrete types; all are set by ``_bind_problem``/``solve`` before they run.
    problem: Problem
    rng: np.random.Generator
    population: Population
    g_best: RuntimeAgent
    bounds: Bounds
    termination: typing.Optional[Termination]

    def __init__(self, **kwargs: typing.Any) -> None:
        spec = getattr(type(self), "__clypto_arguments__", None)
        if spec is None:
            spec = collect_declarations(type(self), (Argument,))
            type(self).__clypto_arguments__ = spec  # type: ignore[attr-defined]

        self.parameters: dict[str, typing.Any] = {}

        for name, declaration in spec.items():
            value = _coerce_argument(name, declaration, kwargs.pop(name, declaration.default))
            setattr(self, name, value)
            self.parameters[name] = value

        if kwargs:
            raise TypeError(f"Unexpected optimizer argument(s): {sorted(kwargs)}.")

        self._nfe = 0

    # -- lifecycle ---------------------------------------------------------
    def initialize(self) -> None:
        """Hook the algorithm may override to set up ``self.population``.

        The population is already generated (randomly, within bounds) before
        this runs, so a default implementation is not required.
        """
        ...

    def evolve(self, epoch: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement evolve().")

    def solve(
        self,
        problem: dict | Problem,
        termination: typing.Optional[Termination | dict] = None,
        seed: typing.Optional[int] = None,
    ) -> RuntimeAgent:
        """Run the optimizer and return the best agent found."""
        self._bind_problem(problem, seed)
        assert self.problem is not None and self.bounds is not None

        self._nfe = 0
        pop_size = typing.cast(int, self.pop_size)
        epochs = typing.cast(int, self.epoch)
        self.rng = np.random.default_rng(seed)
        self.termination = self._build_termination(termination)
        self.population = Population([self.generate_agent() for _ in range(pop_size)], self.problem.sense)

        self.initialize()
        self.g_best = self.population.best

        for epoch in range(1, epochs + 1):
            self.evolve(epoch)
            self.g_best = self.population.best

            if self.termination is not None and self.termination.should_terminate(
                epoch, self._nfe, time.perf_counter(), 0
            ):
                break

        return self.g_best

    # -- helpers -----------------------------------------------------------
    def evaluate_agent(self, solution: NDArrayType) -> Target:
        """Evaluate one solution, counting a function evaluation."""
        assert self.problem is not None
        self._nfe += 1
        return self.problem.get_target(solution)

    def generate(self, agent: RuntimeAgent, solution: NDArrayType) -> RuntimeAgent:
        """Seed extra state on a freshly created agent before it is evaluated.

        Override this to initialise algorithm-specific attributes (velocity,
        memory, a tag, ...). The agent already carries the declared ``@cy.agent``
        defaults and is not yet evaluated; return the agent to keep, which the
        base then evaluates by assigning its solution.
        """
        return agent

    def generate_agent(self, solution: typing.Optional[NDArrayType] = None) -> RuntimeAgent:
        """Create a fully evaluated agent through the ``generate`` hook."""
        assert self.problem is not None
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)

        agent = self.agent_class()
        object.__setattr__(agent, "_evaluator", self.evaluate_agent)
        agent = self.generate(agent, solution)
        agent.solution = solution
        return agent

    def _bind_problem(self, problem: dict | Problem, seed: typing.Optional[int]) -> None:
        self.problem = Problem.coerce(problem, seed)
        self.bounds = self.problem.bounds

    def _build_termination(
        self, termination: typing.Optional[Termination | dict]
    ) -> typing.Optional[Termination]:
        if termination is None:
            return None
        if isinstance(termination, Termination):
            term = termination
        elif isinstance(termination, dict):
            term = Termination(**termination)
        else:
            raise ValueError("Termination needs to be a dict or an instance of Termination class.")

        term.set_start_values(0, self._nfe, time.perf_counter(), 0)
        return term
