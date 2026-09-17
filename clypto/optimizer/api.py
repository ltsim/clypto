#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""Decorator API for writing and compiling optimizers.

Three decorators cover the whole surface:

* ``@cy.optimizer(agent=MyAgent)`` — a new-style optimizer. Its hyper-parameters
  are declared as :class:`Argument` attributes and it implements ``initialize``
  and ``evolve``. The base class supplies ``solve``, ``self.population``,
  ``self.rng``, ``self.bounds`` and ``generate_agent``.
* ``@cy.agent`` — a new-style agent with optional :class:`Attribute` fields.
* ``@cy.legacy(precompile=False)`` — the classic MEALPY-style API (``self.pop``,
  ``self.validator``, ``generate_empty_agent``, ...) without having to name the
  base class; ``precompile=True`` Cython-compiles it.

All three inject their base class instead of asking for inheritance, and all
three accept ``compile``/``precompile`` to run the same JIT Cython builder.
"""

import re
import time
import typing

import numpy as np
from clypto.agents.api import Attribute, RuntimeAgent
from clypto.hints.array import NDArrayType
from clypto.optimizer.classic import LegacyOptimizer
from clypto.utils.population import Population
from clypto.utils.problem import Problem
from clypto.utils.target import Target
from clypto.utils.termination import Termination
from clypto.utils.validator import Validator

__all__ = ["Argument", "DecoratedOptimizer", "optimizer", "legacy"]


class Argument:
    """Declare an optimizer hyper-parameter.

    Example::

        @cy.optimizer(agent=MyAgent)
        class MyOptimizer:
            a: cy.Argument(int, (1, 100), 5)
            c: cy.Argument(np.ndarray)

    Args:
        dtype: Expected type. ``np.array``/``np.ndarray`` accept array values.
        bound: Optional validation range (``tuple`` exclusive, ``list`` inclusive).
        default: Value used when the argument is not passed to the constructor.
    """

    __slots__ = ("type", "bound", "default")

    def __init__(self, dtype: typing.Any = None, bound: typing.Any = None, default: typing.Any = None) -> None:
        self.type = dtype
        self.bound = bound
        self.default = default


class _Bounds:
    """A tiny ``lb``/``ub``/``ndim`` view of the bound problem."""

    __slots__ = ("lb", "ub", "ndim")

    def __init__(self, lb: NDArrayType, ub: NDArrayType, ndim: int) -> None:
        self.lb = lb
        self.ub = ub
        self.ndim = ndim

    def __repr__(self) -> str:
        return f"Bounds(ndim={self.ndim})"


def _collect_arguments(cls: typing.Any) -> dict[str, Argument]:
    """Collect ``Argument`` declarations from a class and its bases."""
    spec: dict[str, Argument] = {}

    for base in reversed(cls.__mro__):
        for name, value in vars(base).items():
            if isinstance(value, Argument):
                spec[name] = value
        for name, declaration in (getattr(base, "__annotations__", {}) or {}).items():
            if isinstance(declaration, Argument):
                spec[name] = declaration

    return spec


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


def _decorate_with_base(cls: typing.Any, base: typing.Any) -> typing.Any:
    """Return ``cls`` with ``base`` injected as an *additional* base.

    CPython refuses to reassign ``__bases__`` when the only base is ``object``
    ("deallocator differs from 'object'"), so the decorator returns a new class
    that inherits from both the user class and the base. Keeping the original
    class in the MRO preserves zero-arg ``super()`` calls such as
    ``super().__init__(**kwargs)`` in a ``@cy.legacy`` class.
    """
    if issubclass(cls, base):
        return cls

    namespace: dict[str, typing.Any] = {
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__": cls.__doc__,
        "__annotations__": {},
    }
    return type(cls.__name__, (cls, base), namespace)


def _class_source_with_base(cls: typing.Any, base_name: str) -> str:
    """Return the class source with ``base_name`` inserted into its header."""
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(cls))
    pattern = re.compile(rf"^class\s+{re.escape(cls.__name__)}\s*:", re.M)
    # ``inspect.getsource`` returns the decorator lines too; only the class
    # header is rewritten, and only when it has no bases of its own.
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"class {cls.__name__}({base_name}):"

    source = pattern.sub(replace, source, count=1)
    if count == 0:
        raise RuntimeError(f"Could not locate class {cls.__name__} in its own source.")

    # Cython compiles `name: cy.Argument(...)` inside a plain class as a type
    # annotation and drops it, so the declarations would vanish from the
    # compiled class. Materialize them as ordinary assignments instead.
    declarations = [
        name
        for name, value in (getattr(cls, "__annotations__", {}) or {}).items()
        if isinstance(value, (Argument, Attribute))
    ]
    for name in declarations:
        assignment = re.compile(
            rf"^(\s*){re.escape(name)}\s*:\s*(cy\.(?:Argument|Attribute)\b.*)$", re.M
        )
        source = assignment.sub(rf"\g<1>{name} = \g<2>", source, count=1)

    return source


def _maybe_compile(
    decorated: typing.Any,
    enabled: bool,
    *,
    source_cls: typing.Optional[typing.Any] = None,
    base_name: typing.Optional[str] = None,
    import_line: typing.Optional[str] = None,
) -> typing.Any:
    """Return the JIT-compiled class when ``enabled``, else ``decorated``.

    ``source_cls`` is the original user class (needed because ``decorated`` is
    built dynamically and has no source of its own).
    """
    if not enabled:
        return decorated

    from clypto.precompile import compile_decorated, is_precompiling

    # While the compiled module imports, its decorator runs again; bail out
    # before touching inspect.getsource (unavailable for a compiled module).
    if is_precompiling():
        return decorated

    original = source_cls or decorated
    source = _class_source_with_base(original, base_name) if base_name else None
    return compile_decorated(original, source=source, imports=[import_line] if import_line else None)


class DecoratedOptimizer:
    """Base class injected by ``@cy.optimizer``.

    Subclasses implement ``initialize`` (optional) and ``evolve``. All other
    lifecycle methods — problem binding, population creation, the epoch loop,
    termination and returning the best agent — live here.
    """

    agent_class: typing.Any = RuntimeAgent
    epoch = Argument(int, [1, 1000000], 100)
    pop_size = Argument(int, [5, 10000], 30)

    def __init__(self, **kwargs: typing.Any) -> None:
        spec = getattr(type(self), "__clypto_arguments__", None)
        if spec is None:
            spec = _collect_arguments(type(self))
            setattr(type(self), "__clypto_arguments__", spec)

        self.parameters: dict[str, typing.Any] = {}

        for name, declaration in spec.items():
            value = _coerce_argument(name, declaration, kwargs.pop(name, declaration.default))
            setattr(self, name, value)
            self.parameters[name] = value

        if kwargs:
            raise TypeError(f"Unexpected optimizer argument(s): {sorted(kwargs)}.")

        self._nfe = 0
        self.problem: typing.Optional[Problem] = None
        self.rng: typing.Optional[np.random.Generator] = None
        self.population: typing.Optional[Population] = None
        self.g_best: typing.Optional[RuntimeAgent] = None
        self.bounds: typing.Optional[_Bounds] = None
        self.termination: typing.Optional[Termination] = None

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
        self.population = Population(pop_size, self.bounds.ndim, self, self.problem.minmax)

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

    def generate_agent(self, solution: typing.Optional[NDArrayType] = None) -> RuntimeAgent:
        """Create a fully evaluated agent; override to seed custom attributes."""
        assert self.problem is not None
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)

        agent = self.agent_class()
        object.__setattr__(agent, "_evaluator", self.evaluate_agent)
        agent.solution = solution
        return agent

    def _bind_problem(self, problem: dict | Problem, seed: typing.Optional[int]) -> None:
        if isinstance(problem, Problem):
            problem.seed = seed
            self.problem = problem
        elif isinstance(problem, dict):
            self.problem = Problem(**{**problem, "seed": seed})
        else:
            raise ValueError("problem needs to be a dict or an instance of Problem class.")

        self.bounds = _Bounds(
            np.asarray(self.problem.lb, dtype=float),
            np.asarray(self.problem.ub, dtype=float),
            self.problem.n_dims,
        )

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


def optimizer(
    cls: typing.Optional[typing.Any] = None,
    *,
    agent: typing.Optional[typing.Any] = None,
    compile: bool = False,
):
    """Turn a plain class into a new-style optimizer (see module docstring)."""

    def decorate(user_cls):
        if "evolve" not in user_cls.__dict__:
            raise TypeError(f"{user_cls.__name__} must define an evolve(self, epoch) method.")

        decorated = _decorate_with_base(user_cls, DecoratedOptimizer)

        if agent is not None:
            selected_agent = agent
            if not issubclass(selected_agent, RuntimeAgent):
                from clypto.agents.api import agent as agent_decorator

                selected_agent = agent_decorator(selected_agent)
            setattr(decorated, "agent_class", selected_agent)

        setattr(decorated, "__clypto_arguments__", _collect_arguments(decorated))

        return _maybe_compile(
            decorated,
            compile,
            source_cls=user_cls,
            base_name="DecoratedOptimizer",
            import_line="from clypto.optimizer.api import DecoratedOptimizer",
        )

    return decorate(cls) if cls is not None else decorate


def legacy(cls: typing.Optional[typing.Any] = None, *, precompile: bool = False):
    """Use the classic optimizer API without inheriting from it explicitly.

    ``precompile=True`` Cython-compiles the class through the JIT builder.
    """

    def decorate(user_cls):
        decorated = _decorate_with_base(user_cls, LegacyOptimizer)
        return _maybe_compile(
            decorated,
            precompile,
            source_cls=user_cls,
            base_name="LegacyOptimizer",
            import_line="from clypto import LegacyOptimizer",
        )

    return decorate(cls) if cls is not None else decorate
