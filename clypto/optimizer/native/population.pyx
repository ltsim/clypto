"""``Population`` (the single, final container of agents) and the bounds-repair functions.

``Population`` holds the agents of the run (``agents``), the bound problem and the
seeded generator, and evaluates/ranks them. ``size()`` is the configured number of
agents (``pop_size``); ``len()`` is how many it holds right now. Slices, ``+``,
``sort()`` and ``greedy()`` give new populations bound to the same problem.

Evaluation follows ``mode``: ``"sequential"`` evaluates only the agents that are
not evaluated yet (agent by agent), ``"batch"`` and ``"parallel"`` evaluate the
whole batch at once (``"parallel"`` on OpenMP threads when the problem has a nogil
``evaluator``). ``threads`` is set once by the engine's ``solve``.

Bounds repair is a set of plain functions (``correct_solution``, ``reset_solution``,
``opposite_solution``), not population methods.
"""
import cython

from collections.abc import MutableSequence

import numpy as np
cimport numpy as cnp

from clypto.optimizer.native.agent cimport Agent, duplicate_agent, greedy_agents, sort_agents, better_fitness
from clypto.optimizer.native.problem cimport Problem
from clypto.optimizer.native.utils cimport validator


@cython.final
cdef class Population:
    """Ordered, mutable sequence of agents, ranked by the bound problem's ``sense``.

    ``size()`` is the configured number of agents (``pop_size``); ``len()`` is
    how many it holds right now. Slices, ``+``, ``sort()`` and ``greedy()`` give
    new populations of the same problem. ``fitness``/``solutions`` are arrays
    built from the agents; assigning ``solutions`` writes each row back to its agent.
    """

    def __init__(self, size=0, agents=()):
        self._size = size
        self.threads = 0
        self._minimize = True
        # A list is wrapped, not copied, so code holding the list sees the same agents.
        self.agents = agents if type(agents) is list else list(agents)

    def bind(self, problem, generator):
        """Attach the problem and the seeded generator, and drop the agents of a previous run."""
        self.problem = problem
        self.generator = generator
        self.agents = []
        self._minimize = problem.sense == "min"

    cpdef Population spawn(self, object agents):
        """A population bound to the same problem holding ``agents`` (a list is wrapped, not copied)."""
        cdef Population new = self.empty()
        new.agents = agents if type(agents) is list else list(agents)
        return new

    cpdef Population empty(self):
        """An empty population bound to the same problem, generator and settings."""
        cdef Population new = Population(self._size)
        new.problem = self.problem
        new.generator = self.generator
        new.threads = self.threads
        new._minimize = self._minimize
        return new

    cpdef Population snapshot(self):
        """A population of copies (``duplicate_agent``) of every agent."""
        return self.spawn([duplicate_agent(<Agent>a) for a in self.agents])

    cpdef Population copy(self):
        """Shallow copy: the same agent objects, bound to the same problem."""
        return self.spawn(list(self.agents))

    cpdef Py_ssize_t size(self):
        """The configured number of agents (``pop_size``)."""
        return self._size

    def resize(self, n):
        self._size = n

    cpdef list toarray(self):
        """The agents, as the live list (a typed ``for`` over it is a C loop)."""
        return self.agents

    @property
    def sense(self):
        return "min" if self.problem is None else self.problem.sense

    # -- agents ------------------------------------------------------------------
    def create_agent(self, solution=None):
        """A new, not yet evaluated agent at ``solution`` (random when ``None``)."""
        if solution is None:
            solution = self.problem.generate_solution(True)
        return Agent(solution)

    def generate_agent(self, solution=None):
        """``create_agent`` then evaluate it."""
        agent = self.create_agent(solution)
        (<Agent>agent).evaluate(self.problem)
        return agent

    def generate(self, n=None, starting=None):
        """A population of ``n`` (default ``size()``) generated agents, or one per ``starting`` solution."""
        if starting is not None:
            return self.spawn([self.generate_agent(x) for x in starting])
        return self.spawn([self.generate_agent() for _ in range(self._size if n is None else n)])

    def evaluate_solution(self, solution):
        """An evaluated plain :class:`Agent` at ``solution`` (a candidate; no random draws)."""
        return Agent(solution).evaluate(self.problem)

    cdef object _batch_matrix(self, Py_ssize_t k):
        return np.empty((k, self.problem.n_dims), dtype=np.float64)

    cpdef object evaluate(self, object agents, str mode):
        """Evaluate ``agents`` (the batch step) and return them.

        ``"sequential"`` evaluates only the agents not evaluated yet (an algorithm
        that evaluates one by one has done it already); ``"batch"``/``"parallel"``
        evaluate every agent. The batch is one ``problem.evaluate(X)`` call, on
        OpenMP threads in ``"parallel"`` mode when the problem has a nogil
        evaluator (``self.threads`` fixes the team size, 0 lets OpenMP decide).
        """
        cdef Agent agent
        cdef Py_ssize_t i, k
        cdef list items, pending
        cdef object weights, fitness, O, X
        items = agents.agents if isinstance(agents, Population) else list(agents)

        if mode == "sequential":
            pending = [agent for agent in items if agent.objectives is None]
            for agent in pending:
                (<Agent>agent).evaluate(self.problem)
            return agents

        k = len(items)
        if k == 0:
            return agents
        try:
            X = self._batch_matrix(k)
            for i in range(k):
                agent = <Agent>items[i]
                X[i, :] = agent.solution
            fitness, O = self.problem.evaluate(
                X, mode == "parallel" and k >= MIN_PARALLEL_BATCH, self.threads if self.threads > 0 else 0
            )
            weights = self.problem.obj_weights
            for i in range(k):
                agent = <Agent>items[i]
                agent.set_evaluation(O[i], weights)
        except (TypeError, ValueError):
            # Non-numeric solutions (string/permutation bounds) cannot share one batch
            # matrix: fall back to the per-agent evaluation (duck-typed, so the
            # decorator API's agents also work).
            for agent in items:
                agent.evaluate(self.problem)
        return agents

    # -- sequence protocol -------------------------------------------------------
    def __len__(self):
        return len(self.agents)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.spawn(self.agents[index])
        return self.agents[index]

    def __setitem__(self, index, value):
        self.agents[index] = value

    def __delitem__(self, index):
        del self.agents[index]

    def __iter__(self):
        return iter(self.agents)

    def __reversed__(self):
        return reversed(self.agents)

    def __contains__(self, agent):
        return agent in self.agents

    def __add__(self, other):
        return self.spawn(self.agents + list(other))

    def __radd__(self, other):
        return self.spawn(list(other) + self.agents)

    def __iadd__(self, other):
        self.agents.extend(other)
        return self

    def __repr__(self):
        return f"{type(self).__name__}(size={self._size}, {self.agents!r})"

    def insert(self, index, agent):
        self.agents.insert(index, agent)

    cpdef void append(self, object agent):
        self.agents.append(agent)

    def extend(self, agents):
        self.agents.extend(agents)

    def pop(self, index=-1):
        return self.agents.pop(index)

    def popleft(self):
        return self.agents.pop(0)

    def remove(self, agent):
        self.agents.remove(agent)

    def clear(self):
        self.agents.clear()

    def index(self, agent, *args):
        return self.agents.index(agent, *args)

    def count(self, agent):
        return self.agents.count(agent)

    def reverse(self):
        self.agents.reverse()

    # -- arrays ------------------------------------------------------------------
    @property
    def fitness(self):
        """``(n,)`` fitness of every agent."""
        return np.array([agent.fitness for agent in self.agents], dtype=float)

    @property
    def solutions(self):
        """``(n, n_dims)`` solution matrix."""
        return np.array([agent.solution for agent in self.agents], dtype=float)

    @solutions.setter
    def solutions(self, values):
        values = np.atleast_2d(np.asarray(values, dtype=float))
        if values.shape[0] != len(self.agents):
            raise ValueError(f"Expected {len(self.agents)} solutions, got {values.shape[0]}.")
        for agent, row in zip(self.agents, values):
            agent.solution = row

    # -- ranking -----------------------------------------------------------------
    def argsort(self):
        """Positions best first (``np.argsort`` of the fitness, reversed when maximizing)."""
        fitness = np.array([agent.fitness for agent in self.agents], dtype=float)
        order = np.argsort(fitness).tolist()
        return order if self._minimize else order[::-1]

    cpdef Population sort(self):
        """New population, best first."""
        return self.spawn(self.selected(self.argsort()))

    def selected(self, order):
        """The agents at ``order`` (positions of ``argsort``), as a plain list."""
        return [self.agents[i] for i in order]

    cpdef list top(self, Py_ssize_t k):
        """The ``k`` best agents, best first (a partial ``sort()[:k]``)."""
        if k <= 0:
            return []
        order = self.argsort()
        if k > len(order):
            k = len(order)
        return [self.agents[i] for i in order[:k]]

    cpdef list bottom(self, Py_ssize_t k):
        """The ``k`` worst agents, worst first."""
        if k <= 0:
            return []
        order = self.argsort()
        if k > len(order):
            k = len(order)
        return [self.agents[i] for i in order[len(order) - k:]]

    @property
    def best(self):
        """The first agent of the best fitness (strictly better replaces; ties keep the first)."""
        if len(self.agents) == 0:
            return None
        sense = self.sense_str()
        best_agent = self.agents[0]
        best_fit = best_agent.fitness
        for agent in self.agents:
            if better_fitness(agent.fitness, best_fit, sense):
                best_agent, best_fit = agent, agent.fitness
        return best_agent

    @property
    def worst(self):
        """The first agent of the worst fitness (ties keep the first)."""
        if len(self.agents) == 0:
            return None
        sense = self.sense_str()
        worst_agent = self.agents[0]
        worst_fit = worst_agent.fitness
        for agent in self.agents:
            if better_fitness(worst_fit, agent.fitness, sense):
                worst_agent, worst_fit = agent, agent.fitness
        return worst_agent

    cpdef Population greedy(self, object candidates, str mode="swarm"):
        """Per position, the candidate when better, else the current agent (``greedy_agents``, same ``mode`` rule)."""
        return self.spawn(greedy_agents(self.agents, candidates, self.sense_str(), mode))

    cdef str sense_str(self):
        if self.problem is not None:
            return self.problem.sense
        return "min" if self._minimize else "max"


cdef Py_ssize_t MIN_PARALLEL_BATCH = 32


cpdef Population population(object size, object range=[5, 10000], type cls=None):
    """An unbound population of ``size`` agents (``cla`` is ignored: the population is a single class)."""
    if cls is not None and cls is not Population:
        raise ValueError(
            "Population is a single concrete class: population(size, cls=...) no longer accepts custom classes."
        )
    return Population(validator(int, size, range, "pop_size"))


cpdef Population empty_snapshot(Population population):
    """An empty population of the same problem and state."""
    return population.empty()


cpdef Population snapshot(Population population):
    """A population holding a copy (``duplicate_agent``) of every agent."""
    return population.snapshot()


cpdef object correct_solution(object problem, object solution):
    """Clip ``solution`` to the bounds, then apply the problem's own correction."""
    return problem.correct_solution(np.clip(solution, problem.bounds.low, problem.bounds.up))


cpdef object reset_solution(object problem, object generator, object solution):
    """Redraw the out-of-bounds values of ``solution`` uniformly, then apply the problem's own correction.

    The draw covers every dimension (one ``generator.uniform(low, up)`` call), as the clip-free MEALPY repair did.
    """
    low, up = problem.bounds.low, problem.bounds.up
    return problem.correct_solution(np.where((low <= solution) & (solution <= up), solution, generator.uniform(low, up)))


cpdef object opposite_solution(object problem, object generator, Agent agent, Agent g_best):
    """Opposition-based candidate of ``agent`` around ``g_best``, corrected."""
    pos_new = (
        problem.bounds.low + problem.bounds.up - g_best.solution
        + generator.uniform() * (g_best.solution - agent.solution)
    )
    return correct_solution(problem, pos_new)


MutableSequence.register(Population)
