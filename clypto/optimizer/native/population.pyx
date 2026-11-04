"""``Population`` (a sequence of agents), its snapshots and the bounds-repair functions.

``Population`` is ``self.population`` in the engine and the decorator API. It is
declared in the optimizer's ``__init__`` with ``cy.population(pop_size)``, bound
to the problem by ``solve()`` (``bind``), and builds and evaluates the agents
(``create_agent``/``generate_agent``, ``evaluate_solution``/``evaluate``). An
algorithm with its own agents subclasses it, overrides those methods and, when
it declares fields, ``copy_state`` (copies are static C code, never ``setattr``).

``evaluate(agents, mode)`` is the batch step of the pipeline
``empty_snapshot`` -> ``evaluate`` -> ``greedy``: one ``problem.evaluate(X)`` call
when the problem is vectorized or, in ``"parallel"`` mode, has a nogil evaluator.

Bounds repair is a set of plain functions (``correct_solution``,
``reset_solution``, ``opposite_solution``), not population methods.
"""
from collections.abc import MutableSequence

import numpy as np

from clypto.optimizer.native.agent cimport Agent, argsort_agents, duplicate_agent, greedy_agents
from clypto.optimizer.native.utils cimport validator


cpdef Population population(object size, object range=[5, 10000], type cls=None):
    """An unbound population of ``size`` agents (``cls``, default :class:`Population`).

    ``size`` is validated like ``cy.validator(int, size, range, "pop_size")``.
    """
    return (Population if cls is None else cls)(validator(int, size, range, "pop_size"))


cpdef Population empty_snapshot(Population population):
    """An empty population of the same class and state (problem, generator, size, fields)."""
    return population.spawn([])


cpdef Population snapshot(Population population):
    """A population of the same class and state holding a copy (``duplicate_agent``) of every agent."""
    return population.spawn([duplicate_agent(agent) for agent in population.agents])


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


cdef class Population:
    """Ordered, mutable sequence of agents, ranked by the bound problem's ``sense``.

    ``size()`` is the configured number of agents (``pop_size``); ``len()`` is
    how many it holds right now. Slices, ``+``, ``sort()`` and ``greedy()`` give
    new populations of the same class, sharing its problem, generator and state.
    ``fitness``/``solutions`` are arrays built from the agents; assigning
    ``solutions`` writes each row back to its agent.
    """

    def __init__(self, size=0, agents=()):
        self._size = size
        # A list is wrapped, not copied, so code holding the list sees the same agents.
        self.agents = agents if type(agents) is list else list(agents)

    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def bind(self, problem, generator):
        """Attach the problem and the seeded generator, and drop the agents of a previous run."""
        self.problem = problem
        self.generator = generator
        self.agents = []
        self.idx = None

    cdef void copy_state(self, Population new):
        """Give ``new`` this population's state; subclasses with fields call it and copy theirs."""
        new.problem, new.generator, new._size = self.problem, self.generator, self._size

    cpdef Population spawn(self, object agents):
        """A population of the same class and state holding ``agents`` (a list is wrapped, not copied)."""
        cdef Population new = <Population>type(self).__new__(type(self))
        self.copy_state(new)
        new.agents = agents if type(agents) is list else list(agents)
        return new

    cpdef Py_ssize_t size(self):
        """The configured number of agents (``pop_size``)."""
        return self._size

    cpdef list toarray(self):
        """The agents, as the live list (a typed ``for`` over it is a C loop)."""
        return self.agents

    def resize(self, n):
        self._size = n

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
        agent.evaluate(self.problem)
        return agent

    def generate(self, n=None, starting=None):
        """A population of ``n`` (default ``size()``) generated agents, or one per ``starting`` solution."""
        if starting is not None:
            return self.spawn([self.generate_agent(x) for x in starting])
        return self.spawn([self.generate_agent() for _ in range(self._size if n is None else n)])

    def evaluate_solution(self, solution):
        """An evaluated plain :class:`Agent` at ``solution`` (a candidate; no random draws)."""
        return Agent(solution).evaluate(self.problem)

    cpdef object evaluate(self, object agents, object mode):
        """Evaluate ``agents`` (the batch step) and return them.

        ``"sequential"`` evaluates only the agents not evaluated yet (an algorithm
        that evaluates one by one has done it already); ``"swarm"``/``"parallel"``
        evaluate every agent. The batch is one ``problem.evaluate(X)`` call when the
        problem is vectorized, or in ``"parallel"`` mode has a nogil evaluator
        (OpenMP threads); otherwise one call per agent.
        """
        cdef Agent agent
        cdef Py_ssize_t i
        cdef bint parallel = mode == "parallel" and self.problem.evaluator is not None and self.problem.n_objs == 1
        pending = [agent for agent in agents if agent.objectives is None] if mode == "sequential" else agents
        if self.problem.vectorized or parallel:
            if len(pending):
                _, O = self.problem.evaluate(np.array([agent.solution for agent in pending], dtype=np.float64), parallel)
                weights = self.problem.obj_weights
                for i in range(len(pending)):
                    agent = pending[i]
                    agent.set_evaluation(O[i], weights)
        else:
            for agent in pending:
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

    def copy(self):
        """Shallow copy: the same agent objects."""
        return self.spawn(list(self.agents))

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
        return argsort_agents(self.agents, self.sense)

    cpdef Population sort(self):
        """New population, best first; ``.idx`` holds the source positions."""
        order = self.argsort()
        ranked = self.spawn([self.agents[i] for i in order])
        ranked.idx = order
        return ranked

    @property
    def best(self):
        fitness = self.fitness
        return self.agents[int(np.argmax(fitness) if self.sense == "max" else np.argmin(fitness))]

    @property
    def worst(self):
        fitness = self.fitness
        return self.agents[int(np.argmin(fitness) if self.sense == "max" else np.argmax(fitness))]

    cpdef Population greedy(self, object candidates, str mode="swarm"):
        """Per position, the candidate when better, else the current agent (``greedy_agents``, same ``mode`` rule)."""
        return self.spawn(greedy_agents(self.agents, candidates, self.sense, mode))


MutableSequence.register(Population)
