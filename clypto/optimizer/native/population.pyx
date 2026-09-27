"""Populations: ``Population`` (a sequence of agents) and ``NativePopulation``.

``Population`` is ``self.population`` in the legacy engine and the decorator API.
It is declared in the optimizer's ``__init__`` with ``cy.population(pop_size)``,
bound to the problem by ``solve()`` (``bind``), and owns the agents' life cycle:
``create_agent``/``generate_agent`` build them, ``amend_solution``/``correct_solution``
keep them inside the bounds and ``evaluate_solution``/``evaluate`` evaluate them.
An algorithm with its own agents subclasses it and overrides those methods.

``NativePopulation`` is the vectorize engine's buffer: every agent is one row of
a single C-contiguous ``float64`` buffer::

    [ F | O (m) | X (d) | algorithm fields ... ]

Algorithm fields are declared as ``(name, width)`` pairs (e.g. PSO's velocity,
personal best and its objectives/fitness) and live in the same row, so copying
an agent is copying a row and sorting is one fancy-index.
"""
from collections.abc import MutableSequence

import numpy as np

from cython.parallel cimport prange

from clypto.optimizer.native.agent cimport Agent, argsort_agents, fields_of
from clypto.optimizer.native.utils cimport validator

cdef tuple PARALLEL_MODES = ("parallel", "thread", "process")


cpdef Population population(object size, object range=[5, 10000], type cls=None):
    """An unbound population of ``size`` agents (``cls``, default :class:`Population`).

    ``size`` is validated like ``cy.validator(int, size, range, "pop_size")``.
    """
    return (Population if cls is None else cls)(validator(int, size, range, "pop_size"))


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

    cpdef Population spawn(self, object agents):
        """A population of the same class and state holding ``agents`` (a list is wrapped, not copied)."""
        cls = type(self)
        cdef Population new = cls.__new__(cls)
        for name in fields_of(cls, Population):
            setattr(new, name, getattr(self, name))
        if hasattr(self, "__dict__"):
            new.__dict__.update(self.__dict__)
        new.problem, new.generator, new._size = self.problem, self.generator, self._size
        new.agents = agents if type(agents) is list else list(agents)
        return new

    def size(self):
        """The configured number of agents (``pop_size``)."""
        return self._size

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

    def amend_solution(self, solution):
        """Bring a moved solution back inside the bounds (default: clip)."""
        return np.clip(solution, self.problem.bounds.low, self.problem.bounds.up)

    def correct_solution(self, solution):
        """``amend_solution`` then the problem's own correction."""
        return self.problem.correct_solution(self.amend_solution(solution))

    def evaluate_solution(self, solution):
        """An evaluated plain :class:`Agent` at ``solution`` (a candidate; no random draws)."""
        return Agent(solution).evaluate(self.problem)

    def evaluate(self, agents, mode):
        """Evaluate every agent of ``agents`` in the batch modes (``swarm``, ``parallel``, ...).

        ``parallel``/``thread``/``process`` use the problem's nogil evaluator on
        OpenMP threads when it has one (single objective); sequential mode
        (``mode=None``) evaluates nothing, the agents were evaluated one by one.
        """
        cdef Py_ssize_t i, n = len(agents)
        cdef Agent agent
        if mode == "swarm" or (
            mode in PARALLEL_MODES and (self.problem.evaluator is None or self.problem.n_objs != 1)
        ):
            for agent in agents:
                agent.evaluate(self.problem)
        elif mode in PARALLEL_MODES:
            _, O = self.problem.evaluate(np.array([agent.solution for agent in agents], dtype=np.float64), True)
            weights = self.problem.obj_weights
            for i in range(n):
                agent = agents[i]
                agent.set_evaluation(O[i], weights)
        return agents

    def opposite_solution(self, agent, g_best):
        """Opposition-based candidate of ``agent`` around ``g_best`` (bounded)."""
        pos_new = (
            self.problem.bounds.low + self.problem.bounds.up - g_best.solution
            + self.generator.uniform() * (g_best.solution - agent.solution)
        )
        return self.correct_solution(pos_new)

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

    def append(self, agent):
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

    def duplicate(self):
        """Deep copy: a copy of every agent."""
        return self.spawn([agent.copy() for agent in self.agents])

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

    def sort(self):
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

    def greedy(self, candidates):
        """Per position, the candidate when strictly better, else the current agent."""
        if len(candidates) != len(self.agents):
            raise ValueError("Greedy selection of two population with different length.")
        if self.sense == "max":
            agents = [new if new.fitness > old.fitness else old for old, new in zip(self.agents, candidates)]
        else:
            agents = [new if new.fitness < old.fitness else old for old, new in zip(self.agents, candidates)]
        return self.spawn(agents)


MutableSequence.register(Population)


cdef class ResetPopulation(Population):
    """A solution that leaves the bounds is redrawn uniformly at random, instead of clipped."""

    def amend_solution(self, solution):
        low, up = self.problem.bounds.low, self.problem.bounds.up
        return np.where((low <= solution) & (solution <= up), solution, self.generator.uniform(low, up))

# Rows x columns below which OpenMP threads cost more than they save.
cdef Py_ssize_t PARALLEL_MIN_WORK = 20000


cdef class NativePopulation:
    def __init__(self, Py_ssize_t n, Py_ssize_t d, Py_ssize_t m, fields=(), weights=None):
        cdef Py_ssize_t col = 1 + m + d
        self.n, self.d, self.m = n, d, m
        self.cF, self.cO, self.cX = 0, 1, 1 + m
        self._fields = {}
        for name, size in fields:
            self._fields[name] = (col, size)
            col += size
        self.width = col
        self.buf = np.full((n, col), np.nan)
        self.view = self.buf
        self.weights = weights

    cpdef Py_ssize_t offset(self, str name):
        return self._fields[name][0]

    def field(self, str name):
        """``(n, width)`` view of a declared field."""
        off, size = self._fields[name]
        return self.buf[:, off:off + size]

    @property
    def X(self):
        return self.buf[:, self.cX:self.cX + self.d]

    @property
    def O(self):
        return self.buf[:, self.cO:self.cO + self.m]

    @property
    def F(self):
        return self.buf[:, self.cF]

    cpdef Agent agent(self, Py_ssize_t i, Py_ssize_t c_obj=-1):
        """Snapshot of row ``i`` as a standalone agent (objectives from column ``c_obj``, default ``cO``)."""
        if c_obj < 0:
            c_obj = self.cO
        return Agent(self.buf[i, self.cX:self.cX + self.d].copy(), self.buf[i, c_obj:c_obj + self.m].copy(), self.weights)

    cpdef NativePopulation take(self, object rows):
        """New population holding copies of ``rows`` (in that order)."""
        cdef NativePopulation out = self.empty_like()
        out.buf = np.ascontiguousarray(self.buf[rows])
        out.view = out.buf
        out.n = out.buf.shape[0]
        return out

    cpdef NativePopulation concat(self, NativePopulation other):
        """Rows of ``self`` followed by the rows of ``other`` (same layout)."""
        cdef NativePopulation out = self.empty_like()
        out.buf = np.concatenate([self.buf, other.buf])
        out.view = out.buf
        out.n = out.buf.shape[0]
        return out

    cpdef NativePopulation empty_like(self):
        cdef NativePopulation out = NativePopulation.__new__(NativePopulation)
        out.n, out.d, out.m, out.width = self.n, self.d, self.m, self.width
        out.cF, out.cO, out.cX = self.cF, self.cO, self.cX
        out._fields = self._fields
        out.weights = self.weights
        out.buf = np.full((self.n, self.width), np.nan)
        out.view = out.buf
        return out

    def __len__(self):
        return self.n

    def __getitem__(self, Py_ssize_t i):
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.agent(i)

    def __iter__(self):
        return (self.agent(i) for i in range(self.n))


cdef void select_better(
    NativePopulation dst, Py_ssize_t dst_x, Py_ssize_t dst_o, Py_ssize_t dst_f,
    NativePopulation src, Py_ssize_t src_x, Py_ssize_t src_o, Py_ssize_t src_f,
    Py_ssize_t start, Py_ssize_t stop, bint maximize,
) noexcept nogil:
    """Copy src rows into dst where src is better (``cy.is_better`` semantics).

    min: ``new < old``; max: ``not (new < old)``. Rows are independent, so the
    result does not depend on the number of threads.
    """
    cdef double[:, ::1] a = dst.view
    cdef double[:, ::1] b = src.view
    cdef Py_ssize_t i, j, d = dst.d, m = dst.m
    cdef bint better
    for i in prange(start, stop, schedule="static",
                    use_threads_if=(stop - start) * (d + m) > PARALLEL_MIN_WORK):
        better = b[i, src_f] < a[i, dst_f]
        if maximize:
            better = not better
        if better:
            for j in range(d):
                a[i, dst_x + j] = b[i, src_x + j]
            for j in range(m):
                a[i, dst_o + j] = b[i, src_o + j]
            a[i, dst_f] = b[i, src_f]
