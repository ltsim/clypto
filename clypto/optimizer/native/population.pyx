"""Populations: ``Population`` (a sequence of agent objects) and ``NativePopulation``.

``Population`` is the legacy engine's ``self.pop`` and the decorator API's
``self.population``. ``NativePopulation`` is the vectorize engine's buffer:
every agent is one row of a single C-contiguous ``float64`` buffer::

    [ F | O (m) | X (d) | algorithm fields ... ]

Algorithm fields are declared as ``(name, width)`` pairs (e.g. PSO's velocity,
personal best and its objectives/fitness) and live in the same row, so copying
an agent is copying a row and sorting is one fancy-index.
"""
from collections.abc import MutableSequence
from typing import Generic, TypeVar

import numpy as np

from cython.parallel cimport prange

AgentT = TypeVar("AgentT")


class Population(MutableSequence, Generic[AgentT]):
    """Ordered, mutable sequence of agents ranked by ``sense`` (``"min"``/``"max"``).

    ``Population[MyAgent](agents, sense="max")``; slices and ``+`` give new
    populations. ``fitness``/``solutions`` are arrays built from the agents;
    assigning ``solutions`` writes each row back to its agent.
    """

    def __init__(self, agents=(), sense="min"):
        # A list is wrapped, not copied, so code holding the list sees the same agents.
        self._agents = agents if type(agents) is list else list(agents)
        self.sense = sense
        self.idx = None  # set by sort(): positions of the ranked agents in the source

    # -- sequence protocol -------------------------------------------------------
    def __len__(self):
        return len(self._agents)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Population(self._agents[index], self.sense)
        return self._agents[index]

    def __setitem__(self, index, value):
        self._agents[index] = value

    def __delitem__(self, index):
        del self._agents[index]

    def insert(self, index, agent):
        self._agents.insert(index, agent)

    def __iter__(self):
        return iter(self._agents)

    def __add__(self, other):
        return Population(self._agents + list(other), self.sense)

    def __radd__(self, other):
        return Population(list(other) + self._agents, self.sense)

    def __repr__(self):
        return f"Population(sense={self.sense!r}, {self._agents!r})"

    def popleft(self):
        return self._agents.pop(0)

    def copy(self):
        """Shallow copy: the same agent objects."""
        return Population(list(self._agents), self.sense)

    def duplicate(self):
        """Deep copy: a copy of every agent."""
        return Population([agent.copy() for agent in self._agents], self.sense)

    # -- arrays ------------------------------------------------------------------
    @property
    def fitness(self):
        """``(n,)`` fitness of every agent."""
        return np.array([agent.target.fitness for agent in self._agents], dtype=float)

    @property
    def solutions(self):
        """``(n, n_dims)`` solution matrix."""
        return np.array([agent.solution for agent in self._agents], dtype=float)

    @solutions.setter
    def solutions(self, values):
        values = np.atleast_2d(np.asarray(values, dtype=float))
        if values.shape[0] != len(self._agents):
            raise ValueError(f"Expected {len(self._agents)} solutions, got {values.shape[0]}.")
        for agent, row in zip(self._agents, values):
            agent.solution = row

    # -- ranking -----------------------------------------------------------------
    def argsort(self):
        """Positions best first (``np.argsort`` of the fitness, reversed when maximizing)."""
        order = np.argsort(self.fitness).tolist()
        return order[::-1] if self.sense == "max" else order

    def sort(self):
        """New population, best first; ``.idx`` holds the source positions."""
        order = self.argsort()
        ranked = Population([self._agents[i] for i in order], self.sense)
        ranked.idx = order
        return ranked

    @property
    def best(self):
        fitness = self.fitness
        return self._agents[int(np.argmax(fitness) if self.sense == "max" else np.argmin(fitness))]

    @property
    def worst(self):
        fitness = self.fitness
        return self._agents[int(np.argmin(fitness) if self.sense == "max" else np.argmax(fitness))]

    def greedy(self, candidates):
        """Per position, the candidate when strictly better, else the current agent."""
        if len(candidates) != len(self._agents):
            raise ValueError("Greedy selection of two population with different length.")
        if self.sense == "max":
            agents = [new if new.target.fitness > old.target.fitness else old for old, new in zip(self._agents, candidates)]
        else:
            agents = [new if new.target.fitness < old.target.fitness else old for old, new in zip(self._agents, candidates)]
        return Population(agents, self.sense)

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

    cpdef NativeTarget target_at(self, Py_ssize_t i, Py_ssize_t c_obj):
        return NativeTarget(self.buf[i, c_obj:c_obj + self.m].copy(), self.weights)

    cpdef LegacyNativeAgent agent(self, Py_ssize_t i):
        """Snapshot of row ``i`` as a standalone agent."""
        return LegacyNativeAgent(
            self.buf[i, self.cX:self.cX + self.d].copy(), self.target_at(i, self.cO)
        )

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
    """Copy src rows into dst where src is better (``cy.compare_target`` semantics).

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
