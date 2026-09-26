"""Vectorized engine: the population is a :class:`NativePopulation` buffer.

``VectorizeOptimizer`` keeps one buffer row per agent (``pop.X``, ``pop.O``,
``pop.F`` plus algorithm fields) instead of a list of agent objects, so an epoch
is a handful of matrix operations. It has no instance ``__dict__``: every
attribute is a typed field, and subclasses declare their own.

Faithfulness to the legacy engine is part of the contract:

* After ``_after_initialization`` ``g_best`` is a *copy* of the best agent; from
  the first epoch on the legacy engine made it an *alias* of the best agent in
  ``pop``, so an update of that agent during ``_evolve`` is seen by the agents
  that come after it. ``_g_best_row`` records the aliased row (``-1`` = copy)
  and ``g_best_x()`` / ``current_g_best()`` read it live.
* Random numbers are drawn from ``self.generator`` in the legacy order, so
  vectorized algorithms draw whole blocks up front (``random((n, k, d))``).
"""
import numpy as np

cdef tuple PARALLEL_MODES = ("parallel", "thread", "process")


cdef class VectorizeOptimizer(NativeOptimizer):
    def __init__(self, parameters=(), sort_flag=False, name=None, mode=None):
        NativeOptimizer.__init__(self, parameters, sort_flag, name, mode)
        self._g_best_row = -1
        self._starting = None
        self.pop = None
        self.g_best = LegacyNativeAgent()
        self.g_worst = None
        self.problem = None

    # -- lifecycle hooks -------------------------------------------------------
    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        """Extra per-agent fields as ``[(name, width), ...]``."""
        return []

    cdef void init_fields(self, NativePopulation pop):
        """Fill the extra fields of freshly evaluated rows (``generate_agent``)."""
        pass

    def _check_problem(self, problem, seed):
        self.problem = Problem.coerce(problem, seed)
        self.pop, self.g_best, self.g_worst = None, None, None
        self._starting = None
        self._g_best_row = -1

    def _before_initialization(self, starting_solutions=None):
        if starting_solutions is None:
            return
        if not (type(starting_solutions) in self.SUPPORTED_ARRAYS and len(starting_solutions) == self.pop_size):
            raise ValueError(
                "Invalid starting_solutions. It should be a list/2D matrix of positions with same length as pop_size."
            )
        if not (type(starting_solutions[0]) in self.SUPPORTED_ARRAYS and len(starting_solutions[0]) == self.problem.n_dims):
            raise ValueError(
                "Invalid starting_solutions. It should be a list of positions or 2D matrix of positions only."
            )
        self._starting = np.array(starting_solutions, dtype=float)

    def _initialization(self):
        if self._starting is not None:
            self.pop = self.new_population(self._starting)
        else:
            self.pop = self.generate_population(self.pop_size)

    def _after_initialization(self):
        # The initial population is sorted or not depending on the algorithm.
        order = self.sorted_order(self.pop)
        self.g_best = self.pop.agent(order[0])
        self.g_worst = self.pop.agent(order[len(order) - 1])
        self._g_best_row = -1  # a copy, not an alias, until the first epoch
        if self.sort_flag:
            self.pop = self.pop.take(order)

    def _after_evolve(self):
        # g_best becomes an alias of the best row (see module docstring);
        # rows are only reordered when the algorithm asks for it.
        cdef Py_ssize_t b = 0
        order = self.sorted_order(self.pop)
        if self.sort_flag:
            self.pop = self.pop.take(order)
        else:
            b = order[0]
        self.pop.weights = self.problem._obj_weights
        self.g_best = self.pop.agent(b)
        self._g_best_row = b

    cdef object _amend_solution(self, object solution):
        return np.clip(solution, self.problem.bounds.low, self.problem.bounds.up)

    cpdef object _correct_solution(self, object solution):
        """``_amend_solution`` then the problem's own correction (rows or a matrix)."""
        return self.problem.correct_solution(self._amend_solution(solution))

    # -- population ----------------------------------------------------------
    cdef NativePopulation generate_population(self, Py_ssize_t k):
        X = np.array([self.problem.generate_solution(True) for _ in range(k)], dtype=float)
        return self.new_population(X)

    cdef NativePopulation new_population(self, object X):
        """Evaluate ``X`` (k, d) into a new population and fill its fields."""
        F, O = self.problem.evaluate(X)
        self._nfe_counter += X.shape[0]
        cdef Py_ssize_t d = X.shape[1], m = O.shape[1]
        cdef NativePopulation pop = NativePopulation(
            X.shape[0], d, m, self.layout(d, m), self.problem._obj_weights
        )
        pop.X[:] = X
        pop.O[:] = O
        pop.F[:] = F
        self.init_fields(pop)
        return pop

    cdef void evaluate(self, NativePopulation pop, Py_ssize_t start, Py_ssize_t stop):
        """Evaluate rows ``[start, stop)`` of ``pop.X`` into ``pop.O``/``pop.F``.

        ``mode="parallel"`` lets ``Problem.evaluate`` run a nogil evaluator on
        OpenMP threads.
        """
        if stop <= start:
            return
        F, O = self.problem.evaluate(pop.X[start:stop], self.mode in PARALLEL_MODES)
        pop.O[start:stop] = O
        pop.F[start:stop] = F
        self._nfe_counter += stop - start

    cdef object sorted_order(self, NativePopulation pop):
        """Row order best-first, identical to the legacy ``_get_sorted_population``."""
        order = np.argsort(np.ascontiguousarray(pop.F))
        return order[::-1] if self.problem.sense == "max" else order

    cdef object g_best_x(self):
        """Global best position as the legacy engine saw it at this point."""
        if self._g_best_row >= 0:
            return self.pop.X[self._g_best_row]
        return self.g_best.solution

    cdef LegacyNativeAgent current_g_best(self):
        if self._g_best_row >= 0:
            return self.pop.agent(self._g_best_row)
        return self.g_best
