"""The optimization problem: an objective over a :class:`Bounds` search space."""
import numbers

import numpy as np
from cython.parallel cimport prange

from clypto.optimizer.bounds import Bounds
from clypto.optimizer.native.nogil cimport _NogilEvaluator

cdef tuple SUPPORTED_ARRAYS = (list, tuple, np.ndarray)


cdef class Problem:
    """``Problem(bounds, sense="min", obj_func=f)``.

    ``bounds`` is a :class:`Bounds`, one block or a list of blocks.
    ``vectorized=True``: ``obj_func`` takes an ``(n, n_dims)`` matrix and returns
    ``(n,)`` or ``(n, n_objs)``. ``evaluator`` is an optional nogil batch
    evaluator used by ``evaluate(X, parallel=True)``. ``obj_weights`` turn
    several objectives into one fitness (default: all ones).
    """

    def __init__(self, bounds, sense="min", obj_func=None, name="Problem",
                 evaluator=None, obj_weights=None, vectorized=False, seed=None):
        if sense not in ("min", "max"):
            raise ValueError(f'sense must be "min" or "max", got {sense!r}.')
        self.bounds = bounds if isinstance(bounds, Bounds) else Bounds(bounds)
        self.sense = sense
        self._obj_func = (lambda _: 0) if obj_func is None else obj_func
        self.name = name
        self.evaluator = evaluator
        self._obj_weights = obj_weights
        self.vectorized = vectorized
        self._n_objs = -1
        self.seed = seed

    @staticmethod
    def coerce(problem, seed=None):
        """``problem`` (a ``Problem`` or the dict of its arguments) seeded with ``seed``."""
        if type(problem) is dict:
            return Problem(**{**problem, "seed": seed})
        if not isinstance(problem, Problem):
            raise ValueError("problem needs to be a dict or an instance of Problem class.")
        problem.seed = seed
        return problem

    @property
    def n_dims(self):
        return self.bounds.n_dims

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.bounds.seed = seed

    @property
    def obj_weights(self):
        """The objective weights (materialized by the first ``n_objs`` access)."""
        if self._n_objs < 0:
            _ = self.n_objs
        return self._obj_weights

    @property
    def n_objs(self):
        """Number of objectives; the first access evaluates one random solution."""
        if self._n_objs < 0:
            result = self.obj_func(self.generate_solution(True))
            if isinstance(result, SUPPORTED_ARRAYS):
                self._n_objs = len(np.asarray(result).ravel())
            elif isinstance(result, numbers.Number):
                self._n_objs = 1
            else:
                raise ValueError("`obj_func` must return a number, list, tuple or numpy array.")
            if self._obj_weights is None:
                self._obj_weights = np.ones(self._n_objs)
            elif len(np.array(self._obj_weights).ravel()) != self._n_objs:
                raise ValueError(
                    f"`obj_weights` length {len(self._obj_weights)} does not match "
                    f"number of objectives {self._n_objs}."
                )
        return self._n_objs

    cpdef object obj_func(self, object x):
        if self.vectorized:
            return self._obj_func(np.asarray(x)[None, :])[0]
        return self._obj_func(x)

    def encode_solution(self, values):
        return self.bounds.encode(values)

    def decode_solution(self, x):
        return self.bounds.decode(x)

    cpdef object correct_solution(self, object x):
        """A valid solution, or ``(n, n_dims)`` matrix of them."""
        return self.bounds.correct(x)

    cpdef object generate_solution(self, bint encoded=True):
        if encoded:
            return self.bounds.generate()
        return [block.generate() for block in self.bounds.blocks]

    cpdef NativeTarget get_target(self, object solution):
        return NativeTarget(self.obj_func(solution), self._obj_weights)

    cpdef object fitness(self, object objectives):
        """Fitness of each ``(k, n_objs)`` objective row, computed like a ``Target``."""
        cdef Py_ssize_t i, k = objectives.shape[0], m = objectives.shape[1]
        w = self._obj_weights
        fw = (1.0,) * m if w is None else np.array(w).flatten()
        if len(fw) != m:
            fw = (1.0,) * m
        if m == 1:
            return objectives[:, 0] * fw[0]
        return np.array([np.dot(fw, objectives[i]) for i in range(k)])

    cpdef tuple evaluate(self, object X, bint parallel=False):
        """``(fitness, objectives)`` of every row of ``X``: ``(k,)`` and ``(k, n_objs)``.

        ``parallel=True`` with a nogil ``evaluator`` (single objective) runs the rows
        on OpenMP threads; otherwise ``obj_func`` is called once per row, or once
        for the whole matrix when the problem is ``vectorized``.
        """
        cdef _NogilEvaluator evaluator
        cdef double[:, ::1] Xv
        cdef double[:, ::1] Ov
        cdef Py_ssize_t i, k = X.shape[0], d
        if parallel and self.evaluator is not None and self.n_objs == 1:
            evaluator = <_NogilEvaluator>self.evaluator
            Xv = np.ascontiguousarray(X, dtype=np.float64)
            O = np.empty((k, 1))
            Ov = O
            d = Xv.shape[1]
            with nogil:
                for i in prange(k, schedule="static"):
                    evaluator.row(&Xv[i, 0], d, &Ov[i, 0])
        elif self.vectorized:
            O = np.asarray(self._obj_func(X), dtype=float).reshape(k, -1)
        else:
            O = np.array([np.array(self.obj_func(X[i])).flatten() for i in range(k)], dtype=float)
        return self.fitness(O), O
