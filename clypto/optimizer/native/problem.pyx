"""The optimization problem: an objective over a :class:`Bounds` search space."""
import numbers
import warnings

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

    ``n_evals`` counts the evaluations (``evaluate_solution`` and ``evaluate``);
    ``solve()`` resets it, and ``optimizer.nf_counter`` reads it.
    """

    def __init__(self, bounds=None, sense="min", obj_func=None, name="Problem",
                 evaluator=None, obj_weights=None, vectorized=False, seed=None,
                 decode=False, **kwargs):
        if "vectorize" in kwargs:
            vectorized = kwargs.pop("vectorize")
        if "evaluate" in kwargs and obj_func is None:
            obj_func = kwargs.pop("evaluate")

        # Allow target callable as first positional: Problem(f, bounds=...)
        if callable(bounds) and obj_func is None:
            obj_func = bounds
            bounds = kwargs.pop("bounds", None)

        if type(self) is not Problem:
            if bounds is None and "bounds" in type(self).__dict__:
                bounds = type(self).__dict__["bounds"]
            if "sense" not in kwargs and "sense" in type(self).__dict__:
                sense = type(self).__dict__["sense"]
            if not decode and "decode" in type(self).__dict__:
                decode = type(self).__dict__["decode"]
            if not vectorized:
                if "vectorize" in type(self).__dict__:
                    vectorized = type(self).__dict__["vectorize"]
                elif "vectorized" in type(self).__dict__:
                    vectorized = type(self).__dict__["vectorized"]

        if bounds is None:
            raise ValueError("`bounds` must be specified for Problem.")

        if sense not in ("min", "max"):
            raise ValueError(f'sense must be "min" or "max", got {sense!r}.')
        self.__bounds = bounds if isinstance(bounds, Bounds) else Bounds(bounds)
        self.__sense = sense

        if type(self) is not Problem:
            type(self).bounds = self.__bounds
            type(self).sense = self.__sense
            type(self).decode = bool(decode)
            type(self).vectorized = bool(vectorized)

        if obj_func is None:
            if "evaluate" in type(self).__dict__:
                obj_func = getattr(self, "evaluate")
            elif "obj_func" in type(self).__dict__:
                obj_func = getattr(self, "obj_func")
            else:
                obj_func = lambda _: 0

        self.__obj_func = obj_func
        self.__name = name
        self.__evaluator = evaluator
        self.__obj_weights = obj_weights
        self.__vectorized = vectorized
        self.__decode = decode
        self.__n_objs = -1
        self.__n_evals = 0
        self.__seed = seed
        self.__bounds.seed = seed

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
    def bounds(self):
        return self.__bounds

    @property
    def sense(self):
        return self.__sense

    @property
    def name(self):
        return self.__name

    @property
    def evaluator(self):
        return self.__evaluator

    @property
    def vectorized(self):
        return self.__vectorized

    @property
    def decode(self):
        return self.__decode

    @property
    def n_evals(self):
        return self.__n_evals

    @n_evals.setter
    def n_evals(self, long long value):
        self.__n_evals = value

    @property
    def n_dims(self):
        return self.__bounds.n_dims

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = seed
        self.__bounds.seed = seed

    @property
    def obj_weights(self):
        """The objective weights (materialized by the first ``n_objs`` access)."""
        if self.__n_objs < 0:
            _ = self.n_objs
        return self.__obj_weights

    @property
    def n_objs(self):
        """Number of objectives; the first access evaluates one random solution."""
        cdef object test_sol
        cdef object test_mat
        cdef object target
        cdef object result
        if self.__n_objs < 0:
            test_sol = self.generate_solution(True)
            if self.__vectorized:
                test_mat = np.atleast_2d(test_sol)
                target = self.decode_solution(test_mat) if self.__decode else test_mat
                result = self.__obj_func(target)
                if isinstance(result, SUPPORTED_ARRAYS) and len(result) > 0:
                    result = result[0]
            else:
                target = self.decode_solution(test_sol) if self.__decode else test_sol
                result = self.obj_func(target)
            if isinstance(result, SUPPORTED_ARRAYS):
                self.__n_objs = len(np.asarray(result).ravel())
            elif isinstance(result, numbers.Number):
                self.__n_objs = 1
            else:
                raise ValueError("`obj_func` must return a number, list, tuple or numpy array.")
            if self.__obj_weights is None:
                self.__obj_weights = np.ones(self.__n_objs)
            elif len(np.array(self.__obj_weights).ravel()) != self.__n_objs:
                raise ValueError(
                    f"`obj_weights` length {len(self.__obj_weights)} does not match "
                    f"number of objectives {self.__n_objs}."
                )
        return self.__n_objs

    cpdef object obj_func(self, object x):
        if self.__vectorized:
            return self.__obj_func(np.atleast_2d(x))[0]
        return self.__obj_func(x)

    def encode_solution(self, values):
        return self.__bounds.encode(values)

    def decode_solution(self, x):
        return self.__bounds.decode(x)

    cpdef object correct_solution(self, object x):
        """A valid solution, or ``(n, n_dims)`` matrix of them."""
        return self.__bounds.correct(x)

    cpdef object generate_solution(self, bint encoded=True):
        if encoded:
            return self.__bounds.generate()
        return [block.generate() for block in self.__bounds.blocks]

    cpdef object evaluate_solution(self, object solution):
        """The objectives of one solution (counted)."""
        self.__n_evals += 1
        cdef object target
        if self.__vectorized:
            if self.__decode:
                target = self.decode_solution(np.atleast_2d(solution))
            else:
                target = np.atleast_2d(solution)
            return self.__obj_func(target)[0]
        target = self.decode_solution(solution) if self.__decode else solution
        return self.obj_func(target)

    def __call__(self, object solution):
        return self.evaluate_solution(solution)

    cpdef object fitness(self, object objectives):
        """Fitness of each ``(k, n_objs)`` objective row, computed like ``Agent.fitness``."""
        cdef Py_ssize_t i, k = objectives.shape[0], m = objectives.shape[1]
        w = self.__obj_weights
        fw = (1.0,) * m if w is None else np.array(w).flatten()
        if len(fw) != m:
            fw = (1.0,) * m
        if m == 1:
            return objectives[:, 0] * fw[0]
        return np.array([np.dot(fw, objectives[i]) for i in range(k)])

    cpdef tuple evaluate(self, object X, bint parallel=False, int threads=0):
        """``(fitness, objectives)`` of every row of ``X``: ``(k,)`` and ``(k, n_objs)``.

        ``parallel=True`` with a nogil ``evaluator`` runs the rows on the OpenMP team
        fixed by the engine's ``solve`` (``threads`` 0 lets OpenMP decide); otherwise
        ``obj_func`` is called once per row, or once for the whole matrix when the
        problem is ``vectorized``.
        """
        cdef _NogilEvaluator evaluator
        cdef double[:, ::1] Xv
        cdef double[:, ::1] Ov
        cdef Py_ssize_t i, k = X.shape[0], d, m = self.n_objs
        if threads < 0:
            raise ValueError("threads must be >= 0.")
        if parallel and self.__evaluator is not None:
            evaluator = <_NogilEvaluator>self.__evaluator
            if evaluator._n_objs != m:
                raise ValueError(
                    f"`evaluator` produces {evaluator._n_objs} objectives, but the problem has {m}."
                )
            Xv = np.ascontiguousarray(X, dtype=np.float64)
            O = np.empty((k, m), dtype=np.float64)
            Ov = O
            d = Xv.shape[1]
            with nogil:
                for i in prange(k, schedule="static"):
                    evaluator.row(&Xv[i, 0], d, &Ov[i, 0])
        elif self.__vectorized:
            if self.__decode:
                O = np.asarray(self.__obj_func(self.decode_solution(X)), dtype=np.float64).reshape(k, -1)
            else:
                O = np.asarray(self.__obj_func(X), dtype=np.float64).reshape(k, -1)
        else:
            if parallel:
                warnings.warn(
                    "mode='parallel' requested without a nogil evaluator; falling back to sequential.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.__decode:
                O = np.array([np.array(self.obj_func(self.decode_solution(X[i]))).flatten() for i in range(k)], dtype=np.float64)
            else:
                O = np.array([np.array(self.obj_func(X[i])).flatten() for i in range(k)], dtype=np.float64)
        self.__n_evals += k
        return self.fitness(O), O
