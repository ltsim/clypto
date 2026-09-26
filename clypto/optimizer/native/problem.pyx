"""Native ``Problem`` used by the Cython collection (see ``clypto.optimizer.problem``)."""
import numbers

import numpy as np

from clypto.optimizer.problem import Problem
from clypto.optimizer.space import BaseVar, FloatVar

cdef tuple SUPPORTED_ARRAYS = (list, tuple, np.ndarray)


cdef class NativeProblem:
    def __init__(self, bounds, minmax="min", obj_func=None, name="Problem",
                 evaluator=None, obj_weights=None, vectorized=False):
        self._obj_func = (lambda _: 0) if obj_func is None else obj_func
        self._name = name
        self.evaluator = evaluator
        self.vectorized = vectorized
        self.minmax = minmax
        self._seed = None
        self._obj_weights = obj_weights
        self._n_objs = -1
        self.set_bounds(bounds)

    @staticmethod
    def coerce(problem):
        """Return ``problem`` as a ``NativeProblem`` (accepts a dict or a ``Problem``)."""
        if isinstance(problem, NativeProblem):
            return problem
        if isinstance(problem, Problem):
            # Bind the instance's obj_func so a Problem subclass override still runs;
            # a vectorized problem needs the raw batch function instead of the
            # single-row wrapper that Problem.obj_func applies.
            obj_func = problem._Problem__obj_func if problem.vectorized else problem.obj_func
            return NativeProblem(
                problem.bounds, problem.minmax, obj_func,
                problem.get_name(), problem.evaluator, None, problem.vectorized,
            )
        if type(problem) is dict:
            # Like Problem(**dict): unknown keys (e.g. "seed") are ignored.
            return NativeProblem(
                problem["bounds"], problem.get("minmax", "min"), problem.get("obj_func"),
                problem.get("name", "Problem"), problem.get("evaluator"),
                None, problem.get("vectorized", False),
            )
        raise ValueError("problem needs to be a dict or an instance of Problem class.")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        for var in self.bounds:
            var.seed = seed

    @property
    def obj_weights(self):
        if self._n_objs < 0:
            _ = self.n_objs
        return self._obj_weights

    @property
    def n_objs(self):
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

    def set_bounds(self, bounds):
        if isinstance(bounds, BaseVar):
            bounds = [bounds]
        elif type(bounds) not in SUPPORTED_ARRAYS:
            raise TypeError(
                f"Invalid bounds. It should be type of {SUPPORTED_ARRAYS} or an instance of BaseVar"
            )
        for var in bounds:
            if not isinstance(var, BaseVar):
                raise ValueError("Invalid bounds. All variables in bounds should be a BaseVar.")
            var.seed = self._seed
        self.bounds = list(bounds)
        self.lb = np.concatenate([var.lb for var in self.bounds])
        self.ub = np.concatenate([var.ub for var in self.bounds])
        self.n_dims = len(self.lb)
        self._all_float = all(type(var) is FloatVar for var in self.bounds)

    def get_name(self):
        return self._name

    cpdef object obj_func(self, object x):
        if self.vectorized:
            return self._obj_func(np.asarray(x)[None, :])[0]
        return self._obj_func(x)

    def encode_solution(self, x):
        return Problem.encode_solution_with_bounds(x, self.bounds)

    def decode_solution(self, x):
        return Problem.decode_solution_with_bounds(x, self.bounds)

    cpdef object correct_solution(self, object x):
        cdef list x_new = []
        cdef Py_ssize_t n_vars = 0
        for var in self.bounds:
            x_new += list(var.correct(x[n_vars : n_vars + var.n_vars]))
            n_vars += var.n_vars
        return np.array(x_new)

    cpdef object correct_solutions(self, object X):
        """``correct_solution`` for one row or, row by row, an ``(n, n_dims)`` matrix."""
        cdef Py_ssize_t i
        if self._all_float:
            # FloatVar.correct is np.clip, which is element-wise: same values as per row.
            return np.clip(X, self.lb, self.ub)
        if np.ndim(X) == 1:
            return self.correct_solution(X)
        return np.array([self.correct_solution(X[i]) for i in range(X.shape[0])])

    cpdef object generate_solution(self, bint encoded=True):
        x = [var.generate() for var in self.bounds]
        if encoded:
            return Problem.encode_solution_with_bounds(x, self.bounds)
        return x

    cpdef NativeTarget get_target(self, object solution):
        return NativeTarget(self.obj_func(solution), self._obj_weights)
