"""Native vectorized legacy engine for the Cython collection.

``LegacyNativeOptimizer`` keeps the population in a :class:`NativePopulation`
(one buffer row per agent) instead of a list of agent objects. It has no
instance ``__dict__``: every attribute is a typed field, and subclasses declare
their own. Lifecycle hooks are ``cdef`` (C vtable, not visible from Python).

Faithfulness to the classic engine (``legacy.pyx``) is part of the contract:

* After ``after_initialization`` ``g_best`` is a *copy* of the best agent; from
  the first epoch on the classic engine made it an *alias* of the best agent in
  ``pop``, so an update of that agent during ``evolve`` is seen by the agents
  that come after it. ``_g_best_row`` records the aliased row (``-1`` = copy)
  and ``g_best_x()`` / ``current_g_best()`` read it live.
* Random numbers are drawn from ``self.generator`` in the classic order, so
  vectorized algorithms draw whole blocks up front (``random((n, k, d))``).
"""
import math
import random
import time
import typing
from time import perf_counter

import numpy as np
from cython.parallel cimport prange

from clypto.optimizer.native.nogil cimport _NogilEvaluator
from clypto.optimizer.native.problem import NativeProblem as _NativeProblemPy
from clypto.optimizer.history import Tracker
from clypto.optimizer.problem import Problem
from clypto.optimizer.termination import Termination

cdef tuple PARALLEL_MODES = ("parallel", "thread", "process")


cdef class LegacyNativeOptimizer:
    def __init__(self, parameters=(), sort_flag=False, parallelizable=True,
                 name=None, mode=None):
        self.tracker = Tracker()
        self.EPSILON = 10e-10
        self.AVAILABLE_MODES = ("swarm", "parallel", "thread", "process")
        self.SUPPORTED_ARRAYS = list, tuple, np.ndarray
        self._last_gbest_fit = None
        self._repeated_times = 0
        self._nfe_counter = 1
        self._name = self.__class__.__name__ if name is None else name
        self._params_name_ordered = tuple(parameters)
        self._g_best_row = -1
        self._starting = None
        self.generator = None
        self._termination = None
        self.mode = mode
        self.epoch = 0
        self.pop_size = 0
        self.n_workers = None
        self.pop = None
        self.g_best = LegacyNativeAgent()
        self.g_worst = None
        self.problem = None
        self.sort_flag = sort_flag
        self.is_parallelizable = parallelizable
        self.rng = None

    @property
    def termination(self):
        return self._termination

    @property
    def name(self):
        return self._name

    @property
    def nf_counter(self):
        return self._nfe_counter

    @property
    def parameters(self) -> typing.Dict:
        """Current values of the parameters named in ``__init__(parameters=...)``."""
        return {name: getattr(self, name) for name in self._params_name_ordered}

    def set_parameters(self, parameters: typing.Dict) -> None:
        """Override parameter values, e.g. ``set_parameters({"w": 0.5})``."""
        invalid = set(parameters) - set(self._params_name_ordered)
        if invalid:
            raise ValueError(
                f"Invalid input parameters: {invalid} for {self.get_name()} optimizer. "
                f"Valid parameters are: {set(self._params_name_ordered)}."
            )
        for key, value in parameters.items():
            setattr(self, key, value)

    def get_parameters(self) -> typing.Dict:
        return self.parameters

    def get_attributes(self) -> typing.Dict:
        data = {}
        for key in (
            "epoch", "pop_size", "pop", "problem", "g_best", "g_worst",
            "mode", "sort_flag", "is_parallelizable", "parameters",
        ):
            data[key] = getattr(self, key, None)
        return data

    def get_name(self) -> str:
        return self.name

    def __str__(self):
        return f"{self.__class__.__name__}"

    # -- lifecycle hooks (override with the same cdef signature) -------------
    cdef void initialize_variables(self):
        pass

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        """Extra per-agent fields as ``[(name, width), ...]``."""
        return []

    cdef void init_fields(self, NativePopulation pop):
        """Fill the extra fields of freshly evaluated rows (``generate_agent``)."""
        pass

    cdef void before_initialization(self, object starting_solutions=None):
        if starting_solutions is None:
            return
        if (
            type(starting_solutions) in self.SUPPORTED_ARRAYS
            and len(starting_solutions) == self.pop_size
        ):
            if (
                type(starting_solutions[0]) in self.SUPPORTED_ARRAYS
                and len(starting_solutions[0]) == self.problem.n_dims
            ):
                self._starting = np.array(starting_solutions, dtype=float)
            else:
                raise ValueError(
                    "Invalid starting_solutions. It should be a list of positions or 2D matrix of positions only."
                )
        else:
            raise ValueError(
                "Invalid starting_solutions. It should be a list/2D matrix of positions with same length as pop_size."
            )

    cdef void initialization(self):
        if self._starting is not None:
            self.pop = self.new_population(self._starting)
        else:
            self.pop = self.generate_population(self.pop_size)

    cdef void after_initialization(self):
        # The initial population is sorted or not depending on the algorithm.
        order = self.sorted_order(self.pop)
        self.g_best = self.pop.agent(order[0])
        self.g_worst = self.pop.agent(order[len(order) - 1])
        self._g_best_row = -1  # a copy, not an alias, until the first epoch
        if self.sort_flag:
            self.pop = self.pop.take(order)

    cdef void before_main_loop(self):
        pass

    cdef void evolve(self, int epoch):
        pass

    cdef object amend_solution(self, object solution):
        return np.clip(solution, self.problem.lb, self.problem.ub)

    cpdef object correct_solution(self, object solution):
        """``amend_solution`` then the problem's own correction (rows or a matrix)."""
        return self.problem.correct_solutions(self.amend_solution(solution))

    # -- population ----------------------------------------------------------
    cdef NativePopulation generate_population(self, Py_ssize_t k):
        X = np.array([self.problem.generate_solution(True) for _ in range(k)], dtype=float)
        return self.new_population(X)

    cdef NativePopulation new_population(self, object X):
        """Evaluate ``X`` (k, d) into a new population and fill its fields."""
        R = self._objectives(X)
        cdef Py_ssize_t d = X.shape[1], m = R.shape[1]
        cdef NativePopulation pop = NativePopulation(
            X.shape[0], d, m, self.layout(d, m), self.problem._obj_weights
        )
        pop.X[:] = X
        pop.O[:] = R
        pop.F[:] = self._fitness(R)
        self.init_fields(pop)
        return pop

    cdef object _objectives(self, object X):
        """Counted evaluation of every row of ``X``; returns ``(k, n_objs)``."""
        cdef Py_ssize_t i, k = X.shape[0]
        if self.problem.vectorized:
            R = np.asarray(self.problem._obj_func(X), dtype=float).reshape(k, -1)
        else:
            R = np.array(
                [np.array(self.problem.obj_func(X[i])).flatten() for i in range(k)],
                dtype=float,
            )
        self._nfe_counter += k
        return R

    cdef object _fitness(self, object R):
        """Fitness of each objective row, computed exactly like ``NativeTarget``."""
        cdef Py_ssize_t i, k = R.shape[0], m = R.shape[1]
        w = self.problem._obj_weights
        fw = (1.0,) * m if w is None else np.array(w).flatten()
        if len(fw) != m:
            fw = (1.0,) * m
        if m == 1:
            return R[:, 0] * fw[0]
        return np.array([np.dot(fw, R[i]) for i in range(k)])

    cdef void evaluate(self, NativePopulation pop, Py_ssize_t start, Py_ssize_t stop):
        """Evaluate rows ``[start, stop)`` of ``pop.X`` into ``pop.O``/``pop.F``.

        ``mode="parallel"`` with a nogil ``Problem.evaluator`` runs the rows on
        OpenMP threads; otherwise ``obj_func`` is called (once per row, or once
        for the block when the problem is vectorized).
        """
        cdef _NogilEvaluator evaluator
        cdef double[:, ::1] buf = pop.view
        cdef Py_ssize_t i, cX = pop.cX, cO = pop.cO, cF = pop.cF, d = pop.d
        if stop <= start:
            return
        if (
            self.mode in PARALLEL_MODES
            and self.problem.evaluator is not None
            and self.problem.n_objs == 1
        ):
            evaluator = <_NogilEvaluator>self.problem.evaluator
            w0 = self._fitness(np.ones((1, 1)))[0]
            with nogil:
                for i in prange(start, stop, schedule="static"):
                    evaluator.row(&buf[i, cX], d, &buf[i, cO])
            pop.F[start:stop] = pop.buf[start:stop, cO] * w0
            self._nfe_counter += stop - start
            return
        R = self._objectives(pop.buf[start:stop, cX:cX + d])
        pop.O[start:stop] = R
        pop.F[start:stop] = self._fitness(R)

    cdef object sorted_order(self, NativePopulation pop):
        """Row order best-first, identical to the classic ``get_sorted_population``."""
        order = np.argsort(np.ascontiguousarray(pop.F))
        return order[::-1] if self.problem.minmax == "max" else order

    cdef object g_best_x(self):
        """Global best position as the classic engine saw it at this point."""
        if self._g_best_row >= 0:
            return self.pop.X[self._g_best_row]
        return self.g_best.solution

    cdef LegacyNativeAgent current_g_best(self):
        if self._g_best_row >= 0:
            return self.pop.agent(self._g_best_row)
        return self.g_best

    cpdef NativeTarget get_target(self, object solution, bint counted=True):
        if counted:
            self._nfe_counter += 1
        return self.problem.get_target(solution)

    # -- solve ---------------------------------------------------------------
    def check_problem(self, problem, seed: int | None) -> None:
        # Mirrors the classic engine: a Problem instance is re-seeded, while a
        # dict builds a fresh problem and ignores any "seed" key.
        self.problem = _NativeProblemPy.coerce(problem)
        if isinstance(problem, Problem):
            problem.seed = seed
        if type(problem) is not dict:
            self.problem.seed = seed

        self.generator = np.random.default_rng(seed)
        self.rng = random.Random(seed)  # local RNG for random module

        self.pop, self.g_best, self.g_worst = None, None, None
        self._starting = None
        self._g_best_row = -1
        self._last_gbest_fit, self._repeated_times = None, 0

    cdef void _update_repeated_times(self):
        if self._termination is None:
            return
        cdef double fit = self.g_best.target.fitness

        if (
            self._last_gbest_fit is not None
            and abs(fit - self._last_gbest_fit) <= self._termination.epsilon
        ):
            self._repeated_times += 1
        else:
            self._repeated_times = 0

        self._last_gbest_fit = fit

    def check_termination(self, mode="start", termination=None, epoch=None):
        if mode == "start":
            self._termination = termination

            if termination is not None:
                if isinstance(termination, Termination):
                    self._termination = termination
                elif type(termination) == dict:
                    self._termination = Termination(**termination)
                else:
                    raise ValueError(
                        "Termination needs to be a dict or an instance of Termination class."
                    )

                self._nfe_counter = 0
                self._termination.set_start_values(
                    0, self._nfe_counter, time.perf_counter(), 0
                )

            return None

        if self._termination is None:
            return False
        return self._termination.should_terminate(
            epoch, self._nfe_counter, time.perf_counter(), self._repeated_times
        )

    cpdef LegacyNativeAgent solve(
        self,
        object problem,
        object termination=None,
        object starting_solutions=None,
        object seed=None,
        bint debug=False,
        bint track_population=False,
        object history_path=None,
        object before_iteration=None,
        object after_iteration=None,
    ):
        cdef int epoch
        cdef Py_ssize_t b
        cdef double time_epoch = 0.0
        cdef bint tracking = debug or track_population or history_path is not None

        self.check_problem(problem, seed)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_solutions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()

        if tracking:
            if before_iteration is not None:
                self.tracker.before = before_iteration
            if after_iteration is not None:
                self.tracker.after = after_iteration
            self.tracker.start(
                optimizer=self,
                problem=self.problem,
                seed=seed,
                capture_population=track_population,
                history_path=history_path,
            )

        for epoch in range(1, self.epoch + 1):
            if tracking:
                self.tracker.epoch = epoch
                if self.tracker.before is not None:
                    self.tracker.before(self.pop)
                time_epoch = perf_counter()

            self.evolve(epoch)

            # g_best becomes an alias of the best row (see module docstring);
            # rows are only reordered when the algorithm asks for it.
            order = self.sorted_order(self.pop)
            if self.sort_flag:
                self.pop = self.pop.take(order)
                b = 0
            else:
                b = order[0]
            self.pop.weights = self.problem._obj_weights
            self.g_best = self.pop.agent(b)
            self._g_best_row = b

            self._update_repeated_times()

            if tracking:
                if self.tracker.after is not None:
                    self.tracker.after(self.pop)
                self.track_optimize_step(self.pop, epoch, perf_counter() - time_epoch)

            if self._termination is not None and self._termination.should_terminate(
                epoch, self._nfe_counter, perf_counter(), self._repeated_times
            ):
                break

        if tracking:
            self.track_optimize_process()

        if self.g_best is None:
            raise ValueError("Best solution was not found.")

        return self.g_best

    def track_optimize_step(self, pop=None, epoch=None, runtime=None) -> None:
        self.tracker.record(epoch, pop, runtime, self.nf_counter, self.g_best.target.fitness)

    def track_optimize_process(self) -> None:
        self.tracker.finalize()

    # -- array helpers kept for the collection ---------------------------------
    def compare_fitness(self, fitness_x, fitness_y, minmax: str = "min") -> bool:
        if minmax == "min":
            return fitness_x < fitness_y
        return not (fitness_x < fitness_y)

    def get_index_roulette_wheel_selection(self, list_fitness: np.ndarray):
        """
        This method can handle min/max problem, and negative or positive fitness value.

        Args:
            list_fitness (nd.array): 1-D numpy array

        Returns:
            int: Index of selected solution
        """
        if isinstance(list_fitness, (list, tuple, np.ndarray)):
            list_fitness = np.array(list_fitness).ravel()

        if np.ptp(list_fitness) == 0:
            return int(self.generator.integers(0, len(list_fitness)))

        if np.any(list_fitness < 0):
            list_fitness = list_fitness - np.min(list_fitness)

        final_fitness: np.ndarray = list_fitness
        if self.problem.minmax == "min":
            final_fitness = np.max(list_fitness) - list_fitness

        prob = final_fitness / np.sum(final_fitness)

        return int(self.generator.choice(range(0, len(list_fitness)), p=prob))

    def get_levy_flight_step(
        self,
        beta: float = 1.0,
        multiplier: float = 0.001,
        size: list | tuple | np.ndarray | None = None,
        case: int = 0,
    ) -> float | list | np.ndarray:
        """
        Get the Levy-flight step size

        Args:
            beta (float): Should be in range [0, 2].

                * 0-1: small range --> exploit
                * 1-2: large range --> explore

            multiplier (float): default = 0.001
            size (tuple, list): size of levy-flight steps, for example: (3, 2), 5, (4, )
            case (int): Should be one of these value [0, 1, -1].

                * 0: return multiplier * s * self.generator.uniform()
                * 1: return multiplier * s * self.generator.normal(0, 1)
                * -1: return multiplier * s

        Returns:
            float, list, np.ndarray: The step size of Levy-flight trajectory
        """
        # u and v are two random variables which follow self.generator.normal distribution
        # sigma_u : standard deviation of u
        sigma_u = np.power(
            math.gamma(1.0 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2.0) * beta * np.power(2.0, (beta - 1) / 2)),
            1.0 / beta,
        )

        # sigma_v : standard deviation of v
        sigma_v = 1
        size = 1 if size is None else size

        u = self.generator.normal(0, sigma_u, size)
        v = self.generator.normal(0, sigma_v, size)
        s = u / np.power(np.abs(v) + self.EPSILON, 1 / beta)

        if case == 0:
            step = multiplier * s * self.generator.uniform()
        elif case == 1:
            step = multiplier * s * self.generator.normal(0, 1)
        else:
            step = multiplier * s

        return step[0] if size == 1 else step

    def crossover_arithmetic(self, dad_pos=None, mom_pos=None):
        """
        Args:
            dad_pos: position of dad
            mom_pos: position of mom

        Returns:
            list: position of 1st and 2nd child
        """
        r = self.generator.uniform()  # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad_pos) + np.multiply((1 - r), mom_pos)
        w2 = np.multiply(r, mom_pos) + np.multiply((1 - r), dad_pos)

        return w1, w2
