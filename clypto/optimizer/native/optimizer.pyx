"""Engine root shared by ``LegacyOptimizer`` and ``VectorizeOptimizer``.

``solve()`` binds the problem, the RNGs and the termination, runs the lifecycle
and the epoch loop, and tracks the run. Algorithms implement ``evolve(epoch)``
and may override ``initialize_variables``, ``initialization`` and
``before_main_loop``. Each engine declares ``self.population``, ``self.problem``
and ``self.g_best`` its own way and implements the C-level engine steps
(``check_problem``, ``before_initialization``, ``after_initialization``,
``after_evolve``, which refreshes ``g_best``).
"""
import random
import time

import numpy as np

from clypto.optimizer.history import Tracker
from clypto.optimizer.termination import Termination


cdef class NativeOptimizer:
    def __init__(self, parameters=(), sort_flag=False, name=None, mode=None):
        self.tracker = Tracker()
        self.EPSILON = 10e-10
        self.AVAILABLE_MODES = ("swarm", "parallel", "thread", "process")
        self.SUPPORTED_ARRAYS = (list, tuple, np.ndarray)
        self.name = type(self).__name__ if name is None else name
        self._params_name_ordered = tuple(parameters)
        self._termination = None
        self.generator = None
        self.rng = None
        self.mode = mode
        self.epoch = 0
        self.sort_flag = sort_flag

    @property
    def termination(self):
        return self._termination

    @property
    def nf_counter(self):
        """Evaluations of the current (or last) ``solve()``."""
        return 0 if self.problem is None else self.problem.n_evals

    @property
    def parameters(self):
        """Current values of the registered hyper-parameters."""
        return {name: getattr(self, name) for name in self._params_name_ordered}

    @parameters.setter
    def parameters(self, parameters):
        """Register hyper-parameter names (list/tuple) or override their values (dict)."""
        if type(parameters) is dict:
            invalid = set(parameters) - set(self._params_name_ordered)
            if invalid:
                raise ValueError(
                    f"Invalid input parameters: {invalid} for {self.name} optimizer. "
                    f"Valid parameters are: {set(self._params_name_ordered)}."
                )
            for key, value in parameters.items():
                setattr(self, key, value)
        else:
            self._params_name_ordered = tuple(parameters)

    def __str__(self):
        return type(self).__name__

    # -- algorithm hooks -----------------------------------------------------------
    def initialize_variables(self):
        """Set up algorithm state once the problem is bound (before the population exists)."""
        pass

    def initialization(self):
        """Create the initial population."""
        raise NotImplementedError

    def before_main_loop(self):
        pass

    def evolve(self, epoch):
        raise NotImplementedError(f"{type(self).__name__} does not implement evolve().")

    # -- engine steps ----------------------------------------------------------------
    cdef void check_problem(self, object problem, object seed):
        raise NotImplementedError

    cdef void before_initialization(self, object starting_solutions):
        pass

    cdef void after_initialization(self):
        pass

    cdef void after_evolve(self):
        pass

    # -- solve -------------------------------------------------------------------
    def solve(
        self,
        problem,
        termination=None,
        starting_solutions=None,
        seed=None,
        debug=False,
        track_population=False,
        history_path=None,
        before_iteration=None,
        after_iteration=None,
    ):
        cdef int epoch, repeated_times = 0
        cdef double time_epoch = 0.0
        cdef bint tracking = debug or track_population or history_path is not None
        last_gbest_fit = None

        self.generator = np.random.default_rng(seed)
        self.rng = random.Random(seed)  # local RNG for the random module
        self.check_problem(problem, seed)
        self.problem.n_evals = 0

        if termination is not None:
            if type(termination) is dict:
                termination = Termination(**termination)
            elif not isinstance(termination, Termination):
                raise ValueError("Termination needs to be a dict or an instance of Termination class.")
            termination.set_start_values(0, self.problem.n_evals, time.perf_counter(), 0)
        self._termination = termination

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
                    self.tracker.before(self.population)
                time_epoch = time.perf_counter()

            self.evolve(epoch)
            self.after_evolve()

            if termination is not None:
                fit = self.g_best.fitness
                if last_gbest_fit is not None and abs(fit - last_gbest_fit) <= termination.epsilon:
                    repeated_times += 1
                else:
                    repeated_times = 0
                last_gbest_fit = fit

            if tracking:
                if self.tracker.after is not None:
                    self.tracker.after(self.population)
                self.tracker.record(
                    epoch, self.population, time.perf_counter() - time_epoch,
                    self.problem.n_evals, self.g_best.fitness,
                )

            if termination is not None and termination.should_terminate(
                epoch, self.problem.n_evals, time.perf_counter(), repeated_times
            ):
                break

        if tracking:
            self.tracker.finalize()

        if self.g_best is None:
            raise ValueError("Best solution was not found.")

        return self.g_best
