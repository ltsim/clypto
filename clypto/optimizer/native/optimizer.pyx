"""Engine root of ``LegacyOptimizer`` (``cy.Optimizer``).

``solve()`` binds the problem, the RNGs and the termination, runs the lifecycle
and the epoch loop, and tracks the run. Algorithms implement the C method
``cdef void evolve(self, int epoch)`` and may override ``initialize_variables``,
``initialization`` and ``before_main_loop``. The engine implements the C-level
steps (``check_problem``, ``before_initialization``, ``after_initialization``,
``after_evolve``, which ratchets ``g_best`` to the best ever seen).

``mode`` picks how a batch of agents is evaluated: ``"sequential"`` (default),
``"batch"`` (the whole batch in one call, one line of the classic loop kept) or
``"parallel"`` (the batch on OpenMP threads when the problem has a nogil
evaluator; the team is fixed once per ``solve`` via ``threads``). ``"swarm"``
remains an alias of ``"batch"``.
"""
import random
import time

import numpy as np

from cython.parallel cimport prange

from clypto.optimizer.history import Tracker
from clypto.optimizer.termination import Termination

cdef tuple MODES = ("sequential", "batch", "parallel")

from openmp cimport omp_set_num_threads, omp_get_max_threads


cdef class NativeOptimizer:
    def __init__(self, parameters=(), sort_flag=False, name=None, mode="sequential"):
        self.tracker = Tracker()
        self.EPSILON = 10e-10
        self.SUPPORTED_ARRAYS = (list, tuple, np.ndarray)
        self.name = type(self).__name__ if name is None else name
        self._params_name_ordered = tuple(parameters)
        self._termination = None
        self.generator = None
        self.rng = None
        if mode is None:
            mode = "sequential"
        if mode == "swarm":
            mode = "batch"
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES} (alias: 'swarm'), got {mode!r}.")
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

    cdef void evolve(self, int epoch):
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
        threads=0,
        before_iteration=None,
        after_iteration=None,
        metrics=None,
    ):
        cdef int epoch, repeated_times = 0, n_threads, previous_threads
        cdef double time_epoch = 0.0
        cdef bint tracking = debug or track_population or (metrics is not None)
        last_gbest_fit = None

        # The tracker always describes the most recent run; drop the previous one.
        self.tracker.reset()

        self.generator = np.random.default_rng(seed)
        self.rng = random.Random(seed)  # local RNG for the random module
        for klass in type(self).__mro__:
            if "evolve" in vars(klass):
                raise TypeError(
                    f"{klass.__name__}.evolve is a Python method; the engine calls the C method "
                    "`cdef void evolve(self, int epoch)` (write a Python optimizer with @cy.optimizer)."
                )
        self.check_problem(problem, seed)
        if type(threads) is not int or threads < 0:
            raise ValueError("threads must be a non-negative int (0 lets OpenMP decide).")
        self.population.threads = threads
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
            self.tracker.start(
                optimizer=self,
                problem=self.problem,
                seed=seed,
                capture_population=track_population,
                metrics=metrics,
            )

        # OpenMP: the team is fixed once for this run; every parallel region reuses it.
        previous_threads = omp_get_max_threads()
        n_threads = threads if threads > 0 else previous_threads
        omp_set_num_threads(n_threads)
        try:
            if self.mode == "parallel" and n_threads > 1:
                with nogil:
                    for i in prange(n_threads, schedule="static"):
                        pass  # warm-up: the team exists after this region
            for epoch in range(1, self.epoch + 1):
                time_epoch = time.perf_counter()

                if before_iteration is not None:
                    before_iteration(self.population, epoch)

                self.evolve(epoch)
                self.after_evolve()

                if termination is not None:
                    fit = self.g_best.fitness
                    if last_gbest_fit is not None and abs(fit - last_gbest_fit) <= termination.epsilon:
                        repeated_times += 1
                    else:
                        repeated_times = 0
                    last_gbest_fit = fit

                if after_iteration is not None:
                    after_iteration(self.population, epoch)

                if tracking:
                    self.tracker.record(
                        self.population, time.perf_counter() - time_epoch,
                        self.problem.n_evals, self.g_best.fitness,
                    )

                if termination is not None and termination.should_terminate(
                    epoch, self.problem.n_evals, time.perf_counter(), repeated_times
                ):
                    break
        finally:
            # Close the recording even when a hook or evolve() raises, so the
            # tracker never stays mid-run.
            omp_set_num_threads(previous_threads)
            if tracking:
                self.tracker.finalize()

        if self.g_best is None:
            raise ValueError("Best solution was not found.")

        return self.g_best
