"""Engine root shared by ``LegacyOptimizer`` and ``VectorizeOptimizer``.

``solve()`` is the only public method: it binds the problem, the RNGs and the
termination, runs the ``_`` lifecycle hooks and the epoch loop, and tracks the
run. Each branch stores the population its own way (``self.pop``, ``self.problem``
and ``self.g_best`` are declared by the branch) and refreshes ``g_best`` in
``_after_evolve``. Algorithms implement ``_evolve(epoch)``.

The agent helpers below rank through :class:`Population` and serve the legacy
engine and the vectorize engine's ``AgentListOptimizer`` compatibility layer alike.
"""
import math
import random
import time

import numpy as np

from clypto.optimizer.history import Tracker
from clypto.optimizer.native.population import Population
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
        self._last_gbest_fit = None
        self._repeated_times = 0
        self._nfe_counter = 0
        self.generator = None
        self.rng = None
        self.mode = mode
        self.epoch = 0
        self.pop_size = 0
        self.sort_flag = sort_flag

    @property
    def termination(self):
        return self._termination

    @property
    def nf_counter(self):
        return self._nfe_counter

    @property
    def parameters(self):
        """Current values of the registered hyper-parameters."""
        return {name: getattr(self, name) for name in self._params_name_ordered}

    def __str__(self):
        return type(self).__name__

    def _set_parameters(self, parameters):
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

    # -- lifecycle hooks ---------------------------------------------------------
    def _initialize_variables(self):
        pass

    def _before_main_loop(self):
        pass

    def _evolve(self, epoch):
        raise NotImplementedError(f"{type(self).__name__} does not implement _evolve().")

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
        cdef int epoch
        cdef double time_epoch = 0.0
        cdef bint tracking = debug or track_population or history_path is not None

        self._check_problem(problem, seed)
        self.generator = np.random.default_rng(seed)
        self.rng = random.Random(seed)  # local RNG for the random module
        self._nfe_counter = 0
        self._last_gbest_fit, self._repeated_times = None, 0
        self._termination = self._build_termination(termination)

        self._initialize_variables()
        self._before_initialization(starting_solutions)
        self._initialization()
        self._after_initialization()
        self._before_main_loop()

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
                time_epoch = time.perf_counter()

            self._evolve(epoch)
            self._after_evolve()
            self._update_repeated_times()

            if tracking:
                if self.tracker.after is not None:
                    self.tracker.after(self.pop)
                self.tracker.record(
                    epoch, self.pop, time.perf_counter() - time_epoch,
                    self._nfe_counter, self.g_best.target.fitness,
                )

            if self._termination is not None and self._termination.should_terminate(
                epoch, self._nfe_counter, time.perf_counter(), self._repeated_times
            ):
                break

        if tracking:
            self.tracker.finalize()

        if self.g_best is None:
            raise ValueError("Best solution was not found.")

        return self.g_best

    def _build_termination(self, termination):
        if termination is None:
            return None
        if type(termination) is dict:
            termination = Termination(**termination)
        elif not isinstance(termination, Termination):
            raise ValueError("Termination needs to be a dict or an instance of Termination class.")
        termination.set_start_values(0, self._nfe_counter, time.perf_counter(), 0)
        return termination

    cdef void _update_repeated_times(self):
        if self._termination is None:
            return
        cdef double fit = self.g_best.target.fitness
        if self._last_gbest_fit is not None and abs(fit - self._last_gbest_fit) <= self._termination.epsilon:
            self._repeated_times += 1
        else:
            self._repeated_times = 0
        self._last_gbest_fit = fit

    # -- evaluation and comparison -------------------------------------------------
    def _get_target(self, solution, counted=True):
        if counted:
            self._nfe_counter += 1
        return self.problem.get_target(solution)

    def _compare_fitness(self, fitness_x, fitness_y, sense="min"):
        """True when ``fitness_x`` is better (``"max"``: at least as good)."""
        if sense == "min":
            return fitness_x < fitness_y
        return not (fitness_x < fitness_y)

    def _compare_target(self, target_x, target_y, sense="min"):
        return self._compare_fitness(target_x.fitness, target_y.fitness, sense)

    # -- list-of-agents helpers ------------------------------------------------------
    def _get_sorted_indices_population(self, pop, sense="min"):
        ranked = Population(pop, sense).sort()
        return ranked, ranked.idx

    def _get_sorted_population(self, pop, sense="min"):
        return Population(pop, sense).sort()

    def _get_best_agent(self, pop, sense="min"):
        return self._get_sorted_population(pop, sense)[0].copy()

    def _get_index_best(self, pop, sense="min"):
        fit_list = np.array([agent.target.fitness for agent in pop])
        return int(np.argmin(fit_list)) if sense == "min" else int(np.argmax(fit_list))

    def _get_special_agents(self, pop, n_best=3, n_worst=3, sense="min"):
        """Sorted population, copies of the ``n_best`` best and ``n_worst`` worst (``None`` skips)."""
        pop = self._get_sorted_population(pop, sense)
        best = None if n_best is None else [agent.copy() for agent in pop[:n_best]]
        worst = None if n_worst is None else [agent.copy() for agent in pop[::-1][:n_worst]]
        return pop, best, worst

    def _get_special_fitness(self, pop, sense="min"):
        """Total, best and worst fitness."""
        total_fitness = np.sum([agent.target.fitness for agent in pop])
        pop = self._get_sorted_population(pop, sense)
        return total_fitness, pop[0].target.fitness, pop[-1].target.fitness

    def _get_better_agent(self, agent_x, agent_y, sense="min", reverse=False):
        """Copy of the better agent; a tie keeps ``agent_y`` when minimizing, ``agent_x`` when maximizing."""
        maximize = (sense != "min") != bool(reverse)
        if agent_x.target.fitness < agent_y.target.fitness:
            return agent_y.copy() if maximize else agent_x.copy()
        return agent_x.copy() if maximize else agent_y.copy()

    def _greedy_selection_population(self, pop_old, pop_new, sense="min"):
        """Per position, the new agent when strictly better, else the old one."""
        return Population(pop_old, sense).greedy(pop_new)

    def _get_sorted_and_trimmed_population(self, pop, pop_size=None, sense="min"):
        return self._get_sorted_population(pop, sense)[:pop_size]

    def _update_global_best_agent(self, pop):
        """Sorted population and its best agent."""
        sorted_pop = self._get_sorted_population(pop, self.problem.sense)
        return sorted_pop, sorted_pop[0]

    def _generate_group_population(self, pop, n_groups, m_agents):
        return [[agent.copy() for agent in pop[idx * m_agents:(idx + 1) * m_agents]] for idx in range(n_groups)]

    def _generate_opposition_solution(self, agent, g_best):
        pos_new = (
            self.problem.bounds.low + self.problem.bounds.up - g_best.solution
            + self.generator.uniform() * (g_best.solution - agent.solution)
        )
        return self._correct_solution(pos_new)

    # -- selection and random steps ------------------------------------------------------
    def _get_index_roulette_wheel_selection(self, list_fitness):
        """Roulette-wheel index for a min or max problem; handles negative fitness."""
        list_fitness = np.array(list_fitness).ravel()
        if np.ptp(list_fitness) == 0:
            return int(self.generator.integers(0, len(list_fitness)))
        if np.any(list_fitness < 0):
            list_fitness = list_fitness - np.min(list_fitness)
        final_fitness = list_fitness
        if self.problem.sense == "min":
            final_fitness = np.max(list_fitness) - list_fitness
        prob = final_fitness / np.sum(final_fitness)
        return int(self.generator.choice(range(0, len(list_fitness)), p=prob))

    def _get_index_kway_tournament_selection(self, pop, k_way=0.2, output=2, reverse=False):
        """Indexes of the ``output`` best (``reverse``: worst) of ``k_way`` random agents."""
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = self.generator.choice(range(len(pop)), int(k_way), replace=False)
        list_parents = sorted(
            [[idx, pop[idx].target.fitness] for idx in list_id],
            key=lambda agent: agent[1], reverse=self.problem.sense != "min",
        )
        if reverse:
            return [parent[0] for parent in list_parents[-output:]]
        return [parent[0] for parent in list_parents[:output]]

    def _get_levy_flight_step(self, beta=1.0, multiplier=0.001, size=None, case=0):
        """Levy-flight step of shape ``size``.

        ``beta`` in [0, 2] (0-1 exploit, 1-2 explore). ``case`` 0 scales by a
        uniform draw, 1 by a normal draw, -1 not at all.
        """
        sigma_u = np.power(
            math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2.0) * beta * np.power(2.0, (beta - 1) / 2)),
            1.0 / beta,
        )
        size = 1 if size is None else size
        u = self.generator.normal(0, sigma_u, size)
        v = self.generator.normal(0, 1, size)
        s = u / np.power(np.abs(v) + self.EPSILON, 1 / beta)
        if case == 0:
            step = multiplier * s * self.generator.uniform()
        elif case == 1:
            step = multiplier * s * self.generator.normal(0, 1)
        else:
            step = multiplier * s
        return step[0] if size == 1 else step

    def _crossover_arithmetic(self, dad_pos, mom_pos):
        r = self.generator.uniform()  # w1 = w2 when r = 0.5
        w1 = np.multiply(r, dad_pos) + np.multiply((1 - r), mom_pos)
        w2 = np.multiply(r, mom_pos) + np.multiply((1 - r), dad_pos)
        return w1, w2
