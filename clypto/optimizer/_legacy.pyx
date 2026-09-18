#!/usr/bin/env python
# Created by "Thieu" at 08:58, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Private compiled engine used by the built-in algorithm collection.

The public classic base injected by ``@cy.legacy`` lives in
``clypto/optimizer/legacy.py`` and is deliberately independent of this module.
"""
import math
import random
import time
import typing

import numpy as np
import numpy.random as npr
import numpy.typing as npt
from cython.parallel cimport parallel, prange

from clypto.agents._core cimport _LegacyAgent
from clypto.utils._nogil cimport _NogilEvaluator
from clypto.utils.history import Tracker
from clypto.utils.problem import Problem
from clypto.utils.target import Target
from clypto.utils.termination import Termination
from clypto.utils.validator import Validator


cdef class _LegacyOptimizer:
    def __init__(self, **kwargs):
        self.tracker = Tracker()
        self.validator = Validator()
        self.EPSILON = 10e-10
        self.AVAILABLE_MODES = ("swarm", "parallel", "thread", "process")
        self.SUPPORTED_ARRAYS = list, tuple, np.ndarray
        self._last_gbest_fit = None
        self._repeated_times = 0
        self._nfe_counter = 1
        self._name = kwargs.get("name", self.__class__.__name__)
        self._params_name_ordered = None
        self.generator = None
        self._termination = None
        self.mode = kwargs.get("mode", None)
        self.epoch = 0
        self.pop_size = 0
        self.n_workers = None
        self.pop = None
        self.g_best = _LegacyAgent()
        self.g_worst = None
        self.problem = None
        self.sort_flag = False
        self.parameters = {}
        self.is_parallelizable = True
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

    def set_parameters(
        self, parameters: typing.Union[typing.List, typing.Tuple, typing.Dict]
    ) -> None:
        """
        Set the parameters for current optimizer.

        if paras is a list of parameter's name, then it will set the default value in optimizer as current parameters
        if paras is a dict of parameter's name and value, then it will override the current parameters

        Args:
            parameters: The parameters
        """
        if type(parameters) in (list, tuple):
            self._params_name_ordered = tuple(parameters)
            self.parameters = {}

            for name in parameters:
                self.parameters[name] = getattr(self, name)

        elif type(parameters) is dict:
            valid_para_names = set(self.parameters.keys())
            new_para_names = set(parameters.keys())

            if new_para_names.issubset(valid_para_names):
                for key, value in parameters.items():
                    setattr(self, key, value)
                    self.parameters[key] = value
            else:
                raise ValueError(
                    f"Invalid input parameters: {new_para_names} for {self.get_name()} optimizer. "
                    f"Valid parameters are: {valid_para_names}."
                )

    def get_parameters(self) -> typing.Dict:
        """
        Get parameters of optimizer.
        """
        return self.parameters

    def get_attributes(self) -> typing.Dict:
        """
        Get all attributes in optimizer.
        """
        data = {}
        for key in (
            "epoch", "pop_size", "pop", "problem", "g_best", "g_worst",
            "mode", "sort_flag", "is_parallelizable", "parameters",
        ):
            data[key] = getattr(self, key, None)
        return data

    def get_name(self) -> str:
        """
        Get name of the optimizer
        """
        return self.name

    def __str__(self):
        return f"{self.__class__.__name__}"

    def initialize_variables(self):
        pass

    def before_initialization(
        self,
        starting_solutions: (typing.Sequence[float] | NDArrayType | None) = None,
    ) -> None:
        """
        Args:
            starting_solutions: The starting solutions (not recommended)
        """
        if starting_solutions is None:
            ...
        elif (
            type(starting_solutions) in self.SUPPORTED_ARRAYS
            and len(starting_solutions) == self.pop_size
        ):
            if (
                type(starting_solutions[0]) in self.SUPPORTED_ARRAYS
                and len(starting_solutions[0]) == self.problem.n_dims
            ):
                self.pop = [
                    self.generate_agent(solution) for solution in starting_solutions
                ]
            else:
                raise ValueError(
                    "Invalid starting_solutions. It should be a list of positions or 2D matrix of positions only."
                )
        else:
            raise ValueError(
                "Invalid starting_solutions. It should be a list/2D matrix of positions with same length as pop_size."
            )

    def initialization(self) -> None:
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

    def after_initialization(self) -> None:
        # The initial population is sorted or not depended on algorithm's strategy
        pop_temp, best, worst = self.get_special_agents(
            self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax
        )

        self.g_best, self.g_worst = best[0], worst[0]

        if self.sort_flag:
            self.pop = pop_temp

    def before_main_loop(self):
        pass

    def evolve(self, epoch: int) -> None:
        pass

    def check_problem(self, problem: dict | Problem, seed: int | None) -> None:
        if isinstance(problem, Problem):
            problem.seed = seed
            self.problem = problem
        elif type(problem) == dict:
            problem["seed"] = seed
            self.problem = Problem(**problem)
        else:
            raise ValueError(
                "problem needs to be a dict or an instance of Problem class."
            )

        self.generator = np.random.default_rng(seed)
        self.rng = random.Random(seed)  # local RNG for random module

        # Reset for this solve() call; initialization()/after_initialization() (called
        # immediately after, still within solve(), before any evolve()) set these back.
        self.pop, self.g_best, self.g_worst = None, None, None  # type: ignore[assignment]
        self._last_gbest_fit, self._repeated_times = None, 0

    def __update_repeated_times(self) -> None:
        if self._termination is None:
            return
        fit = float(self.g_best.target.fitness)

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
        else:
            finished = False

            if self._termination is not None:
                finished = self._termination.should_terminate(
                    epoch,
                    self._nfe_counter,
                    time.perf_counter(),
                    self._repeated_times,
                )

            return finished

    def solve(
        self,
        problem: Problem,
        termination: Termination | None = None,
        starting_solutions: (
            typing.Sequence[float] | npt.NDArray[np.float64] | None
        ) = None,
        seed: int | None = None,
        debug: bool = False,
        track_population: bool = False,
        history_path: str | None = None,
        before_iteration: typing.Optional[typing.Callable] = None,
        after_iteration: typing.Optional[typing.Callable] = None,
    ) -> _LegacyAgent:
        self.check_problem(problem, seed)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_solutions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()

        tracking = debug or track_population or history_path is not None
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

        loop = range(1, self.epoch + 1)

        for epoch in loop:
            if tracking:
                self.tracker.epoch = epoch
                if self.tracker.before is not None:
                    self.tracker.before(self.pop)

            time_epoch = time.perf_counter()

            ## Evolve method will be called in child class
            self.evolve(epoch)

            # Update global best solution, the population is sorted or not depended on algorithm's strategy
            pop_temp = self.get_sorted_population(self.pop, self.problem.minmax)
            self.g_best = pop_temp[0]

            if self.sort_flag:
                self.pop = pop_temp

            self.__update_repeated_times()

            if tracking and self.tracker.after is not None:
                self.tracker.after(self.pop)

            time_epoch = time.perf_counter() - time_epoch

            if tracking:
                self.track_optimize_step(self.pop, epoch, time_epoch)

            if self.check_termination("end", None, epoch):
                break

        if tracking:
            self.track_optimize_process()

        if self.g_best is None:
            raise ValueError("Best solution was not found.")

        return self.g_best

    def track_optimize_step(
        self,
        pop: list[_LegacyAgent] | None = None,
        epoch: int | None = None,
        runtime: float | None = None,
    ) -> None:
        self.tracker.record(epoch, pop, runtime, self.nf_counter, self.g_best.target.fitness)

    def track_optimize_process(self) -> None:
        self.tracker.finalize()

    def generate_empty_agent(self, solution: typing.Optional[NDArrayType] = None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)

        return _LegacyAgent(solution=solution)

    def generate_agent(self, solution: typing.Optional[NDArrayType] = None):
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)

        return agent

    def generate_population(
        self, pop_size: typing.Optional[int] = None
    ) -> list[_LegacyAgent]:
        if pop_size is None:
            pop_size = self.pop_size

        return [self.generate_agent() for _ in range(0, pop_size)]

    def amend_solution(self, solution: NDArrayType) -> NDArrayType:
        return np.clip(solution, self.problem.lb, self.problem.ub)

    def correct_solution(self, solution: NDArrayType) -> NDArrayType:
        solution = self.amend_solution(solution)

        return self.problem.correct_solution(solution)

    def update_target_for_population(self, pop):
        cdef object pos_list = [agent.solution for agent in pop]
        cdef Py_ssize_t n = len(pop)
        cdef Py_ssize_t idx

        if self.mode == "swarm":
            for idx in range(n):
                pop[idx].target = self.get_target(pos_list[idx], counted=False)
        elif self.mode in ("parallel", "thread", "process"):
            self._evaluate_parallel(pop, pos_list, n)
        else:
            return pop

        self._nfe_counter += n

        return pop

    def _evaluate_parallel(self, pop, pos_list, Py_ssize_t n):
        """Re-evaluate ``pop`` in place, using a nogil batch evaluator when present.

        Without a ``Problem.evaluator`` (the default) this falls back to the
        sequential Python evaluation, preserving the classic results.
        """
        cdef _NogilEvaluator evaluator
        cdef double[:, ::1] X
        cdef double[:, ::1] F
        cdef Py_ssize_t i
        cdef Py_ssize_t n_dims

        if self.problem.evaluator is not None and self.problem.n_objs == 1:
            evaluator = <_NogilEvaluator>self.problem.evaluator
            X = np.ascontiguousarray(pos_list, dtype=np.float64)
            F = np.empty((n, 1), dtype=np.float64)
            n_dims = X.shape[1]
            with nogil, parallel():
                for i in prange(n, schedule="static"):
                    evaluator.row(&X[i, 0], n_dims, &F[i, 0])
            for i in range(n):
                pop[i].target = Target(
                    objectives=float(F[i, 0]), weights=self.problem.obj_weights
                )
        else:
            for i in range(n):
                pop[i].target = self.get_target(pos_list[i], counted=False)

    def get_target(self, solution: NDArrayType, counted: bool = True) -> Target:
        if counted:
            self._nfe_counter += 1

        return self.problem.get_target(solution)

    def compare_target(self, target_x: Target, target_y: Target, minmax: str = "min") -> bool:
        if minmax == "min":
            return True if target_x.fitness < target_y.fitness else False
        else:
            return False if target_x.fitness < target_y.fitness else True

    def compare_fitness(self, 
        fitness_x: float | int, fitness_y: float | int, minmax: str = "min"
    ) -> bool:
        if minmax == "min":
            return True if fitness_x < fitness_y else False
        else:
            return False if fitness_x < fitness_y else True

    def duplicate_pop(self, pop: list[_LegacyAgent]) -> list[_LegacyAgent]:
        return [agent.copy() for agent in pop]

    def get_sorted_population(self, 
        pop: list[_LegacyAgent], minmax: str = "min"
    ) -> list[_LegacyAgent]:
        """
        Get sorted population based on type (minmax) of problem

        Args:
            pop: The population
            minmax: The type of the problem

        Returns:
            Sorted population (1st agent is the best, last agent is the worst
        """

        list_fits = [agent.target.fitness for agent in pop]
        indices = np.argsort(list_fits).tolist()

        if minmax == "max":
            indices = indices[::-1]

        pop_new = [pop[idx] for idx in indices]

        return pop_new

    def get_sorted_indices_population(self, 
        pop: list[_LegacyAgent], minmax: str = "min"
    ) -> tuple[list[_LegacyAgent], list[int]]:
        """
        Get sorted indices population based on type (minmax) of problem

        Args:
            pop: The population
            minmax: The type of the problem

        Returns:
            Sorted population (1st agent is the best, last agent is the worst
        """

        list_fits = [agent.target.fitness for agent in pop]
        indices = np.argsort(list_fits).tolist()

        if minmax == "max":
            indices = indices[::-1]

        pop_new = [pop[idx] for idx in indices]

        return pop_new, indices

    def get_best_agent(self, pop: list[_LegacyAgent], minmax: str = "min") -> _LegacyAgent:
        """
        Args:
            pop: The population of agents
            minmax: The type of problem

        Returns:
            The best agent
        """
        pop = self.get_sorted_population(pop, minmax)
        return pop[0].copy()

    def get_index_best(self, pop: list[_LegacyAgent], minmax: str = "min") -> int:
        fit_list = np.array([agent.target.fitness for agent in pop])
        if minmax == "min":
            return int(np.argmin(fit_list))
        else:
            return int(np.argmax(fit_list))

    def get_worst_agent(self, pop: list[_LegacyAgent], minmax: str = "min") -> _LegacyAgent:
        """
        Args:
            pop: The population of agents
            minmax: The type of problem

        Returns:
            The worst agent
        """
        pop = self.get_sorted_population(pop, minmax)
        return pop[-1].copy()

    def get_special_agents(self, 
        pop: list[_LegacyAgent],
        n_best: int = 3,
        n_worst: int = 3,
        minmax: str = "min",
    ) -> tuple[list[_LegacyAgent], list[_LegacyAgent] | None, list[_LegacyAgent] | None]:
        """
        Get special agents include sorted population, n1 best agents, n2 worst agents

        Args:
            pop: The population
            n_best: Top n1 best agents, default n1=3, good level reduction
            n_worst: Top n2 worst agents, default n2=3, worst level reduction
            minmax: The problem type

        Returns:
            The sorted_population, n1 best agents and n2 worst agents
        """
        pop = self.get_sorted_population(pop, minmax)

        if n_best is None:
            if n_worst is None:
                return pop, None, None
            else:
                return pop, None, [agent.copy() for agent in pop[::-1][:n_worst]]
        else:
            if n_worst is None:
                return pop, [agent.copy() for agent in pop[:n_best]], None
            else:
                return (
                    pop,
                    [agent.copy() for agent in pop[:n_best]],
                    [agent.copy() for agent in pop[::-1][:n_worst]],
                )

    def get_special_fitness(self, 
        pop: list[_LegacyAgent], minmax: str = "min"
    ) -> tuple[float | np.ndarray, float, float]:
        """
        Get special target include the total fitness, the best fitness, and the worst fitness

        Args:
            pop: The population
            minmax: The problem type

        Returns:
            The total fitness, the best fitness, and the worst fitness
        """
        total_fitness = np.sum([agent.target.fitness for agent in pop])
        pop = self.get_sorted_population(pop, minmax)
        return total_fitness, pop[0].target.fitness, pop[-1].target.fitness

    def get_better_agent(self, 
        agent_x: _LegacyAgent,
        agent_y: _LegacyAgent,
        minmax: str = "min",
        reverse: bool = False,
    ) -> _LegacyAgent:
        """
        Args:
            agent_x: First agent
            agent_y: Second agent
            minmax: The problem type
            reverse: Reverse the minmax

        Returns:
            The better agent based on fitness
        """
        minmax_dict = {"min": 0, "max": 1}
        idx = minmax_dict[minmax]

        if reverse:
            idx = 1 - idx
        if idx == 0:
            return (
                agent_x.copy()
                if agent_x.target.fitness < agent_y.target.fitness
                else agent_y.copy()
            )
        else:
            return (
                agent_y.copy()
                if agent_x.target.fitness < agent_y.target.fitness
                else agent_x.copy()
            )

    ### Survivor Selection
    def greedy_selection_population(self, 
        pop_old: list[_LegacyAgent] | None = None,
        pop_new: list[_LegacyAgent] | None = None,
        minmax: str = "min",
    ) -> list[_LegacyAgent]:
        """
        Args:
            pop_old: The current population
            pop_new: The next population
            minmax: The problem type

        Returns:
            The new population with better solutions
        """
        len_old, len_new = len(pop_old), len(pop_new)

        if len_old != len_new:
            raise ValueError(
                "Greedy selection of two population with different length."
            )
        if minmax == "min":
            return [
                (
                    pop_new[idx]
                    if pop_new[idx].target.fitness < pop_old[idx].target.fitness
                    else pop_old[idx]
                )
                for idx in range(len_old)
            ]
        else:
            return [
                (
                    pop_new[idx]
                    if pop_new[idx].target.fitness > pop_old[idx].target.fitness
                    else pop_old[idx]
                )
                for idx in range(len_old)
            ]

    def get_sorted_and_trimmed_population(self, 
        pop: list[_LegacyAgent] | None = None,
        pop_size: int | None = None,
        minmax: str = "min",
    ) -> list[_LegacyAgent]:
        """
        Args:
            pop: The population
            pop_size: The number of selected agents
            minmax: The problem type

        Returns:
            The sorted and trimmed population with pop_size size
        """
        pop = self.get_sorted_population(pop, minmax)

        return pop[:pop_size]

    def update_global_best_agent(
        self, pop: list[_LegacyAgent], save: bool = False
    ) -> tuple[list[_LegacyAgent], _LegacyAgent]:
        """
        Update global best and current best solutions in history object.
        Also update global worst and current worst solutions in history object.

        Args:
            pop (list): The population of pop_size individuals
            save (bool): True if you want to add new current/global best to history, False if you just want to update current/global best

        Returns:
            list: Sorted population and the global best solution
        """
        sorted_pop = self.get_sorted_population(pop, self.problem.minmax)

        c_best, c_worst = sorted_pop[0], sorted_pop[-1]

        return sorted_pop, c_best

    ## Selection techniques
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

    def get_index_kway_tournament_selection(
        self,
        pop: list | None = None,
        k_way: float = 0.2,
        output: int = 2,
        reverse: bool = False,
    ) -> list:
        """
        Args:
            pop: The population
            k_way (float/int): The percent or number of solutions are randomized pick
            output (int): The number of outputs
            reverse (bool): set True when finding the worst fitness

        Returns:
            list: List of the selected indexes
        """
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))

        k_way_count: int = int(k_way)
        list_id = self.generator.choice(
            range(len(pop)), k_way_count, replace=False
        )
        list_parents = [[idx, pop[idx].target.fitness] for idx in list_id]

        if self.problem.minmax == "min":
            list_parents = sorted(list_parents, key=lambda agent: agent[1])
        else:
            list_parents = sorted(
                list_parents, key=lambda agent: agent[1], reverse=True
            )

        if reverse:
            return [parent[0] for parent in list_parents[-output:]]

        return [parent[0] for parent in list_parents[:output]]

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

    def generate_opposition_solution(
        self, agent: _LegacyAgent | None = None, g_best: _LegacyAgent | None = None
    ) -> np.ndarray:
        """
        Args:
            agent: The current agent
            g_best: the global best agent

        Returns:
            The opposite solution
        """
        pos_new = (
            self.problem.lb
            + self.problem.ub
            - g_best.solution
            + self.generator.uniform() * (g_best.solution - agent.solution)
        )

        return self.correct_solution(pos_new)

    def generate_group_population(
        self, pop: list[_LegacyAgent], n_groups: int, m_agents: int
    ) -> list:
        """
        Generate a list of group population from pop

        Args:
            pop: The current population
            n_groups: The n groups
            m_agents: The m agents in each group

        Returns:
            A list of group population
        """
        pop_group = []

        for idx in range(0, n_groups):
            group = pop[idx * m_agents : (idx + 1) * m_agents]
            pop_group.append([agent.copy() for agent in group])

        return pop_group

    ### Crossover
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

    #### Improved techniques can be used in any algorithms: 1
    ## Based on this paper: An efficient equilibrium optimizer with mutation strategy for numerical optimization (but still different)
    ## This scheme used after the original and including 4 step:
    ##  s1: sort population, take p1 = 1/2 best population for next round
    ##  s2: do the mutation for p1, using greedy method to select the better solution
    ##  s3: do the search mechanism for p1 (based on global best solution and the updated p1 above), to make p2 population
    ##  s4: construct the new population for next generation
    def improved_ms(self, pop=None, g_best=None):  ## m: mutation, s: search
        pop_len = int(len(pop) / 2)
        ## Sort the updated population based on fitness
        pop = sorted(pop, key=lambda agent: agent.target.fitness)
        pop_s1, pop_s2 = pop[:pop_len], pop[pop_len:]

        ## Mutation scheme
        pop_new = []

        for idx in range(0, pop_len):
            agent = pop_s1[idx].copy()
            pos_new = pop_s1[idx].solution * (
                1 + self.generator.normal(0, 1, self.problem.n_dims)
            )
            agent.solution = self.correct_solution(pos_new)
            pop_new.append(agent)

        pop_new = self.update_target_for_population(pop_new)
        pop_s1 = self.greedy_selection_population(
            pop_s1, pop_new, self.problem.minmax
        )  ## Greedy method --> improved exploitation

        ## Search Mechanism
        pos_s1_list = [agent.solution for agent in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_new = []

        for idx in range(0, pop_len):
            agent = pop_s2[idx].copy()
            pos_new = (g_best.solution - pos_s1_mean) - self.generator.random() * (
                self.problem.lb
                + self.generator.random() * (self.problem.ub - self.problem.lb)
            )
            agent.solution = self.correct_solution(pos_new)
            pop_new.append(agent)

        ## Keep the diversity of populatoin and still improved the exploration
        pop_s2 = self.update_target_for_population(pop_new)
        pop_s2 = self.greedy_selection_population(pop_s2, pop_new, self.problem.minmax)

        ## Construct a new population
        pop = pop_s1 + pop_s2

        return pop
