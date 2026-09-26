#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget
from clypto.optimizer.native.agent cimport LegacyNativeAgent


cdef class OriginalBCO(VectorizeOptimizer):
    """
    The original version of: Bacterial Colony Optimization (BCO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/4475427

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import BCO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = BCO.OriginalBCO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Niu, B., & Wang, H. (2012). Bacterial colony optimization. Discrete dynamics in nature and society, 2012(1), 698057.
    """

    cdef public object c_min
    cdef public object c_max
    cdef public object n_chemotaxis
    cdef public object max_swim_steps
    cdef public object energy_threshold
    cdef public object migration_prob
    cdef public object energy
    cdef public object pop_local
    cdef public object objs

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c_min: float = 0.01,
        c_max: float = 0.2,
        n_chemotaxis: int = 1,
        max_swim_steps: int = 4,
        energy_threshold: float = 0.5,
        migration_prob: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            c_min: Minimum chemotaxis step size
            c_max: Maximum chemotaxis step size
            n_chemotaxis: Nonlinear parameter for chemotaxis step
            max_swim_steps: Maximum swimming steps
            energy_threshold: Energy threshold for reproduction/elimination
            migration_prob: Migration probability
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "c_min",
                "c_max",
                "n_chemotaxis",
                "max_swim_steps",
                "energy_threshold",
                "migration_prob",
            ],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c_min = cy.validator(float, c_min, (0.0, 1.0), "c_min")
        self.c_max = cy.validator(int, c_max, (c_min, 10.0), "c_max")
        self.n_chemotaxis = cy.validator(int, n_chemotaxis, [1, 5], "n_chemotaxis")
        self.max_swim_steps = cy.validator(int, max_swim_steps, (2, 10), "max_swim_steps")
        self.energy_threshold = cy.validator(float, energy_threshold, (0, 1.0), "energy_threshold")
        self.migration_prob = cy.validator(float, migration_prob, (0, 1.0), "migration_prob")

    def mirror__(self):
        """Copy the agents into the population the engine sorts and reports."""
        cdef NativePopulation pop = self.pop.take(np.zeros(len(self.objs), dtype=int))
        for i, agent in enumerate(self.objs):
            ops.set_row(pop, i, agent.solution, agent.target)
        return pop

    def _initialize_variables(self):
        self.energy = self.generator.uniform(0, 1, self.pop_size)

    def _initialization(self):
        # The classic algorithm shares agents (and their solution arrays) between its lists and edits
        # them in place, so it keeps agent objects and mirrors them into ``self.pop``.
        cdef NativePopulation base
        VectorizeOptimizer._initialization(self)
        base = self.pop
        self.objs = [base.agent(i) for i in range(base.n)]
        self.pop_local = list(self.objs)  # pop.copy(): the same agents

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativeTarget tar
        cdef Py_ssize_t idx, jdx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        minimize = self.problem.sense == "min"
        pop = self.objs
        # g_best is one of the agents (aliased) once the first epoch is over; before, a copy that shares its array
        if self._g_best_row >= 0:
            g_best = pop[self._g_best_row]
        else:
            best = pop[self.sorted_order(self.pop)[0]]
            g_best = LegacyNativeAgent(best.solution, best.target.copy())

        # Calculate adaptive chemotaxis step size
        step = (
                self.c_min
                + (self.c_max - self.c_min) * (<object>(1 - epoch / self.epoch)) ** self.n_chemotaxis
        )
        pop_new = []
        ## Perform chemotaxis and communication
        for idx in range(0, self.pop_size):
            # Random factor for personal vs global best
            f_i = self.generator.random()
            personal_direction = self.pop_local[idx].solution - pop[idx].solution
            global_direction = g_best.solution - pop[idx].solution

            # Tumbling (with random turbulence)
            turbulent = self.generator.normal(0, 0.1, self.problem.n_dims)
            pos_new = f_i * global_direction + (1 - f_i) * personal_direction + turbulent
            for jdx in range(0, self.max_swim_steps):
                # Swimming (no turbulence)
                pos_new = f_i * global_direction + (1 - f_i) * personal_direction
            pos_new = pop[idx].solution + step * pos_new
            pos_new = self._correct_solution(pos_new)
            agent_new = LegacyNativeAgent(pos_new, None)
            pop_new.append(agent_new)
            if not swarm:
                agent_new.target = self._get_target(pos_new)
                # get_better_agent(old, new): copies of the winner
                old = pop[idx]
                if minimize:
                    pop[idx] = old.copy() if old.target.fitness < agent_new.target.fitness else agent_new.copy()
                else:
                    pop[idx] = agent_new.copy() if old.target.fitness < agent_new.target.fitness else old.copy()
        if swarm:
            for agent in pop_new:
                agent.target = self._get_target(agent.solution, counted=False)
            self._nfe_counter += len(pop_new)
            if minimize:
                pop = [pop_new[i] if pop_new[i].target.fitness < pop[i].target.fitness else pop[i] for i in range(len(pop))]
            else:
                pop = [pop_new[i] if pop_new[i].target.fitness > pop[i].target.fitness else pop[i] for i in range(len(pop))]

        ## Perform interactive exchange between bacteria
        for idx in range(0, self.pop_size):
            exchange_type = self.generator.choice(["individual", "group"])
            if exchange_type == "individual":
                if self.generator.random() < 0.5:
                    # Dynamic neighbor oriented
                    if idx == 0:
                        neighbor = 1
                    elif idx == self.pop_size - 1:
                        neighbor = self.pop_size - 2
                    else:
                        neighbor = idx + 1 if self.generator.random() < 0.5 else idx - 1
                else:
                    # Random oriented
                    neighbor = self.generator.choice(list(set(range(self.pop_size)) - {idx}))

                # Exchange information if neighbor is better
                if pop[neighbor].target.fitness < pop[idx].target.fitness:
                    pop[idx] = pop[neighbor]
            else:
                # Group exchange
                if pop[idx].target.fitness < g_best.target.fitness:
                    # Move towards global best
                    pop[idx].solution += 0.1 * (g_best.solution - pop[idx].solution)
        if swarm:
            for agent in pop:
                agent.target = self._get_target(agent.solution, counted=False)
            self._nfe_counter += len(pop)
        self.objs = pop
        self.pop = self.mirror__()
