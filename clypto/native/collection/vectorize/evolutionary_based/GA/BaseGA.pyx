#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class BaseGA(AgentListOptimizer):
    """
    The original version of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = True
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import GA    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation_multipoints=True, mutation="swap")
    >>>
    >>> model5 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pc: float = 0.95,
        pm: float = 0.025,
        *,
        selection: str | None = None,
        k_way: float | None = None,
        crossover: str | None = None,
        mutation_multipoints: bool | None = None,
        mutation: str | None = None,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pc", "pm"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pc = cy.validator(float, pc, (0, 1.0), "pc")
        self.pm = cy.validator(float, pm, (0, 1.0), "pm")
        self.selection = "tournament"
        self.k_way = 0.2
        self.crossover = "uniform"
        self.mutation = "flip"
        self.mutation_multipoints = True

        if selection is not None:
            self.selection = cy.validator(str, selection, ["tournament", "random", "roulette"], "selection")
        if k_way is not None:
            self.k_way = cy.validator(float, k_way, (0, 1.0), "k_way")
        if crossover is not None:
            self.crossover = cy.validator(str, crossover, ["one_point", "multi_points", "uniform", "arithmetic"], "crossover")
        if mutation_multipoints is not None:
            self.mutation_multipoints = cy.validator(bool, mutation_multipoints, None, "mutation_multipoints")
        if mutation is not None:
            if self.mutation_multipoints:
                self.mutation = cy.validator(str, mutation, ["flip", "swap"], "mutation")
            else:
                self.mutation = cy.validator(str, mutation, ["flip", "swap", "scramble", "inversion"], "mutation")

    def selection_process__(self, list_fitness):
        if self.selection == "roulette":
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            if id_c2 == id_c1:
                # Fall back to a uniform pick among the remaining indices instead of
                # retrying roulette selection, which can loop forever once floating-point
                # underflow drives every other candidate's probability to exactly 0.0.
                others = [i for i in range(len(list_fitness)) if i != id_c1]
                id_c2 = self.generator.choice(others)
        elif self.selection == "random":
            id_c1, id_c2 = self.generator.choice(range(self.pop_size), 2, replace=False)
        else:  ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(
                self.objs, k_way=self.k_way, output=2
            )
        return self.objs[id_c1].solution, self.objs[id_c2].solution

    def selection_process_00__(self, pop_selected):
        if self.selection == "roulette":
            list_fitness = np.array([agent.target.fitness for agent in pop_selected])
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            if id_c2 == id_c1:
                # Fall back to a uniform pick among the remaining indices instead of
                # retrying roulette selection, which can loop forever once floating-point
                # underflow drives every other candidate's probability to exactly 0.0.
                others = [i for i in range(len(list_fitness)) if i != id_c1]
                id_c2 = self.generator.choice(others)
        elif self.selection == "random":
            id_c1, id_c2 = self.generator.choice(
                range(len(pop_selected)), 2, replace=False
            )
        else:  ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(
                pop_selected, k_way=self.k_way, output=2
            )
        return pop_selected[id_c1].solution, pop_selected[id_c2].solution

    def selection_process_01__(self, pop_dad, pop_mom):
        if self.selection == "roulette":
            list_fit_dad = np.array([agent.target.fitness for agent in pop_dad])
            list_fit_mom = np.array([agent.target.fitness for agent in pop_mom])
            id_c1 = self.get_index_roulette_wheel_selection(list_fit_dad)
            id_c2 = self.get_index_roulette_wheel_selection(list_fit_mom)
        elif self.selection == "random":
            id_c1 = self.generator.choice(range(len(pop_dad)))
            id_c2 = self.generator.choice(range(len(pop_mom)))
        else:  ## tournament
            id_c1 = self.get_index_kway_tournament_selection(
                pop_dad, k_way=self.k_way, output=1
            )[0]
            id_c2 = self.get_index_kway_tournament_selection(
                pop_mom, k_way=self.k_way, output=1
            )[0]
        return pop_dad[id_c1].solution, pop_mom[id_c2].solution

    def crossover_process__(self, dad, mom):
        if self.crossover == "arithmetic":
            w1, w2 = self.crossover_arithmetic(dad, mom)
        elif self.crossover == "one_point":
            cut = self.generator.integers(1, self.problem.n_dims - 1)
            w1 = np.concatenate([dad[:cut], mom[cut:]])
            w2 = np.concatenate([mom[:cut], dad[cut:]])
        elif self.crossover == "multi_points":
            idxs = self.generator.choice(
                range(1, self.problem.n_dims - 1), 2, replace=False
            )
            cut1, cut2 = np.min(idxs), np.max(idxs)
            w1 = np.concatenate([dad[:cut1], mom[cut1:cut2], dad[cut2:]])
            w2 = np.concatenate([mom[:cut1], dad[cut1:cut2], mom[cut2:]])
        else:  # uniform
            flip = self.generator.integers(0, 2, self.problem.n_dims)
            w1 = dad * flip + mom * (1 - flip)
            w2 = mom * flip + dad * (1 - flip)
        return w1, w2

    def mutation_process__(self, child):

        if self.mutation_multipoints:
            if self.mutation == "swap":
                for idx in range(self.problem.n_dims):
                    idx_swap = self.generator.choice(
                        list(set(range(0, self.problem.n_dims)) - {idx})
                    )
                    child[idx], child[idx_swap] = child[idx_swap], child[idx]
                    return child
            else:  # "flip"
                mutation_child = self.problem.generate_solution()
                flag_child = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
                return np.where(flag_child, mutation_child, child)
        else:
            if self.mutation == "swap":
                idx1, idx2 = self.generator.choice(
                    range(0, self.problem.n_dims), 2, replace=False
                )
                child[idx1], child[idx2] = child[idx2], child[idx1]
                return child
            elif self.mutation == "inversion":
                cut1, cut2 = self.generator.choice(
                    range(0, self.problem.n_dims), 2, replace=False
                )
                temp = child[cut1:cut2]
                temp = temp[::-1]
                child[cut1:cut2] = temp
                return child
            elif self.mutation == "scramble":
                cut1, cut2 = self.generator.choice(
                    range(0, self.problem.n_dims), 2, replace=False
                )
                temp = child[cut1:cut2]
                self.generator.shuffle(temp)
                child[cut1:cut2] = temp
                return child
            else:  # "flip"
                idx = self.generator.integers(0, self.problem.n_dims)
                child[idx] = self.generator.uniform(
                    self.problem.lb[idx], self.problem.ub[idx]
                )
                return child

    def survivor_process__(self, pop, pop_child):
        pop_new = []
        for idx in range(0, self.pop_size):
            id_child = self.get_index_kway_tournament_selection(
                pop, k_way=0.1, output=1, reverse=True
            )[0]
            agent_x = pop_child[idx]
            agent_y = pop[id_child]
            pop_new.append(self.get_better_agent(agent_x, agent_y, self.problem.minmax))
        return pop_new

    def evolve_agents(self, epoch):
        list_fitness = np.array([agent.target.fitness for agent in self.objs])
        pop_new = []
        for i in range(0, -(-self.pop_size // 2)):  # ceil division, safe for odd pop_size
            ### Selection
            child1, child2 = self.selection_process__(list_fitness)

            ### Crossover
            if self.generator.random() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)

            ### Mutation
            child1 = self.mutation_process__(child1)
            child2 = self.mutation_process__(child2)

            child1 = self.correct_solution(child1)
            child2 = self.correct_solution(child2)

            agent1 = self.generate_empty_agent(child1)
            agent2 = self.generate_empty_agent(child2)

            pop_new.append(agent1)
            pop_new.append(agent2)

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-2].target = self.get_target(child1)
                pop_new[-1].target = self.get_target(child2)
        pop_new = pop_new[: self.pop_size]
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        ### Survivor Selection
        self.objs = self.survivor_process__(self.objs, pop_new)
