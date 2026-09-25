#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.collection.evolutionary_based.GA.BaseGA cimport BaseGA
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class SingleGA(BaseGA):
    """
    The developed single-point mutation of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation (str): Optional, can be ["flip", "swap", "scramble", "inversion"] for one-point
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2

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
    >>> model = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection = "roulette", crossover = "uniform", mutation = "swap")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="swap")
    >>>
    >>> model5 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pc: float = 0.95,
        pm: float = 0.8,
        selection: str = "roulette",
        crossover: str = "uniform",
        mutation: str = "swap",
        k_way: float = 0.2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.8
            selection: Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            crossover: Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation: Optional, can be ["flip", "swap", "scramble", "inversion"], default="flip"
            k_way: Optional, set it when use "tournament" selection, default = 0.2
        """
        super().__init__(epoch, pop_size, pc, pm, name=name, mode=mode)
        self._params_name_ordered = tuple([
                "epoch",
                "pop_size",
                "pc",
                "pm",
                "selection",
                "crossover",
                "mutation",
                "k_way",
            ])
        self.sort_flag = False
        self.selection = cy.validator(str, selection, ["tournament", "random", "roulette"], "selection")
        self.crossover = cy.validator(str, crossover, ["one_point", "multi_points", "uniform", "arithmetic"], "crossover")
        self.mutation = cy.validator(str, mutation, ["flip", "swap", "scramble", "inversion"], "mutation")
        self.k_way = cy.validator(float, k_way, (0, 1.0), "k_way")

    def mutation_process__(self, child):
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
