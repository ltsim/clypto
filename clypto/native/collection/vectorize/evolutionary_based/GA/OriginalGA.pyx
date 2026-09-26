#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalGA(AgentListOptimizer):
    """
    The fully tuned version of: Genetic Algorithm (GA)

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
    >>> from clypto.native.collection.vectorize.evolutionary_based import GA    >>> import numpy as np
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
    >>> model = GA.OriginalGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05)
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

    cdef public object pc
    cdef public object pm
    cdef public object selection
    cdef public object crossover
    cdef public object mutation_multipoints
    cdef public object k_way
    cdef public object mutation

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pc: float = 0.95,
        pm: float = 0.025,
        selection: str = "tournament",
        crossover: str = "uniform",
        mutation: str = "flip",
        k_way: float = 0.2,
        mutation_multipoints: bool = True,
        *,
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
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pc", "pm", "selection", "crossover", "mutation", "k_way", "mutation_multipoints"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pc = cy.validator(float, pc, (0, 1.0), "pc")
        self.pm = cy.validator(float, pm, (0, 1.0), "pm")
        self.selection = cy.validator(str, selection, ["tournament", "random", "roulette"], "selection")
        self.crossover = cy.validator(str, crossover, ["one_point", "multi_points", "uniform", "arithmetic"], "crossover")
        self.mutation_multipoints = cy.validator(bool, mutation_multipoints, None, "mutation_multipoints")
        if self.mutation_multipoints:
            self.mutation = cy.validator(str, mutation, ["flip", "swap"], "mutation")
        else:
            self.mutation = cy.validator(str, mutation, ["flip", "swap", "scramble", "inversion"], "mutation")
        self.k_way = cy.validator(float, k_way, (0, 1.0), "k_way")
