#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.native.collection.vectorize.evolutionary_based.GA.BaseGA cimport BaseGA
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class MultiGA(BaseGA):
    """
    The developed multipoints-mutation version of: Genetic Algorithm (GA)

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
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints

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
    >>> model = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection = "roulette", crossover = "uniform", mutation = "swap", k_way=0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="one_point", mutation="flip")
    >>>
    >>> model4 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation_multipoints=True, mutation="swap")
    >>>
    >>> model5 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="random", mutation="swap")
    >>>
    >>> model7 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="flip")

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
        selection: str = "roulette",
        crossover: str = "arithmetic",
        mutation: str = "flip",
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
            pm: mutation probability, default = 0.025
            selection: Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            crossover: Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation: Optional, can be ["flip", "swap"] for multipoints
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
        self.selection = cy.validator(str, selection, ["tournament", "random", "roulette"], "selection")
        self.crossover = cy.validator(str, crossover, ["one_point", "multi_points", "uniform", "arithmetic"], "crossover")
        self.mutation = cy.validator(str, mutation, ["flip", "swap"], "mutation")
        self.k_way = cy.validator(float, k_way, (0, 1.0), "k_way")

    def mutation_batch__(self, child, multipoints=None):
        return BaseGA.mutation_batch__(self, child, True)
