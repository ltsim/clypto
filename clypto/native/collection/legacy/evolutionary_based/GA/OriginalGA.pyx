#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


class OriginalGA(_LegacyOptimizer):
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
    >>> from clypto.native.collection.legacy.evolutionary_based import GA    >>> import numpy as np
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
            **kwargs: object
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
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.selection = self.validator.check_str(
            "selection", selection, ["tournament", "random", "roulette"]
        )
        self.crossover = self.validator.check_str(
            "crossover",
            crossover,
            ["one_point", "multi_points", "uniform", "arithmetic"],
        )
        self.mutation_multipoints = self.validator.check_bool(
            "mutation_multipoints", mutation_multipoints
        )
        if self.mutation_multipoints:
            self.mutation = self.validator.check_str(
                "mutation", mutation, ["flip", "swap"]
            )
        else:
            self.mutation = self.validator.check_str(
                "mutation", mutation, ["flip", "swap", "scramble", "inversion"]
            )
        self.k_way = self.validator.check_float("k_way", k_way, (0, 1.0))
        self.set_parameters(
            [
                "epoch",
                "pop_size",
                "pc",
                "pm",
                "selection",
                "crossover",
                "mutation",
                "k_way",
                "mutation_multipoints",
            ]
        )
        self.sort_flag = False
