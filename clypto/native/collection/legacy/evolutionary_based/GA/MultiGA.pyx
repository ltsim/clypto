#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.native.collection.legacy.evolutionary_based.GA.BaseGA import BaseGA


class MultiGA(BaseGA):
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
    >>> from clypto.native.collection.legacy.evolutionary_based import GA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
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
            **kwargs: object
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
        super().__init__(epoch, pop_size, pc, pm, **kwargs)
        self.selection = self.validator.check_str(
            "selection", selection, ["tournament", "random", "roulette"]
        )
        self.crossover = self.validator.check_str(
            "crossover",
            crossover,
            ["one_point", "multi_points", "uniform", "arithmetic"],
        )
        self.mutation = self.validator.check_str("mutation", mutation, ["flip", "swap"])
        self.k_way = self.validator.check_float("k_way", k_way, (0, 1.0))
        self._set_parameters(
            [
                "epoch",
                "pop_size",
                "pc",
                "pm",
                "selection",
                "crossover",
                "mutation",
                "k_way",
            ]
        )

    def mutation_process__(self, child):
        """
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
        + Mutated on the whole vector is effected by parameter: pm
            + flip --> (default in this case) should set the pm small such as: [0.01 -> 0.2]
            + swap --> should set the pm small such as: [0.01 -> 0.2]

        Args:
            child (np.array): The position of the child

        Returns:
            np.array: The mutated vector of the child
        """
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
