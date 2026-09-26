#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.legacy.human_based.QSA.OppoQSA import OppoQSA
from clypto.native.collection.legacy.human_based.QSA.LevyQSA import LevyQSA


class ImprovedQSA(OppoQSA, LevyQSA):
    """
    The original version of: Improved Queuing Search Algorithm (QSA)

    Links:
       1. https://doi.org/10.1007/s12652-020-02849-4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import QSA    >>> import numpy as np
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
    >>> model = QSA.ImprovedQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, B.M., Hoang, B., Nguyen, T. and Nguyen, G., 2021. nQSV-Net: a novel queuing search variant for
    global space search and workload modeling. Journal of Ambient Intelligence and Humanized Computing, 12(1), pp.27-46.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.pop, epoch)
        pop = self.update_business_2__(pop, epoch)
        pop = self.update_business_3__(pop, self.g_best)
        self.pop = self.opposition_based__(pop, self.g_best)
