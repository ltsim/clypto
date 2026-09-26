#!/usr/bin/env python
# Created by "Thieu" at 00:27, 18/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevTPO(VectorizeOptimizer):
    """
    The original version: Tree Physiology Optimization (TPO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/63982-tree-physiology-optimization-tpo-algorithm-for-stochastic-test-function-optimization

    Notes:
        1. The paper is difficult to read and understand, and the provided MATLAB code is also challenging to understand.
        2. Based on my idea:
            + pop_size = number of branhes, the population size should be equal to the number of branches.
            + The number of leaves should be calculated as int(sqrt(pop_size) + 1), so we don't need to specify the n_leafs parameter, which will also reduce computation time.
            + When using this algorithm, especially when setting stopping conditions, be careful and set it to the FE type.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [-10, 10.] -> better [0.2, 0.5], Absorption constant for tree root elongation, default = 0.5
        + beta (float): [-100, 100.] -> better [10, 50], Diversification facor of tree shoot, default=50.
        + theta (float): (0, 1.0] -> better [0.5, 0.9], Factor to reduce randomization, Theta = Power law to reduce randomization as iteration increases, default=0.9

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import TPO    >>> import numpy as np
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
    >>> model = TPO.DevTPO(epoch=1000, pop_size=50, alpha = 0.3, beta = 50., theta = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Halim, A. H., & Ismail, I. (2017). Tree physiology optimization in benchmark function and
    traveling salesman problem. Journal of Intelligent Systems, 28(5), 849-871.
    """

    cdef public object alpha
    cdef public object beta
    cdef public object theta
    cdef public object n_leafs
    cdef public object _theta
    cdef public object roots
    cdef public object pop_total

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 0.3,
        beta: float = 50.0,
        theta: float = 0.9,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): Absorption constant for tree root elongation, default=0.3
            beta (float): Diversification factor of tree shoot, default=50.
            theta (float): Factor to reduce randomization, Theta = Power law to reduce randomization as iteration increases, default=0.9
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "alpha", "beta", "theta"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.alpha = cy.validator(float, alpha, [-10.0, 10.0], "alpha")
        self.beta = cy.validator(float, beta, [-100.0, 100], "beta")
        self.theta = cy.validator(float, theta, (0, 1.0), "theta")

    def _initialize_variables(self):
        self.n_leafs = int(np.sqrt(self.pop_size) + 1)  # Number of leafs
        self._theta = self.theta
        self.roots = self.generator.uniform(0, 1, (self.n_leafs, self.problem.n_dims))

    def _initialization(self):
        cdef NativePopulation leafs, pop
        self.pop_total = []
        pop = None
        for idx in range(self.pop_size):
            leafs = self.generate_population(self.n_leafs)
            best = leafs.take([self.sorted_order(leafs)[0]])  # The best leaf in each branches
            pop = best if pop is None else pop.concat(best)
            self.pop_total.append(leafs)
        self.pop = pop

    def _evolve(self, int epoch_c):
        cdef NativePopulation leafs, cand, pop
        cdef Py_ssize_t idx
        g_best = np.array(self.g_best_x())
        for idx in range(0, self.pop_size):
            leafs = self.pop_total[idx]
            pos_list = np.array(leafs.X)
            carbon_gain = self._theta * g_best - pos_list
            roots_old = np.copy(self.roots)
            self.roots += (
                    self.alpha
                    * carbon_gain
                    * self.generator.uniform(-0.5, 0.5, (self.n_leafs, self.problem.n_dims))
            )
            nutrient_value = self._theta * (self.roots - roots_old)
            pos_list_new = g_best + self.beta * nutrient_value
            cand = leafs.empty_like()
            cand.X[:] = self._correct_solution(pos_list_new)
            self.evaluate(cand, 0, cand.n)
            if self.mode in self.AVAILABLE_MODES:
                # greedy_selection_population(candidates, leafs): a leaf stays only if strictly better
                keep = leafs.F < cand.F if self.problem.sense == "min" else leafs.F > cand.F
                rows = np.flatnonzero(~keep)
                leafs.buf[rows] = cand.buf[rows]
            else:
                ops.accept(self, cand, dst=leafs)
        self._theta = self._theta * self.theta
        pop = None
        for idx in range(0, self.pop_size):
            leafs = self.pop_total[idx]
            best = leafs.take([self.sorted_order(leafs)[0]])
            pop = best if pop is None else pop.concat(best)
        self.pop = pop
