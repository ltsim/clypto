#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalEVO(VectorizeOptimizer):
    """
    The original version of: Energy Valley Optimizer (EVO)

    Links:
        1. https://www.nature.com/articles/s41598-022-27344-y
        2. https://www.mathworks.com/matlabcentral/fileexchange/123130-energy-valley-optimizer-a-novel-metaheuristic-algorithm

    Notes:
        1. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        2. The algorithm may not perform optimally due to slow convergence and no good operations, which could be improved by implementing better strategies and operations.
        3. The problem is that it is stuck at a local optimal around 1/2 of the max generations because fitness distance is being used as a factor in the equations.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import EVO    >>> import numpy as np
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
    >>> model = EVO.OriginalEVO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Azizi, M., Aickelin, U., A. Khorshidi, H., & Baghalzadeh Shishehgarkhaneh, M. (2023). Energy valley optimizer: a novel
    metaheuristic algorithm for global and engineering optimization. Scientific Reports, 13(1), 226.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _evolve(self, int epoch):
        # One or two candidates per agent depending on its branch (different draws): built agent
        # by agent from the unchanged population; evaluation and the merge are batched.
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t idx
        pos_list = np.array(pop.X)
        fit_list = np.array(pop.F)
        gb_fit = self.current_g_best().target.fitness
        gw_fit = self.g_worst.target.fitness
        x_avg_pop = np.mean(pos_list, axis=0)
        eb = np.mean(fit_list)
        gb_pos = np.array(self.g_best_x())
        pop_new = []
        for idx in range(0, self.pop_size):
            dis = np.sqrt(np.sum((pos_list[idx] - pos_list) ** 2, axis=1))
            idx_dis_sort = np.argsort(dis)
            CnPtIdx = self.generator.choice(list(set(range(2, self.pop_size)) - {idx}))
            x_team = pos_list[idx_dis_sort[1:CnPtIdx], :]
            x_avg_team = np.mean(x_team, axis=0)
            sl = (fit_list[idx] - gb_fit) / (gw_fit - gb_fit + self.EPSILON)

            pos_new1 = pos_list[idx].copy()
            pos_new2 = pos_list[idx].copy()
            if self._compare_fitness(eb, fit_list[idx], self.problem.sense):
                if self.generator.random() > sl:
                    a1_idx = self.generator.integers(self.problem.n_dims)
                    a2_idx = self.generator.integers(0, self.problem.n_dims, size=a1_idx)
                    pos_new1[a2_idx] = gb_pos[a2_idx]
                    g1_idx = self.generator.integers(self.problem.n_dims)
                    g2_idx = self.generator.integers(0, self.problem.n_dims, size=g1_idx)
                    pos_new2[g2_idx] = x_avg_team[g2_idx]
                else:
                    ir = self.generator.uniform(0, 1, 2)
                    jr = self.generator.uniform(0, 1, self.problem.n_dims)
                    pos_new1 += jr * (ir[0] * gb_pos - ir[1] * x_avg_pop) / sl
                    ir = self.generator.uniform(0, 1, 2)
                    jr = self.generator.uniform(0, 1, self.problem.n_dims)
                    pos_new2 += jr * (ir[0] * gb_pos - ir[1] * x_avg_team)
                pop_new.append(self._correct_solution(pos_new1))
                pop_new.append(self._correct_solution(pos_new2))
            else:
                pos_new = (
                        pos_new1
                        + self.generator.random()
                        * sl
                        * self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up, self.problem.n_dims)
                )
                pop_new.append(self._correct_solution(pos_new))
        merged = pop.concat(self.new_population(np.array(pop_new)))
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])
