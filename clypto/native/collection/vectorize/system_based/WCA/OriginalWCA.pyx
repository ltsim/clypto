#!/usr/bin/env python
# Created by "Thieu" at 09:55, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalWCA(VectorizeOptimizer):
    """
    The original version of: Water Cycle Algorithm (WCA)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2012.07.010

    Notes
    ~~~~~
    The ideas are (almost the same as ICO algorithm):
        + 1 sea is global best solution
        + a few river which are second, third, ...
        + other left are stream (will flow directed to sea or river)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + nsr (int): [4, 10], Number of rivers + sea (sea = 1), default = 4
        + wc (float): [1.0, 3.0], Weighting coefficient (C in the paper), default = 2
        + dmax (float): [1e-6], fixed parameter, Evaporation condition constant, default=1e-6

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.system_based import WCA    >>> import numpy as np
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
    >>> model = WCA.OriginalWCA(epoch=1000, pop_size=50, nsr = 4, wc = 2.0, dmax = 1e-6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Eskandar, H., Sadollah, A., Bahreininejad, A. and Hamdi, M., 2012. Water cycle algorithm–A novel metaheuristic
    optimization method for solving constrained engineering optimization problems. Computers & Structures, 110, pp.151-166.
    """

    cdef public int nsr
    cdef public double wc
    cdef public double dmax
    cdef public object ecc
    cdef public object pop_best
    cdef public object pop_stream
    cdef public object streams

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        nsr: int = 4,
        wc: float = 2.0,
        dmax: float = 1e-6,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            nsr (int): Number of rivers + sea (sea = 1), default = 4
            wc (float): Weighting coefficient (C in the paper), default = 2.0
            dmax (float): Evaporation condition constant, default=1e-6
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "nsr", "wc", "dmax"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.nsr = cy.validator(int, nsr, [2, int(self.pop_size / 2)], "nsr")
        self.wc = cy.validator(float, wc, (1.0, 3.0), "wc")
        self.dmax = cy.validator(float, dmax, (0, 1.0), "dmax")

    def _initialization(self):
        VectorizeOptimizer._initialization(self)
        cdef NativePopulation pop = self.pop
        pop = pop.take(self.sorted_order(pop))
        self.pop = pop
        self.ecc = self.dmax  # Evaporation condition constant - variable
        n_stream = self.pop_size - self.nsr
        self.pop_best = pop.take(np.arange(self.nsr))  # Including sea and river (1st solution is sea)
        pop_stream = pop.take(np.arange(self.nsr, pop.n))  # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = np.array(self.pop_best.F)
        num_child_in_river_list = np.round(np.abs(cost_river_list / np.sum(cost_river_list)) * n_stream).astype(int)
        if np.sum(num_child_in_river_list) < n_stream:
            num_child_in_river_list[-1] += n_stream - np.sum(num_child_in_river_list)
        streams = {}
        idx_already_selected = []
        for i in range(0, self.nsr - 1):
            idx_list = self.generator.choice(
                list(set(range(0, n_stream)) - set(idx_already_selected)),
                num_child_in_river_list[i],
                replace=False,
            ).tolist()
            idx_already_selected += idx_list
            streams[i] = pop_stream.take(idx_list)
        idx_last = list(set(range(0, n_stream)) - set(idx_already_selected))
        streams[self.nsr - 1] = pop_stream.take(idx_last)
        self.streams = streams

    def _evolve(self, int epoch):
        # Rivers, streams and the sea live in their own sub-populations (row blocks); the
        # streams of a river are replaced by their moved versions, without selection.
        cdef NativePopulation river = self.pop_best
        cdef NativePopulation stream, stream_new, merged
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        sense = self.problem.sense
        g_best = np.array(self.g_best_x())
        # Update stream and river
        for idx in range(self.nsr):
            stream = self.streams[idx]
            # Update stream
            u = self.generator.random(stream.n)  # one uniform() per stream
            pos_new = stream.X + (u[:, None] * self.wc) * (river.X[idx] - stream.X)
            stream_new = self.new_population(self._correct_solution(pos_new))
            self.streams[idx] = stream_new
            best = self.sorted_order(stream_new)[0]
            if self._compare_fitness(stream_new.F[best], river.F[idx], sense):
                river.buf[idx] = stream_new.buf[best]
            # Update river
            pos_new = river.X[idx] + self.generator.uniform() * self.wc * (g_best - river.X[idx])
            pos_new = self._correct_solution(pos_new)
            tar = self._get_target(pos_new)
            if self._compare_fitness(tar.fitness, river.F[idx], sense):
                ops.set_row(river, idx, pos_new, tar)
        # Evaporation
        for idx in range(1, self.nsr):
            distance = np.sqrt(np.sum((g_best - river.X[idx]) ** 2))
            if distance < self.ecc or self.generator.random() < 0.1:
                merged = self.streams[idx].concat(self.new_population(self.problem.generate_solution(True)[None]))
                order = self.sorted_order(merged)
                river.buf[idx] = merged.buf[order[0]]
                self.streams[idx] = merged.take(order[1:])
        merged = river
        for idx in range(self.nsr):
            merged = merged.concat(self.streams[idx])
        self.pop = merged
        # Reduce the ecc
        self.ecc = self.ecc - self.ecc / self.epoch
