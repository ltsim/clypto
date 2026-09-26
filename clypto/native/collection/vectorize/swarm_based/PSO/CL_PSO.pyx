#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np

from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget
from clypto.native.collection.vectorize.swarm_based.PSO._base cimport _PSOBase


cdef class CL_PSO(_PSOBase):
    """
    The original version of: Comprehensive Learning Particle Swarm Optimization (CL-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_local (float): [1.0, 3.0], local coefficient, default = 1.2
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.7, 2.0], Weight max of bird, default = 0.9
        + max_flag (int): [5, 20], Number of times, default = 7

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.CL_PSO(epoch=1000, pop_size=50, c_local = 1.2, w_min=0.4, w_max=0.9, max_flag = 7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer
    for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
    """

    cdef public double c_local
    cdef public double w_min
    cdef public double w_max
    cdef public int max_flag
    cdef public object flags

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c_local: float = 1.2,
        w_min: float = 0.4,
        w_max: float = 0.9,
        max_flag: int = 7,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c_local: local coefficient, default = 1.2
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
            max_flag: Number of times, default = 7
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c_local", "w_min", "w_max", "max_flag"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c_local = cy.validator(float, c_local, (0, 5.0), "c_local")
        self.w_min = cy.validator(float, w_min, (0, 0.5), "w_min")
        self.w_max = cy.validator(float, w_max, [0.5, 2.0], "w_max")
        self.max_flag = cy.validator(int, max_flag, [2, 100], "max_flag")

    def _initialize_variables(self):
        self.v_max = 0.5 * (self.problem.bounds.up - self.problem.bounds.low)
        self.v_min = -self.v_max
        self.flags = np.zeros(self.pop_size)

    cdef void _bump_flag(self, Py_ssize_t idx):
        self.flags[idx] += 1
        if self.flags[idx] >= self.max_flag:
            self.flags[idx] = 0

    def _evolve(self, int epoch):
        # Sequential: agents compare against pop[id1]/pop[id2] as already updated
        # in this epoch, so they are processed one by one on the buffer rows.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation new = pop.empty_like()
        cdef NativeTarget target
        cdef double[:, ::1] buf = pop.view
        cdef Py_ssize_t idx, jdx, id1, id2, n = pop.n, d = pop.d
        cdef Py_ssize_t cX = pop.cX, cF = pop.cF, cV = self.cV, cP = self.cP
        cdef double wk = self.w_max * (epoch / <double>self.epoch) * (self.w_max - self.w_min)
        cdef bint swarm = self.mode not in self.AVAILABLE_MODES
        cdef bint better
        sense = self.problem.sense
        others = [list(set(range(0, self.pop_size)) - {i}) for i in range(n)]
        for idx in range(n):
            pci = 0.05 + 0.45 * (np.exp(10 * (idx + 1) / self.pop_size) - 1) / (np.exp(10) - 1)
            vec_new = pop.buf[idx, cV:cV + d].copy()
            for jdx in range(d):
                if self.generator.random() > pci:
                    vj = wk * buf[idx, cV + jdx] + self.c_local * self.generator.random() * (
                        buf[idx, cP + jdx] - buf[idx, cX + jdx]
                    )
                else:
                    id1, id2 = self.generator.choice(others[idx], 2, replace=False)
                    better = buf[id1, cF] < buf[id2, cF]
                    if sense != "min":
                        better = not better
                    if not better:
                        id1 = id2
                    vj = wk * buf[idx, cV + jdx] + self.c_local * self.generator.random() * (
                        buf[id1, cP + jdx] - buf[idx, cX + jdx]
                    )
                vec_new[jdx] = vj
            vec_new = np.clip(vec_new, self.v_min, self.v_max)
            pos_new = self._correct_solution(pop.buf[idx, cX:cX + d] + vec_new)
            pos_new = self._correct_solution(pos_new)
            # generate_empty_agent(pos_new): fresh velocity, personal best = itself.
            new.X[idx] = pos_new
            new.field("V")[idx] = self.generator.uniform(-self.v_max, self.v_max)
            new.field("P")[idx] = pos_new
            if swarm:
                target = self._get_target(pos_new)
                new.O[idx] = target.objectives
                new.F[idx] = target.fitness
                new.field("PO")[idx] = target.objectives
                new.field("PF")[idx] = target.fitness
                # get_better_agent(current, new): ties go to the new agent for
                # "min" and to the current one for "max".
                if sense == "min":
                    better = not (pop.F[idx] < target.fitness)
                else:
                    better = pop.F[idx] < target.fitness
                if better:
                    pop.buf[idx] = new.buf[idx]
                if cy.compare_target(target, pop.target_at(idx, self.cPO), sense):
                    pop.field("P")[idx] = pos_new
                    pop.field("PO")[idx] = target.objectives
                    pop.field("PF")[idx] = target.fitness
                    self.flags[idx] = 0
                else:
                    self._bump_flag(idx)
        if not swarm:
            self.evaluate(new, 0, n)
            child = pop.empty_like()
            for idx in range(n):
                if sense == "min":
                    better = new.F[idx] < pop.F[idx]
                else:
                    better = new.F[idx] > pop.F[idx]
                child.buf[idx] = new.buf[idx] if better else pop.buf[idx]
                # Compared with the *old* agent's personal best, as the classic code.
                if cy.compare_target(new.target_at(idx, new.cO), pop.target_at(idx, self.cPO), sense):
                    child.field("P")[idx] = new.X[idx]
                    child.field("PO")[idx] = new.O[idx]
                    child.field("PF")[idx] = new.F[idx]
                    self.flags[idx] = 0
                else:
                    self._bump_flag(idx)
            self.pop = child
