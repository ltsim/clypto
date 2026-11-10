#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Per-particle state operations shared by every legacy PSO variant."""
import numpy as np
cimport numpy as cnp
cimport clypto.core as cy

from clypto.optimizer.native.agent cimport Agent


cpdef void update_velocity(Agent agent, object gbest_solution, object w,
                           double c1, double c2, object generator,
                           object v_min=None, object v_max=None):
    """Inertia + pull towards pbest (cognitive) + pull towards gbest (social), clipped when limits are given."""
    cdef Py_ssize_t dims = agent.solution.shape[0]
    cognitive = c1 * generator.random(dims) * (agent.pbest_solution - agent.solution)
    social = c2 * generator.random(dims) * (gbest_solution - agent.solution)
    velocity = w * agent.velocity + cognitive + social
    agent.velocity = velocity if v_min is None else np.clip(velocity, v_min, v_max)


cpdef object move(Agent agent):
    """The position the velocity leads to; the optimizer decides whether the particle goes there."""
    return agent.solution + agent.velocity


cpdef bint update_pbest(Agent agent, Agent candidate, str sense):
    """Keep ``candidate`` as the personal best when it beats it (always, the first time)."""
    if agent.pbest_solution is not None and not cy.better_fitness(candidate.fitness, agent.pbest_fitness, sense):
        return False
    agent.pbest_solution = candidate.solution.copy()
    agent.pbest_fitness = candidate.fitness
    return True
