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
    cdef Py_ssize_t dims, i
    cdef double w_scalar = 0.0, x
    cdef bint w_is_scalar
    cdef cnp.ndarray vel, pbest, sol, gbest, r1, r2, out, wo, vmin, vmax
    cdef double* vp
    cdef double* pbp
    cdef double* sp
    cdef double* gp
    cdef double* r1p
    cdef double* r2p
    cdef double* op
    cdef double* wp = NULL
    sol = agent.solution
    vel = agent.velocity
    pbest = agent.pbest_solution
    gbest = gbest_solution
    dims = agent.solution.shape[0]
    w_is_scalar = isinstance(w, (float, int))
    if (w_is_scalar or (type(w) is np.ndarray and w.ndim == 1 and w.dtype == np.float64
                              and cnp.PyArray_ISCARRAY_RO(w) and w.shape[0] == dims)) \
            and type(sol) is np.ndarray and type(vel) is np.ndarray \
            and type(pbest) is np.ndarray and type(gbest) is np.ndarray \
            and sol.ndim == 1 and vel.ndim == 1 and pbest.ndim == 1 and gbest.ndim == 1 \
            and sol.dtype == np.float64 and vel.dtype == np.float64 \
            and pbest.dtype == np.float64 and gbest.dtype == np.float64 \
            and cnp.PyArray_ISCARRAY_RO(sol) and cnp.PyArray_ISCARRAY_RO(vel) \
            and cnp.PyArray_ISCARRAY_RO(pbest) and cnp.PyArray_ISCARRAY_RO(gbest) \
            and vel.shape[0] == dims and pbest.shape[0] == dims and gbest.shape[0] == dims:
        # Two ``random(dims)`` draws in the same order as the NumPy expression:
        # one uniform per dimension, sequential in the seeded stream.
        r1 = generator.random(dims)
        r2 = generator.random(dims)
        out = np.empty(dims, dtype=np.float64)
        sp = <double*>cnp.PyArray_DATA(sol)
        vp = <double*>cnp.PyArray_DATA(vel)
        pbp = <double*>cnp.PyArray_DATA(pbest)
        gp = <double*>cnp.PyArray_DATA(gbest)
        r1p = <double*>cnp.PyArray_DATA(r1)
        r2p = <double*>cnp.PyArray_DATA(r2)
        op = <double*>cnp.PyArray_DATA(out)
        if w_is_scalar:
            w_scalar = w
            for i in range(dims):
                # (w * velocity + cognitive) + social, as the NumPy expression groups it
                op[i] = (w_scalar * vp[i] + (c1 * r1p[i]) * (pbp[i] - sp[i])) \
                    + (c2 * r2p[i]) * (gp[i] - sp[i])
        else:
            wo = <cnp.ndarray>w
            wp = <double*>cnp.PyArray_DATA(wo)
            for i in range(dims):
                op[i] = (wp[i] * vp[i] + (c1 * r1p[i]) * (pbp[i] - sp[i])) \
                    + (c2 * r2p[i]) * (gp[i] - sp[i])
        if v_min is not None:
            if (type(v_min) is np.ndarray and type(v_max) is np.ndarray
                    and v_min.ndim == 1 and v_max.ndim == 1
                    and v_min.dtype == np.float64 and v_max.dtype == np.float64
                    and cnp.PyArray_ISCARRAY_RO(v_min) and cnp.PyArray_ISCARRAY_RO(v_max)
                    and v_min.shape[0] == dims and v_max.shape[0] == dims):
                vmin = v_min
                vmax = v_max
                for i in range(dims):
                    x = op[i]
                    op[i] = vmin[i] if x < vmin[i] else (vmax[i] if x > vmax[i] else x)
            else:
                agent.velocity = np.clip(out, v_min, v_max)
                return
        agent.velocity = out
        return
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
