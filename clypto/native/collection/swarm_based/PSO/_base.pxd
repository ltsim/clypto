from clypto.optimizer.native.agent cimport Agent


cpdef void update_velocity(Agent agent, object gbest_solution, object w, double c1, double c2,
                           object generator, object v_min=*, object v_max=*)
cpdef object move(Agent agent)
cpdef bint update_pbest(Agent agent, Agent candidate, str sense)
