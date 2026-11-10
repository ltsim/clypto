# Shared agent factory of the SLO family.
# --------------------------------------------------%
"""The agent factory shared by the SLO family.

At birth of every sea lion the local position is evaluated too and the agent
keeps whichever solution wins (local best idea from PSO). Both evaluations go
through ``Agent(...).evaluate(problem)`` — the same counter and no RNG beyond
the solution draw itself.
"""
import numpy as np
cimport numpy as cnp
cimport clypto.core as cy

from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.factory cimport AgentFactory

cnp.import_array()


cdef class ModifiedSLOAgents(AgentFactory):
    """The agent factory of ``ModifiedSLO``: the local best is compared to the target at birth."""

    cdef Agent create(self, object solution=None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        local_pos = self.problem.bounds.low + self.problem.bounds.up - solution
        local_pos = cy.correct_solution(self.problem, local_pos)

        agent = cy.Agent(solution)
        agent.local_solution = local_pos
        return agent

    cdef Agent generate(self, object solution=None):
        agent = self.create(solution)
        target = Agent(agent.solution).evaluate(self.problem)
        local_best = Agent(agent.local_solution).evaluate(self.problem)
        if cy.is_better(target, local_best, self.problem.sense):
            t1 = agent.local_solution.copy()
            t2 = agent.solution.copy()
            agent.update_solution(local_best, t1)
            agent.local_solution = t2
            agent.local_best = target
        else:
            t1 = agent.solution.copy()
            t2 = agent.local_solution.copy()
            agent.update_solution(target, t1)
            agent.local_solution = t2
            agent.local_best = local_best
        return agent
