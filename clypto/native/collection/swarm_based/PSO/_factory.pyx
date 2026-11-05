# Shared agent factory of the PSO family.
# --------------------------------------------------%
"""The particle lifecycle shared by every PSO variant (``cy.duplicate_agent``-free).

``PSOAgents`` creates a particle with a velocity inside the algorithm's
*current* ``[v_min, v_max]`` — read through the optimizer reference at creation
time, so variants that rescale their limits on the fly (``P_PSO``) keep behaving
exactly as before — and generates it evaluated with ``pbest = self``.
"""
from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.factory cimport AgentFactory

from clypto.native.collection.swarm_based.PSO._base cimport update_pbest


cdef class PSOAgents(AgentFactory):
    """The particle factory of the PSO family.

    Holds a reference to its optimizer instead of copying ``v_min``/``v_max``:
    the limits are the algorithm's live hyper-state (``P_PSO`` recomputes
    ``v_max`` per epoch), and ``create`` always reads them at call time.
    """

    def __init__(self, owner=None):
        self.owner = owner

    cdef Agent create(self, object solution=None):
        cdef Agent agent = AgentFactory.create(self, solution)
        agent.velocity = self.generator.uniform(self.owner.v_min, self.owner.v_max)
        return agent

    cdef Agent generate(self, object solution=None):
        cdef Agent agent = self.create(solution)
        agent.evaluate(self.problem)
        update_pbest(agent, agent, self.problem.sense)
        return agent
