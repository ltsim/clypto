# ``AgentFactory``: the hooks Population delegates agent creation to.
# --------------------------------------------------%
# cdef class PSOAgents(AgentFactory):
#     cdef Agent create(self, object solution=None):
#         cdef Agent agent = AgentFactory.create(self, solution)
#         agent.velocity = self.generator.uniform(-half_range, half_range)
#         return agent
from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.problem cimport Problem


cdef class AgentFactory:
    """The per-algorithm lifecycle of an agent, held by :class:`Population`.

    The one agent-creation entry point of the engine: ``Population.create_agent``
    and ``Population.generate_agent`` delegate to ``create``/``generate``. The
    base implements the general case (a random or given solution, evaluated);

    algorithms that need extra per-agent state (a PSO velocity, a BSA memory
    position, SAP_DE's crossover/mutation rates, ...) subclass and override them.
    ``bind`` attaches the bound problem and the seeded generator — call
    ``AgentFactory.bind(self, problem, generator)`` first when overriding.

    Must not draw randomness anywhere except inside ``create``/``generate``, in
    the same order the catalog draws today: the population's random streams are
    part of the golden parity.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(problem={type(self.problem).__name__ if self.problem is not None else None})"

    cdef void bind(self, Problem problem, object generator):
        """Attach the bound problem and the seeded generator (called by ``Population.bind``)."""
        self.problem = problem
        self.generator = generator

    cdef Agent create(self, object solution=None):
        """A new, not yet evaluated agent at ``solution`` (random when ``None``)."""
        if solution is None:
            solution = self.problem.generate_solution(True)
        return Agent(solution)

    cdef Agent generate(self, object solution=None):
        """``create`` then evaluate it (the initial agent of the algorithm)."""
        cdef Agent agent = self.create(solution)
        agent.evaluate(self.problem)
        return agent
