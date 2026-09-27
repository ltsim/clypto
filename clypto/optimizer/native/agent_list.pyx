"""Classic list-of-agents API on the native engine.

Algorithms whose logic is built on agent objects (aliased agents edited in place, growing archives,
per-agent fields, variable population sizes, ...) subclass :class:`AgentListOptimizer` and keep their
classic code: ``self.objs`` is the list of agents and ``evolve_agents`` is the classic ``evolve``.
The engine still sees a :class:`NativePopulation` (``self.pop``), mirrored from the agents after
every epoch, so ``g_best`` tracking, termination, history and ``solve()`` are the native ones.
"""
import numpy as np

from clypto.optimizer.native.agent cimport Agent, sort_agents
from clypto.optimizer.native.population cimport NativePopulation


class FieldAgent(Agent):
    """An agent with extra named fields (``FieldAgent(x, age=0)``); ``copy()`` shares them, like the legacy agents."""


cdef class AgentListOptimizer(VectorizeOptimizer):

    # -- agents ----------------------------------------------------------------
    def create_agent(self, solution=None):
        if solution is None:
            solution = self.problem.generate_solution(True)
        return FieldAgent(solution)

    def generate_agent(self, solution=None):
        agent = self.create_agent(solution)
        agent.evaluate(self.problem)
        return agent

    def generate_agents(self, pop_size=None):
        if pop_size is None:
            pop_size = self.pop_size
        return [self.generate_agent() for _ in range(0, pop_size)]

    def evaluate_agents(self, agents):
        """Evaluate every agent in the batch modes (``swarm``, ``parallel``, ...); sequential mode does nothing."""
        if self.mode in self.AVAILABLE_MODES:
            for agent in agents:
                agent.evaluate(self.problem)
        return agents

    def mirror__(self):
        """The agents as the population the engine sorts and reports."""
        # (problem.n_objs is not read: its first access draws a random solution from the seeded stream)
        cdef Py_ssize_t n = len(self.objs), d = self.problem.n_dims, m = 1
        if n:
            m = np.asarray(self.objs[0].objectives).size
        cdef NativePopulation pop = NativePopulation(n, d, m, self.layout(d, m), self.problem._obj_weights)
        if n:
            pop.X[:] = np.array([agent.solution for agent in self.objs])
            pop.O[:] = np.array([np.asarray(agent.objectives).reshape(-1) for agent in self.objs])
            pop.F[:] = np.array([agent.fitness for agent in self.objs])
        return pop

    # -- lifecycle -------------------------------------------------------------
    def initialization(self):
        if self._starting is not None:
            self.objs = [self.generate_agent(x) for x in self._starting]
        else:
            self.objs = self.generate_agents(self.pop_size)
        self.pop = self.mirror__()

    cdef void after_initialization(self):
        # The initial population is sorted or not depending on the algorithm.
        sorted_objs = sort_agents(self.objs, self.problem.sense)
        self.g_best, self.g_worst = sorted_objs[0].copy(), sorted_objs[-1].copy()
        self._g_best_row = -1  # a copy until the first epoch
        if self.sort_flag:
            self.objs = sorted_objs
            self.pop = self.mirror__()

    def evolve(self, int epoch):
        if self._g_best_row >= 0:
            if self.sort_flag:  # the engine sorted the population after the last epoch
                self.objs = sort_agents(self.objs, self.problem.sense)
                self._g_best_row = 0
            self.g_best = self.objs[self._g_best_row]  # the classic engine aliases g_best to the best agent
        self.evolve_agents(epoch)
        self.pop = self.mirror__()

    def evolve_agents(self, epoch):
        raise NotImplementedError(f"{type(self).__name__} does not implement evolve_agents().")
