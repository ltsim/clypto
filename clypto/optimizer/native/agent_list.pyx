"""Classic list-of-agents API on the native engine.

Algorithms whose logic is built on agent objects (aliased agents edited in place, growing archives,
per-agent fields, variable population sizes, ...) subclass :class:`AgentListOptimizer` and keep their
classic code: ``self.objs`` is the list of agents (the classic ``self.pop``), ``_evolve_agents`` is the
classic ``_evolve`` and the helpers below are the classic engine helpers with identical semantics.
The engine still sees a :class:`NativePopulation` (``self.pop``), mirrored from the agents after
every epoch, so ``g_best`` tracking, termination, history and ``solve()`` are the native ones.
The list helpers (``_get_sorted_population``, ``_greedy_selection_population``, ...) come from
:class:`NativeOptimizer`.
"""
import numpy as np

from clypto.optimizer.native.agent cimport LegacyNativeAgent
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


class FieldAgent(LegacyNativeAgent):
    """An agent with extra named fields (``FieldAgent(x, t, age=0)``); ``copy()`` shares them, like the legacy agents."""


cdef class AgentListOptimizer(VectorizeOptimizer):

    # -- agents ----------------------------------------------------------------
    def _generate_empty_agent(self, solution=None):
        if solution is None:
            solution = self.problem.generate_solution(True)
        return FieldAgent(solution)

    def _generate_agent(self, solution=None):
        agent = self._generate_empty_agent(solution)
        agent.target = self._get_target(agent.solution)
        return agent

    def _generate_agents(self, pop_size=None):
        if pop_size is None:
            pop_size = self.pop_size
        return [self._generate_agent() for _ in range(0, pop_size)]

    def mirror__(self):
        """The agents as the population the engine sorts and reports."""
        # (problem.n_objs is not read: its first access draws a random solution from the seeded stream)
        cdef Py_ssize_t n = len(self.objs), d = self.problem.n_dims, m = 1
        if n:
            m = np.asarray(self.objs[0].target.objectives).size
        cdef NativePopulation pop = NativePopulation(n, d, m, self.layout(d, m), self.problem._obj_weights)
        if n:
            pop.X[:] = np.array([agent.solution for agent in self.objs])
            pop.O[:] = np.array([np.asarray(agent.target.objectives).reshape(-1) for agent in self.objs])
            pop.F[:] = np.array([agent.target.fitness for agent in self.objs])
        return pop

    # -- lifecycle -------------------------------------------------------------
    def _initialization(self):
        if self._starting is not None:
            self.objs = [self._generate_agent(x) for x in self._starting]
        else:
            self.objs = self._generate_agents(self.pop_size)
        self.pop = self.mirror__()

    def _after_initialization(self):
        # The initial population is sorted or not depending on the algorithm.
        sorted_objs = self._get_sorted_population(self.objs, self.problem.sense)
        self.g_best, self.g_worst = sorted_objs[0].copy(), sorted_objs[-1].copy()
        self._g_best_row = -1  # a copy until the first epoch
        if self.sort_flag:
            self.objs = sorted_objs
            self.pop = self.mirror__()

    def _evolve(self, int epoch):
        if self._g_best_row >= 0:
            if self.sort_flag:  # the engine sorted the population after the last epoch
                self.objs = self._get_sorted_population(self.objs, self.problem.sense)
                self._g_best_row = 0
            self.g_best = self.objs[self._g_best_row]  # the classic engine aliases g_best to the best agent
        self._evolve_agents(epoch)
        self.pop = self.mirror__()

    def _evolve_agents(self, epoch):
        raise NotImplementedError(f"{type(self).__name__} does not implement _evolve_agents().")

    # -- classic engine helpers --------------------------------------------------
    def _update_target_for_population(self, pop):
        if self.mode in self.AVAILABLE_MODES:
            for agent in pop:
                agent.target = self._get_target(agent.solution, counted=False)
            self._nfe_counter += len(pop)
        return pop
