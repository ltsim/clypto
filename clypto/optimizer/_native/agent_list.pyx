"""Classic list-of-agents API on the native engine.

Algorithms whose logic is built on agent objects (aliased agents edited in place, growing archives,
per-agent fields, variable population sizes, ...) subclass :class:`AgentListOptimizer` and keep their
classic code: ``self.objs`` is the list of agents (the classic ``self.pop``), ``evolve_agents`` is the
classic ``evolve`` and the helpers below are the classic engine helpers with identical semantics.
The engine still sees a :class:`NativePopulation` (``self.pop``), mirrored from the agents after
every epoch, so ``g_best`` tracking, termination, history and ``solve()`` are the native ones.
"""
import numpy as np

from clypto.optimizer._native.agent cimport LegacyNativeAgent
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


class FieldAgent(LegacyNativeAgent):
    """An agent with extra named fields; ``copy()`` is shallow (arrays are shared), like the classic agents."""

    def __init__(self, solution=None, target=None, **fields):
        LegacyNativeAgent.__init__(self, solution, target)
        self.__dict__.update(fields)

    def copy(self):
        return FieldAgent(self.solution, None if self.target is None else self.target.copy(), **self.__dict__)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


cdef class AgentListOptimizer(LegacyNativeOptimizer):

    # -- agents ----------------------------------------------------------------
    def generate_empty_agent(self, solution=None):
        if solution is None:
            solution = self.problem.generate_solution(True)
        return FieldAgent(solution)

    def generate_agent(self, solution=None):
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        return agent

    def generate_agents(self, pop_size=None):
        if pop_size is None:
            pop_size = self.pop_size
        return [self.generate_agent() for _ in range(0, pop_size)]

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
    cdef void initialization(self):
        if self._starting is not None:
            self.objs = [self.generate_agent(x) for x in self._starting]
        else:
            self.objs = self.generate_agents(self.pop_size)
        self.pop = self.mirror__()

    cdef void after_initialization(self):
        # The initial population is sorted or not depending on the algorithm.
        sorted_objs = self.get_sorted_population(self.objs, self.problem.minmax)
        self.g_best, self.g_worst = sorted_objs[0].copy(), sorted_objs[-1].copy()
        self._g_best_row = -1  # a copy until the first epoch
        if self.sort_flag:
            self.objs = sorted_objs
            self.pop = self.mirror__()

    cdef void evolve(self, int epoch):
        if self._g_best_row >= 0:
            if self.sort_flag:  # the engine sorted the population after the last epoch
                self.objs = self.get_sorted_population(self.objs, self.problem.minmax)
                self._g_best_row = 0
            self.g_best = self.objs[self._g_best_row]  # the classic engine aliases g_best to the best agent
        self.evolve_agents(epoch)
        self.pop = self.mirror__()

    def evolve_agents(self, epoch):
        pass

    # -- classic engine helpers --------------------------------------------------
    def update_target_for_population(self, pop):
        if self.mode in self.AVAILABLE_MODES:
            for agent in pop:
                agent.target = self.get_target(agent.solution, counted=False)
            self._nfe_counter += len(pop)
        return pop

    def compare_target(self, target_x, target_y, minmax="min"):
        return self.compare_fitness(target_x.fitness, target_y.fitness, minmax)

    def get_sorted_population(self, pop, minmax="min"):
        indices = np.argsort([agent.target.fitness for agent in pop]).tolist()
        if minmax == "max":
            indices = indices[::-1]
        return [pop[idx] for idx in indices]

    def get_sorted_indices_population(self, pop, minmax="min"):
        indices = np.argsort([agent.target.fitness for agent in pop]).tolist()
        if minmax == "max":
            indices = indices[::-1]
        return [pop[idx] for idx in indices], indices

    def get_best_agent(self, pop, minmax="min"):
        return self.get_sorted_population(pop, minmax)[0].copy()

    def get_index_best(self, pop, minmax="min"):
        fit_list = np.array([agent.target.fitness for agent in pop])
        return int(np.argmin(fit_list)) if minmax == "min" else int(np.argmax(fit_list))

    def get_worst_agent(self, pop, minmax="min"):
        return self.get_sorted_population(pop, minmax)[-1].copy()

    def get_special_agents(self, pop, n_best=3, n_worst=3, minmax="min"):
        pop = self.get_sorted_population(pop, minmax)
        if n_best is None:
            if n_worst is None:
                return pop, None, None
            return pop, None, [agent.copy() for agent in pop[::-1][:n_worst]]
        if n_worst is None:
            return pop, [agent.copy() for agent in pop[:n_best]], None
        return pop, [agent.copy() for agent in pop[:n_best]], [agent.copy() for agent in pop[::-1][:n_worst]]

    def get_special_fitness(self, pop, minmax="min"):
        total_fitness = np.sum([agent.target.fitness for agent in pop])
        pop = self.get_sorted_population(pop, minmax)
        return total_fitness, pop[0].target.fitness, pop[-1].target.fitness

    def get_better_agent(self, agent_x, agent_y, minmax="min", reverse=False):
        idx = 0 if minmax == "min" else 1
        if reverse:
            idx = 1 - idx
        if idx == 0:
            return agent_x.copy() if agent_x.target.fitness < agent_y.target.fitness else agent_y.copy()
        return agent_y.copy() if agent_x.target.fitness < agent_y.target.fitness else agent_x.copy()

    def greedy_selection_population(self, pop_old=None, pop_new=None, minmax="min"):
        if len(pop_old) != len(pop_new):
            raise ValueError("Greedy selection of two population with different length.")
        if minmax == "min":
            return [pop_new[i] if pop_new[i].target.fitness < pop_old[i].target.fitness else pop_old[i] for i in range(len(pop_old))]
        return [pop_new[i] if pop_new[i].target.fitness > pop_old[i].target.fitness else pop_old[i] for i in range(len(pop_old))]

    def get_sorted_and_trimmed_population(self, pop=None, pop_size=None, minmax="min"):
        return self.get_sorted_population(pop, minmax)[:pop_size]

    def update_global_best_agent(self, pop, save=False):
        sorted_pop = self.get_sorted_population(pop, self.problem.minmax)
        return sorted_pop, sorted_pop[0]

    def get_index_kway_tournament_selection(self, pop=None, k_way=0.2, output=2, reverse=False):
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        k_way_count = int(k_way)
        list_id = self.generator.choice(range(len(pop)), k_way_count, replace=False)
        list_parents = [[idx, pop[idx].target.fitness] for idx in list_id]
        if self.problem.minmax == "min":
            list_parents = sorted(list_parents, key=lambda agent: agent[1])
        else:
            list_parents = sorted(list_parents, key=lambda agent: agent[1], reverse=True)
        if reverse:
            return [parent[0] for parent in list_parents[-output:]]
        return [parent[0] for parent in list_parents[:output]]

    def generate_opposition_solution(self, agent=None, g_best=None):
        pos_new = (
            self.problem.lb + self.problem.ub - g_best.solution
            + self.generator.uniform() * (g_best.solution - agent.solution)
        )
        return self.correct_solution(pos_new)

    def generate_group_population(self, pop, n_groups, m_agents):
        pop_group = []
        for idx in range(0, n_groups):
            group = pop[idx * m_agents: (idx + 1) * m_agents]
            pop_group.append([agent.copy() for agent in group])
        return pop_group

    def improved_ms(self, pop=None, g_best=None):  ## m: mutation, s: search
        pop_len = int(len(pop) / 2)
        pop = sorted(pop, key=lambda agent: agent.target.fitness)
        pop_s1, pop_s2 = pop[:pop_len], pop[pop_len:]
        pop_new = []
        for idx in range(0, pop_len):
            agent = pop_s1[idx].copy()
            pos_new = pop_s1[idx].solution * (1 + self.generator.normal(0, 1, self.problem.n_dims))
            agent.solution = self.correct_solution(pos_new)
            pop_new.append(agent)
        pop_new = self.update_target_for_population(pop_new)
        pop_s1 = self.greedy_selection_population(pop_s1, pop_new, self.problem.minmax)
        pos_s1_list = [agent.solution for agent in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_new = []
        for idx in range(0, pop_len):
            agent = pop_s2[idx].copy()
            pos_new = (g_best.solution - pos_s1_mean) - self.generator.random() * (
                self.problem.lb + self.generator.random() * (self.problem.ub - self.problem.lb)
            )
            agent.solution = self.correct_solution(pos_new)
            pop_new.append(agent)
        pop_s2 = self.update_target_for_population(pop_new)
        pop_s2 = self.greedy_selection_population(pop_s2, pop_new, self.problem.minmax)
        return pop_s1 + pop_s2
