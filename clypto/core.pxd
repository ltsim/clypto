# The ``cy`` namespace of the native collection: ``cimport clypto.core as cy``.
#
#   cdef class OriginalX(cy.Optimizer):
#       def __init__(self, epoch=10000, pop_size=100, **kwargs):
#           super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
#           self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
#           self.population = cy.population(pop_size, range=[5, 10000])
#
#       cdef void evolve(self, int epoch):
#           ...

from clypto.optimizer.native.agent cimport (
    Agent, argsort_agents, better_fitness, duplicate_agent, get_better_agent, greedy_agents, is_better, sort_agents,
    sync_if_duplicate,
)
from clypto.optimizer.native.legacy cimport LegacyOptimizer as Optimizer
from clypto.optimizer.native.population cimport (
    Population, correct_solution, empty_snapshot, opposite_solution, population, reset_solution, snapshot,
)
from clypto.optimizer.native.problem cimport Problem
from clypto.optimizer.native.utils cimport (
    check_is_int_and_float, kway_tournament, levy_flight, roulette_wheel, split_groups, validator,
)
