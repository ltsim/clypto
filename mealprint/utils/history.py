#!/usr/bin/env python
# Created by "Thieu" at 14:51, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from mealprint.agents.virtual_agent import VirtualAgent


class History:
    """
    A History class is responsible for saving each iteration's output.

    Notes
    ~~~~~
    + Access to variables in this class:
        + list_global_best: List of global best SOLUTION found so far in all previous generations
        + list_current_best: List of current best SOLUTION in each previous generations
        + list_epoch_time: List of runtime for each generation
        + list_global_best_fit: List of global best FITNESS found so far in all previous generations
        + list_current_best_fit: List of current best FITNESS in each previous generations
        + list_diversity: List of DIVERSITY of swarm in all generations
        + list_exploitation: List of EXPLOITATION percentages for all generations
        + list_exploration: List of EXPLORATION percentages for all generations
        + list_global_worst: List of global worst SOLUTION found so far in all previous generations
        + list_current_worst: List of current worst SOLUTION in each previous generations
        + list_population: List of POPULATION in each generations
        + **Warning**, the last variable 'list_population' can cause the error related to 'memory' when saving model.
            Better to set parameter 'save_population' to False in the input problem dictionary to not using it.
    """

    def __init__(self, **kwargs):
        self.list_global_best = []  # List of global best solution found so far in all previous generations
        self.list_current_best = []  # List of current best solution in each previous generations
        self.list_epoch_time = []  # List of runtime for each generation
        self.list_global_best_fit = []  # List of global best fitness found so far in all previous generations
        self.list_current_best_fit = []  # List of current best fitness in each previous generations
        self.list_population = []  # List of population in each generation
        self.list_diversity = []  # List of diversity of swarm in all generations
        self.list_exploitation = []  # List of exploitation percentages for all generations
        self.list_exploration = []  # List of exploration percentages for all generations
        self.list_global_worst = []  # List of global worst solution found so far in all previous generations
        self.list_current_worst = []  # List of current worst solution in each previous generations
        self.epoch = None

    def store_initial_best_worst(self, best_agent: VirtualAgent, worst_agent: VirtualAgent) -> None:
        self.list_global_best = [best_agent.copy()]
        self.list_current_best = [best_agent.copy()]
        self.list_global_worst = [worst_agent.copy()]
        self.list_current_worst = [worst_agent.copy()]

    def get_global_repeated_times(self, epsilon: float) -> int:
        count = 0

        for idx in range(0, len(self.list_global_best) - 1):
            temp = np.abs(self.list_global_best[idx].target.fitness - self.list_global_best[idx + 1].target.fitness)

            if temp <= epsilon:
                count += 1
            else:
                count = 0

        return count
