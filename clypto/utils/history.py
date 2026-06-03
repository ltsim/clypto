#!/usr/bin/env python
# Created by "Thieu" at 14:51, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import dataclasses

import numpy as np

from clypto.agents.virtual import BaseAgent


@dataclasses.dataclass
class TrackHistory:
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

    global_best: list[BaseAgent] = dataclasses.field(default_factory=list)
    current_best: list[BaseAgent] = dataclasses.field(default_factory=list)
    epoch_time: list[BaseAgent] = dataclasses.field(default_factory=list)
    global_best_fit: list[BaseAgent] = dataclasses.field(default_factory=list)
    current_best_fit: list[BaseAgent] = dataclasses.field(default_factory=list)
    population: list[BaseAgent] = dataclasses.field(default_factory=list)
    diversity: list[BaseAgent] = dataclasses.field(default_factory=list)
    exploitation: list[BaseAgent] = dataclasses.field(default_factory=list)
    exploration: list[BaseAgent] = dataclasses.field(default_factory=list)
    global_worst: list[BaseAgent] = dataclasses.field(default_factory=list)
    current_worst: list[BaseAgent] = dataclasses.field(default_factory=list)
    epoch: int = 0

    def store_initial_best_worst(self, best_agent: BaseAgent, worst_agent: BaseAgent) -> None:
        self.global_best = [best_agent.copy()]
        self.current_best = [best_agent.copy()]
        self.global_worst = [worst_agent.copy()]
        self.current_worst = [worst_agent.copy()]

    def get_global_repeated_times(self, epsilon: float) -> int:
        count = 0

        for idx in range(0, len(self.global_best) - 1):
            temp = np.abs(self.global_best[idx].target.fitness - self.global_best[idx + 1].target.fitness)

            if temp <= epsilon:
                count += 1
            else:
                count = 0

        return count
