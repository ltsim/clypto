import numpy as np

from clypto.agents import AgentStatic


def global_repeated_times(global_best: list[AgentStatic], epsilon: float) -> int:
    count = 0

    for idx in range(0, len(global_best) - 1):
        temp = np.abs(
            global_best[idx].target.fitness - global_best[idx + 1].target.fitness
        )

        if temp <= epsilon:
            count += 1
        else:
            count = 0

    return count
