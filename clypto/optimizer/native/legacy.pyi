import typing

from clypto.optimizer.native.optimizer import NativeOptimizer
from clypto.optimizer.native.population import Population

class LegacyOptimizer(NativeOptimizer):
    population: Population
    pop_size: int
    g_best: typing.Any
    g_worst: typing.Any
    problem: typing.Any

    def __init__(self, parameters: typing.Sequence[str] = ..., sort_flag: bool = ..., **kwargs: object) -> None: ...
