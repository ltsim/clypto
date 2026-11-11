import typing

from clypto.optimizer.native.optimizer import NativeOptimizer
from clypto.optimizer.native.population import Population

class LegacyOptimizer(NativeOptimizer):
    population: Population
    pop_size: int
    g_best: typing.Any
    g_worst: typing.Any
    problem: typing.Any
    bounds: typing.Any

    def __init__(
        self,
        parameters: typing.Sequence[str] = ...,
        sort_flag: bool = ...,
        epoch: typing.Any = ...,
        pop_size: typing.Any = ...,
        **kwargs: typing.Any,
    ) -> None: ...

    def generate(self, agent: typing.Any, solution: typing.Any) -> typing.Any: ...
    def generate_agent(self, solution: typing.Optional[typing.Any] = ...) -> typing.Any: ...
