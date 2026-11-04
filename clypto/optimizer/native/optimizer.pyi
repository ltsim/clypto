import typing

from clypto.optimizer.history import Tracker

class NativeOptimizer:
    epoch: int
    mode: str
    sort_flag: bool
    EPSILON: float
    name: str
    generator: typing.Any
    rng: typing.Any
    tracker: Tracker
    parameters: dict

    @property
    def nf_counter(self) -> int: ...
    @property
    def termination(self) -> typing.Any: ...
    def initialize_variables(self) -> None: ...
    def initialization(self) -> None: ...
    def before_main_loop(self) -> None: ...
    def solve(self, problem: object, termination: object = ..., starting_solutions: object = ..., seed: int | None = ...,
              debug: bool = ..., track_population: bool = ...,
              before_iteration: typing.Callable[[object, int], None] | None = ...,
              after_iteration: typing.Callable[[object, int], None] | None = ...) -> typing.Any: ...
    def __getattr__(self, name: str) -> typing.Any: ...
