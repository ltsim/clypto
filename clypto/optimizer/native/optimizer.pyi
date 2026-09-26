import typing

class NativeOptimizer:
    epoch: int
    pop_size: int
    mode: object
    sort_flag: bool
    EPSILON: float
    AVAILABLE_MODES: tuple
    name: str
    generator: typing.Any
    rng: typing.Any
    tracker: typing.Any

    @property
    def parameters(self) -> dict: ...
    @property
    def nf_counter(self) -> int: ...
    @property
    def termination(self) -> typing.Any: ...
    def solve(self, problem: object, termination: object = ..., starting_solutions: object = ..., seed: int | None = ...,
              debug: bool = ..., track_population: bool = ..., history_path: str | None = ...,
              before_iteration: object = ..., after_iteration: object = ...) -> typing.Any: ...
    def __getattr__(self, name: str) -> typing.Any: ...
