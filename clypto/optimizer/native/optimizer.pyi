import typing

class NativeOptimizer:
    epoch: int
    mode: str
    sort_flag: bool
    EPSILON: float
    name: str
    generator: typing.Any
    rng: typing.Any
    tracker: typing.Any
    parameters: dict

    @property
    def nf_counter(self) -> int: ...
    @property
    def termination(self) -> typing.Any: ...
    def initialize_variables(self) -> None: ...
    def initialization(self) -> None: ...
    def before_main_loop(self) -> None: ...
    def solve(self, problem: object, termination: object = ..., starting_solutions: object = ..., seed: int | None = ...,
              debug: bool = ..., track_population: bool = ..., history_path: str | None = ...,
              before_iteration: object = ..., after_iteration: object = ...) -> typing.Any: ...
    def __getattr__(self, name: str) -> typing.Any: ...
