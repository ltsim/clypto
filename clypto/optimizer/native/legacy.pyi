import typing

from clypto.optimizer.native.optimizer import NativeOptimizer
from clypto.optimizer.validator import Validator

class LegacyOptimizer(NativeOptimizer):
    pop: typing.Any
    g_best: typing.Any
    g_worst: typing.Any
    problem: typing.Any
    validator: Validator

    def __init__(self, **kwargs: object) -> None: ...
