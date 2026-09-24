from clypto.optimizer._native.agent import _LegacyAgent
from clypto.optimizer.termination import Termination
from clypto.optimizer.validator import Validator

class _LegacyOptimizer:
    epoch: int
    pop_size: int
    g_best: _LegacyAgent
    tracker: object
    validator: Validator
    mode: object
    AVAILABLE_MODES: object
    EPSILON: float
    parameters: dict

    def __init__(self, **kwargs: object) -> None: ...
    def set_parameters(self, parameters: object) -> None: ...
    def get_parameters(self) -> dict: ...
    def get_attributes(self) -> dict: ...
    def get_name(self) -> str: ...
    def solve(self, problem: object, termination: object = ..., **kwargs: object) -> _LegacyAgent: ...
