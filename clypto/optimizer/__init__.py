"""Reusable Python API for writing optimizers (legacy or decorator-based)."""
# precompile must load before agents: agents.declaration imports
# precompile.decoration, whose package __init__ imports agents back.
from clypto.optimizer.precompile import (
    Argument,
    DecoratedOptimizer,
    legacy,
    optimizer,
    precompile,
)
from clypto.optimizer.agents import Attribute, LegacyAgent, RuntimeAgent, agent
from clypto.optimizer.bounds import (
    BaseBounds,
    Bounds,
    NumberBounds,
    PermutationBounds,
    SequenceBounds,
    StringBounds,
    TransferBounds,
)
from clypto.optimizer.history import Tracker
from clypto.optimizer.native.legacy import LegacyOptimizer
from clypto.optimizer.native.population import Population
from clypto.optimizer.native.problem import Problem
from clypto.optimizer.native.target import NativeTarget as Target
from clypto.optimizer.termination import Termination
from clypto.optimizer.validator import Validator

Optimizer = LegacyOptimizer

__all__ = [
    "Argument",
    "Attribute",
    "BaseBounds",
    "Bounds",
    "DecoratedOptimizer",
    "LegacyAgent",
    "LegacyOptimizer",
    "NumberBounds",
    "Optimizer",
    "PermutationBounds",
    "Population",
    "Problem",
    "RuntimeAgent",
    "SequenceBounds",
    "StringBounds",
    "Target",
    "Termination",
    "TransferBounds",
    "Tracker",
    "Validator",
    "agent",
    "legacy",
    "optimizer",
    "precompile",
]
