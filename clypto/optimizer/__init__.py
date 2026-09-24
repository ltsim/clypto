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
from clypto.optimizer.history import Tracker
from clypto.optimizer.legacy import LegacyOptimizer
from clypto.optimizer.population import Population
from clypto.optimizer.problem import Problem
from clypto.optimizer.space import (
    BinaryVar,
    BoolVar,
    CategoricalVar,
    FloatVar,
    IntegerVar,
    PermutationVar,
    SequenceVar,
    StringVar,
    TransferBinaryVar,
    TransferBoolVar,
)
from clypto.optimizer.target import Target
from clypto.optimizer.termination import Termination
from clypto.optimizer.validator import Validator

Optimizer = LegacyOptimizer

__all__ = [
    "Argument",
    "Attribute",
    "BinaryVar",
    "BoolVar",
    "CategoricalVar",
    "DecoratedOptimizer",
    "FloatVar",
    "IntegerVar",
    "LegacyAgent",
    "LegacyOptimizer",
    "Optimizer",
    "PermutationVar",
    "Population",
    "Problem",
    "RuntimeAgent",
    "SequenceVar",
    "StringVar",
    "Target",
    "Termination",
    "TransferBinaryVar",
    "TransferBoolVar",
    "Tracker",
    "Validator",
    "agent",
    "legacy",
    "optimizer",
    "precompile",
]
