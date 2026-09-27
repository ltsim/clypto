"""Reusable Python API for writing optimizers (decorator-based) and the engine classes."""
# precompile must load before agents: agents.declaration imports
# precompile.decoration, whose package __init__ imports agents back.
from clypto.optimizer.precompile import Argument, DecoratedOptimizer, optimizer
from clypto.optimizer.agents import Attribute, RuntimeAgent, agent
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
from clypto.optimizer.native.agent import Agent
from clypto.optimizer.native.legacy import LegacyOptimizer
from clypto.optimizer.native.population import Population, ResetPopulation, population
from clypto.optimizer.native.problem import Problem
from clypto.optimizer.native.utils import validator
from clypto.optimizer.termination import Termination

Optimizer = LegacyOptimizer

__all__ = [
    "Agent",
    "Argument",
    "Attribute",
    "BaseBounds",
    "Bounds",
    "DecoratedOptimizer",
    "LegacyOptimizer",
    "NumberBounds",
    "Optimizer",
    "PermutationBounds",
    "Population",
    "Problem",
    "ResetPopulation",
    "RuntimeAgent",
    "SequenceBounds",
    "StringBounds",
    "Termination",
    "TransferBounds",
    "Tracker",
    "agent",
    "optimizer",
    "population",
    "validator",
]
