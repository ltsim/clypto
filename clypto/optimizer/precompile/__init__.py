from clypto.optimizer.precompile.declaration import Argument
from clypto.optimizer.precompile.decorator import legacy, optimizer
from clypto.optimizer.precompile.base import DecoratedOptimizer
from clypto.optimizer.precompile.compiler import compile_decorated, is_precompiling, precompile

__all__ = [
    "Argument",
    "DecoratedOptimizer",
    "compile_decorated",
    "is_precompiling",
    "legacy",
    "optimizer",
    "precompile",
]
