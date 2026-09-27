#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

__version__ = "2026a0"

import functools
import importlib
import inspect
import pkgutil

from clypto.optimizer import (
    Agent,
    Argument,
    Attribute,
    BaseBounds,
    Bounds,
    DecoratedOptimizer,
    LegacyOptimizer,
    NumberBounds,
    Optimizer,
    PermutationBounds,
    Population,
    Problem,
    RuntimeAgent,
    SequenceBounds,
    StringBounds,
    Termination,
    TransferBounds,
    agent,
    correct_solution,
    duplicate_agent,
    empty_snapshot,
    opposite_solution,
    optimizer,
    population,
    reset_solution,
    snapshot,
    validator,
)
from clypto.optimizer.native.optimizer import NativeOptimizer


def _algorithm_modules():
    """Yield ``(module_name, module)`` for every algorithm module of the collection."""
    try:
        root = importlib.import_module("clypto.native.collection")
    except ImportError as exc:  # e.g. built with CLYPTO_COLLECTION=0
        raise ImportError(f"the collection is not available in this build: {exc}") from exc
    for category in pkgutil.iter_modules(root.__path__):
        category_pkg = importlib.import_module(f"{root.__name__}.{category.name}")
        for module in pkgutil.iter_modules(category_pkg.__path__):
            if module.ispkg:
                yield module.name, importlib.import_module(f"{category_pkg.__name__}.{module.name}")


def _optimizer_classes(module):
    for cls_name, cls_obj in inspect.getmembers(module):
        if (
                inspect.isclass(cls_obj)
                and not cls_name.startswith("_")
                and issubclass(cls_obj, NativeOptimizer)
                and cls_obj is not LegacyOptimizer
        ):
            yield cls_name, cls_obj


@functools.cache
def get_all_optimizers(verbose=False):
    """
    Get all available optimizer classes in clypto library

    The collection is native: the classes are discovered by walking
    ``clypto/native/collection/<category>/<Module>/<Class>.pyx``.

    Args:
        verbose (bool): whether to print the optimizer information

    Returns:
        dict_optimizers (dict): key is the string optimizer class name, value is the actual optimizer class
    """
    cls = {}

    for _, module in _algorithm_modules():
        cls.update(_optimizer_classes(module))

    if verbose:
        for name, optimizer in cls.items():
            print(f"Optimizer: {name} - {optimizer}")

    return cls


def get_optimizer_by_class(class_name: str, verbose=False):
    """
    Get an optimizer class by its class name

    Args:
        class_name (str): the classname of the optimizer (e.g, C_PSO, OriginalGA), don't pass the module name (e.g, PSO, GA)
        verbose (bool): whether to print the optimizer information

    Returns:
        optimizer (Optimizer): the actual optimizer class or None if the classname is not supported
    """
    try:
        all_optimizers = get_all_optimizers(verbose=verbose)
        return all_optimizers[class_name]
    except KeyError:
        print(f"clypto doesn't support optimizer named: {class_name}.\n")
        return None


def get_optimizer_by_name(name: str, verbose=False):
    """
    Get the optimizer classes of one algorithm module, by module name

    Args:
        name (str): the module name of the algorithm (e.g, PSO, GA, WOA)
        verbose (bool): whether to print the optimizer information

    Returns:
        dict_optimizers (dict): key is the optimizer class name, value is the class (empty if the module is unknown)
    """
    cls = {}
    for module_name, module in _algorithm_modules():
        if module_name == name:
            cls.update(_optimizer_classes(module))
    if verbose:
        if not cls:
            print(f"clypto doesn't support optimizer named: {name}.\n")
        else:
            print(f"Found algorithm: {name}, the supported variants are:")
            for cls_name, optimizer in cls.items():
                print(f"Optimizer: {cls_name} - {optimizer} - {optimizer()}")

    return cls


__all__ = [
    "Problem", "Optimizer", "NativeOptimizer", "LegacyOptimizer",
    "agent", "optimizer", "Attribute", "Argument", "Agent", "Population", "population", "snapshot", "empty_snapshot", "duplicate_agent",
    "correct_solution", "reset_solution", "opposite_solution", "validator", "Termination",
    "DecoratedOptimizer", "RuntimeAgent",
    "get_all_optimizers", "get_optimizer_by_name", "get_optimizer_by_class",
    "Bounds", "BaseBounds", "NumberBounds", "TransferBounds", "StringBounds", "SequenceBounds", "PermutationBounds",
]
