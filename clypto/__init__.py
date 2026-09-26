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
    Argument,
    Attribute,
    BaseBounds,
    Bounds,
    DecoratedOptimizer,
    LegacyAgent,
    LegacyOptimizer,
    NumberBounds,
    Optimizer,
    PermutationBounds,
    Population,
    Problem,
    RuntimeAgent,
    SequenceBounds,
    StringBounds,
    Target,
    Termination,
    TransferBounds,
    agent,
    legacy,
    optimizer,
)
from clypto.optimizer.native.optimizer import NativeOptimizer
from clypto.optimizer.native.vectorize import VectorizeOptimizer

ENGINES = ("vectorize", "legacy")


def _engine_modules(engine):
    """Yield ``(module_name, module)`` for every algorithm module of a collection (``vectorize`` or ``legacy``)."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    try:
        root = importlib.import_module(f"clypto.native.collection.{engine}")
    except ImportError as exc:  # e.g. built with CLYPTO_LEGACY=0
        raise ImportError(f"the {engine!r} collection is not available in this build: {exc}") from exc
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
                and cls_obj not in (LegacyOptimizer, VectorizeOptimizer)
        ):
            yield cls_name, cls_obj


@functools.cache
def get_all_optimizers(verbose=False, *, engine="vectorize"):
    """
    Get all available optimizer classes in clypto library

    The collections are native: the classes are discovered by walking
    ``clypto/native/collection/<engine>/<category>/<Module>/<Class>.pyx``.

    Args:
        verbose (bool): whether to print the optimizer information
        engine (str): ``"vectorize"`` (default, the vectorized collection) or ``"legacy"`` (the classic collection)

    Returns:
        dict_optimizers (dict): key is the string optimizer class name, value is the actual optimizer class
    """
    cls = {}

    for _, module in _engine_modules(engine):
        cls.update(_optimizer_classes(module))

    if verbose:
        for name, optimizer in cls.items():
            print(f"Optimizer: {name} - {optimizer}")

    return cls


def get_optimizer_by_class(class_name: str, verbose=False, *, engine="vectorize"):
    """
    Get an optimizer class by its class name

    Args:
        class_name (str): the classname of the optimizer (e.g, C_PSO, OriginalGA), don't pass the module name (e.g, PSO, GA)
        verbose (bool): whether to print the optimizer information
        engine (str): ``"vectorize"`` (default) or ``"legacy"``

    Returns:
        optimizer (Optimizer): the actual optimizer class or None if the classname is not supported
    """
    try:
        all_optimizers = get_all_optimizers(verbose=verbose, engine=engine)
        return all_optimizers[class_name]
    except KeyError:
        print(f"clypto doesn't support optimizer named: {class_name}.\n")
        return None


def get_optimizer_by_name(name: str, verbose=False, *, engine="vectorize"):
    """
    Get the optimizer classes of one algorithm module, by module name

    Args:
        name (str): the module name of the algorithm (e.g, PSO, GA, WOA)
        verbose (bool): whether to print the optimizer information
        engine (str): ``"vectorize"`` (default) or ``"legacy"``

    Returns:
        dict_optimizers (dict): key is the optimizer class name, value is the class (empty if the module is unknown)
    """
    cls = {}
    for module_name, module in _engine_modules(engine):
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
    "Problem", "Optimizer", "NativeOptimizer", "LegacyOptimizer", "VectorizeOptimizer",
    "agent", "optimizer", "legacy", "Attribute", "Argument", "Population", "LegacyAgent", "Target", "Termination",
    "DecoratedOptimizer", "RuntimeAgent",
    "get_all_optimizers", "get_optimizer_by_name", "get_optimizer_by_class",
    "Bounds", "BaseBounds", "NumberBounds", "TransferBounds", "StringBounds", "SequenceBounds", "PermutationBounds",
]
