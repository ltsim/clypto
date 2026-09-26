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
    BinaryVar,
    BoolVar,
    CategoricalVar,
    DecoratedOptimizer,
    FloatVar,
    IntegerVar,
    LegacyOptimizer,
    Optimizer,
    PermutationVar,
    Population,
    Problem,
    RuntimeAgent,
    SequenceVar,
    StringVar,
    Termination,
    TransferBinaryVar,
    TransferBoolVar,
    agent,
    legacy,
    optimizer,
)
from clypto.optimizer.native.legacy import _LegacyOptimizer
from clypto.optimizer.native.optimizer import LegacyNativeOptimizer

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
                and issubclass(cls_obj, (_LegacyOptimizer, LegacyNativeOptimizer))
                and cls_obj is not Optimizer
        ):
            yield cls_name, cls_obj


@functools.cache
def get_all_optimizers(verbose=False, *, engine="vectorize"):
    """
    Get all available optimizer classes in clypto library

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
    Get an optimizer class by name

    Args:
        name (str): the classname of the optimizer (e.g, OriginalGA, OriginalWOA), don't pass the module name (e.g, ABC, WOA, GA)
        verbose (bool): whether to print the optimizer information
        engine (str): ``"vectorize"`` (default) or ``"legacy"``

    Returns:
        dict_optimizers (dict): key is the string optimizer class name, value is the actual optimizer class
    """
    cls = {}
    flag = False

    for module_name, module in _engine_modules(engine):
        if module_name == name:
            flag = True
            cls.update(_optimizer_classes(module))
    if verbose:
        if not flag:
            print(f"clypto doesn't support optimizer named: {name}.\n")
            return None

        print(f"Found algorithm: {name}, the supported variants are:")

        for name, optimizer in cls.items():
            print(f"Optimizer: {name} - {optimizer} - {optimizer()}")

    return cls


__all__ = [
    "Problem", "Optimizer", "LegacyOptimizer",
    "agent", "optimizer", "legacy", "Attribute", "Argument", "Population",
    "DecoratedOptimizer", "RuntimeAgent",
    "IntegerVar",
    "FloatVar",
    "StringVar",
    "BinaryVar",
    "BoolVar",
    "CategoricalVar",
    "SequenceVar",
    "PermutationVar",
    "TransferBinaryVar",
    "TransferBoolVar",
]
