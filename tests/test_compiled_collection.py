#!/usr/bin/env python
"""Verify every catalog optimizer is imported from a compiled extension.

The collection is native ``.pyx``. Run ``make build-ext`` (or
``python setup.py build_ext --inplace``) first; otherwise the import silently
falls back to the ``.py`` sources and this test reports which modules are not
compiled.
"""
import importlib
import importlib.machinery

import pytest

import clypto as cy
from tests._engines import ENGINES

EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)


@pytest.mark.parametrize("engine", ENGINES)
def test_all_optimizer_modules_are_compiled(engine):
    modules = sorted({cls.__module__ for cls in cy.get_all_optimizers(engine=engine).values()})
    assert modules, "no optimizers discovered"

    not_compiled = [
        name
        for name in modules
        if not getattr(importlib.import_module(name), "__file__", "").endswith(
            EXTENSION_SUFFIXES
        )
    ]

    assert not not_compiled, (
        "optimizer modules imported from source instead of compiled extensions; "
        f"run `make build-ext`: {not_compiled}"
    )
