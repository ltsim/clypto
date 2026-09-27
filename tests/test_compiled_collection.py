#!/usr/bin/env python
"""Verify every catalog optimizer is imported from a compiled extension.

The collection is native ``.pyx``. Run ``make build-ext`` (or
``python setup.py build_ext --inplace``) first; otherwise the import silently
falls back to the ``.py`` sources and this test reports which modules are not
compiled.
"""
import importlib
import importlib.machinery

import clypto as cy

EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)


def test_all_optimizer_modules_are_compiled():
    modules = sorted({cls.__module__ for cls in cy.get_all_optimizers().values()})
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
