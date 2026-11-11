#!/usr/bin/env python
"""Tests for Clypto JIT compiler, IPython extension, and CLI."""

import importlib
import os
import sys
import numpy as np
import pytest

import clypto
from clypto.compiler import build_extension, compile, compile_file, openmp_flags
from clypto.ipython import ClyptoMagics, load_ipython_extension
import clypto.cli as cli


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


@pytest.fixture
def problem():
    return clypto.Problem(
        obj_func=sphere,
        bounds=clypto.NumberBounds(float, low=[-5.0] * 3, up=[5.0] * 3),
        sense="min",
    )


def test_reexports():
    assert hasattr(clypto, "compile")
    assert hasattr(clypto, "compile_file")
    assert hasattr(clypto, "build_extension")
    assert "compile" in clypto.__all__
    assert "compile_file" in clypto.__all__
    assert "build_extension" in clypto.__all__


def test_build_extension(tmp_path):
    pyx_file = tmp_path / "test_ext.pyx"
    pyx_file.write_text("def test_func(): return 42\n", encoding="utf-8")

    exts = build_extension(str(pyx_file), name="test_ext")
    assert len(exts) == 1
    ext = exts[0]
    assert ext.name == "test_ext"
    assert clypto.get_include() in ext.include_dirs
    assert any("numpy" in d.lower() for d in ext.include_dirs)

    # OpenMP flags check
    omp_c, omp_l = openmp_flags()
    for arg in omp_c:
        assert arg in ext.extra_compile_args
    for arg in omp_l:
        assert arg in ext.extra_link_args


def test_compile_jit(problem):
    source = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class JITUserOptimizer(cy.Optimizer):
    cdef public double factor
    cdef public bint init_called

    def __init__(self, epoch=10, pop_size=8, factor=0.4, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")
        self.init_called = False

    def initialize(self):
        self.init_called = True

    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * self.factor
            agent.evaluate(self.problem)
"""
    # First compile
    cls1 = compile(source)
    assert cls1.__name__ == "JITUserOptimizer"

    # Second compile (should hit cache)
    cls2 = compile(source)
    assert cls2.__name__ == "JITUserOptimizer"
    assert cls1 is cls2

    # Instantiate and solve
    opt = cls1(epoch=5, pop_size=6, factor=0.5)
    best = opt.solve(problem, seed=42)
    assert opt.init_called is True
    assert np.isfinite(best.fitness)


def test_compile_named_and_multi():
    source = """# cython: language_level=3
cimport clypto.core as cy

cdef class FirstAlgo(cy.Optimizer):
    cdef void evolve(self, int epoch):
        pass

cdef class SecondAlgo(cy.Optimizer):
    cdef void evolve(self, int epoch):
        pass
"""
    # Without name: multiple optimizer classes -> returns dict
    res_dict = compile(source)
    assert isinstance(res_dict, dict)
    assert "FirstAlgo" in res_dict
    assert "SecondAlgo" in res_dict

    # With name: returns specific class
    first = compile(source, name="FirstAlgo")
    assert first.__name__ == "FirstAlgo"
    second = compile(source, name="SecondAlgo")
    assert second.__name__ == "SecondAlgo"


def test_compile_file(tmp_path, problem):
    pyx_content = """# cython: language_level=3
cimport clypto.core as cy

cdef class FileOptimizer(cy.Optimizer):
    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * 0.5
            agent.evaluate(self.problem)
"""
    pyx_file = tmp_path / "file_opt.pyx"
    pyx_file.write_text(pyx_content, encoding="utf-8")

    cls = compile_file(str(pyx_file))
    assert cls.__name__ == "FileOptimizer"
    opt = cls(epoch=5, pop_size=6)
    best = opt.solve(problem, seed=99)
    assert np.isfinite(best.fitness)


def test_ipython_magic():
    class MockInteractiveShell:
        def __init__(self):
            self.user_ns = {}
            self.magics = []

        def register_magics(self, magic_cls):
            self.magics.append(magic_cls(shell=self))

    shell = MockInteractiveShell()
    load_ipython_extension(shell)
    assert len(shell.magics) == 1
    magics = shell.magics[0]
    assert isinstance(magics, ClyptoMagics)

    cell_code = """# cython: language_level=3
cimport clypto.core as cy

cdef class NotebookAlgo(cy.Optimizer):
    cdef void evolve(self, int epoch):
        pass

def magic_calc(int x):
    return x * 10
"""
    magics.clypto("", cell_code)
    assert "NotebookAlgo" in shell.user_ns
    assert "magic_calc" in shell.user_ns
    assert shell.user_ns["magic_calc"](5) == 50


def test_cli_new_and_build(tmp_path, problem):
    target_pyx = tmp_path / "custom_metaheuristic.pyx"

    # Test "clypto new"
    rc_new = cli.main(["new", str(target_pyx)])
    assert rc_new == 0
    assert target_pyx.is_file()

    content = target_pyx.read_text(encoding="utf-8")
    assert "class CustomMetaheuristic" in content

    # Test "clypto build"
    rc_build = cli.main(["build", str(target_pyx)])
    assert rc_build == 0

    so_files = [f for f in os.listdir(tmp_path) if f.endswith((".so", ".pyd"))]
    assert len(so_files) >= 1

    # Verify compiled extension can be imported and executed
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("custom_metaheuristic")
        algo_cls = mod.CustomMetaheuristic
        opt = algo_cls(epoch=5, pop_size=6, factor=0.5)
        best = opt.solve(problem, seed=123)
        assert np.isfinite(best.fitness)
    finally:
        sys.path.remove(str(tmp_path))
