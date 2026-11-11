#!/usr/bin/env python
# Created by "LTSIM" at 18:25, 10/11/2026 ----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
"""Unit and integration tests for Clypto JIT compiler, extension builder, IPython magic, and CLI."""

from __future__ import annotations

import importlib
import os
import sys
import time
from typing import Any

import numpy as np
import pytest
from Cython.Compiler.Errors import CompileError
from setuptools import Extension

import clypto
import clypto.cli as cli
from clypto.compiler import (
    DEFAULT_CACHE_DIR,
    DEFAULT_COMPILER_DIRECTIVES,
    build_extension,
    compile,
    compile_file,
    openmp_flags,
)
from clypto.ipython import ClyptoMagics, load_ipython_extension


def sphere(x: np.ndarray) -> float:
    """Standard sphere objective function for testing optimizers."""
    return float(np.sum(x**2))


@pytest.fixture
def problem() -> clypto.Problem:
    """Fixture providing a standard 3D sphere minimization problem."""
    return clypto.Problem(
        obj_func=sphere,
        bounds=clypto.NumberBounds(float, low=[-5.0] * 3, up=[5.0] * 3),
        sense="min",
    )


# ---------------------------------------------------------------------------
# 1. clypto.compile and clypto.compile_file tests
# ---------------------------------------------------------------------------


def test_public_reexports() -> None:
    """Verify compile, compile_file, build_extension, and get_include are part of clypto public API."""
    assert hasattr(clypto, "compile")
    assert hasattr(clypto, "compile_file")
    assert hasattr(clypto, "build_extension")
    assert hasattr(clypto, "get_include")
    assert "compile" in clypto.__all__
    assert "compile_file" in clypto.__all__
    assert "build_extension" in clypto.__all__
    assert "get_include" in clypto.__all__
    assert DEFAULT_CACHE_DIR is not None
    assert "language_level" in DEFAULT_COMPILER_DIRECTIVES


def test_compile_jit_and_solve(problem: clypto.Problem) -> None:
    """Test JIT compiling Cython optimizer code with clypto.compile() and executing solve()."""
    source = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class CustomJITOptimizer(cy.Optimizer):
    cdef public double step_scale
    cdef public bint initialized

    def __init__(self, epoch=10, pop_size=8, step_scale=0.5, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.step_scale = cy.validator(float, step_scale, (0.0, 1.0), "step_scale")
        self.initialized = False

    def initialize(self):
        self.initialized = True

    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * self.step_scale
            agent.evaluate(self.problem)
"""
    cls = compile(source)
    assert cls.__name__ == "CustomJITOptimizer"
    assert issubclass(cls, clypto.LegacyOptimizer)

    opt = cls(epoch=5, pop_size=6, step_scale=0.3)
    best = opt.solve(problem, seed=42)
    assert opt.initialized is True
    assert best is not None
    assert np.isfinite(best.fitness)
    assert isinstance(best.solution, np.ndarray)
    assert len(best.solution) == 3


def test_compile_caching() -> None:
    """Test caching: second call with identical code reuses cached module/class and returns quickly."""
    source = """# cython: language_level=3
cimport clypto.core as cy

cdef class CachedJITOptimizer(cy.Optimizer):
    cdef void evolve(self, int epoch):
        pass
"""
    # First compilation
    cls1 = compile(source)
    assert cls1.__name__ == "CachedJITOptimizer"

    # Second compilation with identical source: must hit sys.modules or cache_dir directly
    t0 = time.perf_counter()
    cls2 = compile(source)
    elapsed = time.perf_counter() - t0

    assert cls1 is cls2
    assert cls1.__module__ == cls2.__module__
    assert elapsed < 0.05, f"Cache lookup took {elapsed:.4f}s, expected < 0.05s (< 50ms)"


def test_compile_named_and_multi_class() -> None:
    """Test compiling code with multiple optimizer classes and extracting target by name."""
    source = """# cython: language_level=3
cimport clypto.core as cy

cdef class MultiOptAlpha(cy.Optimizer):
    cdef void evolve(self, int epoch):
        pass

cdef class MultiOptBeta(cy.Optimizer):
    cdef void evolve(self, int epoch):
        pass
"""
    # Without name specified: returns dict of discovered optimizers
    result_dict = compile(source)
    assert isinstance(result_dict, dict)
    assert "MultiOptAlpha" in result_dict
    assert "MultiOptBeta" in result_dict
    assert issubclass(result_dict["MultiOptAlpha"], clypto.LegacyOptimizer)
    assert issubclass(result_dict["MultiOptBeta"], clypto.LegacyOptimizer)

    # Specifying target name returns that specific class
    opt_alpha = compile(source, name="MultiOptAlpha")
    assert opt_alpha.__name__ == "MultiOptAlpha"
    opt_beta = compile(source, name="MultiOptBeta")
    assert opt_beta.__name__ == "MultiOptBeta"


def test_compile_nonexistent_name() -> None:
    """Test requesting a nonexistent class or attribute raises AttributeError."""
    source = """# cython: language_level=3
cimport clypto.core as cy

cdef class ExistingClass(cy.Optimizer):
    cdef void evolve(self, int epoch):
        pass
"""
    with pytest.raises(AttributeError, match="Could not find attribute or class"):
        compile(source, name="NonExistentClass")


def test_compile_syntax_error() -> None:
    """Test that Cython syntax errors raise an exception during compilation."""
    source = "def broken_syntax(::: invalid cython :::"
    with pytest.raises(CompileError):
        compile(source)


def test_compile_file(tmp_path: Any, problem: clypto.Problem) -> None:
    """Test clypto.compile_file() by compiling a .pyx file and solving a problem."""
    source = """# cython: language_level=3
cimport clypto.core as cy

cdef class FromFileOptimizer(cy.Optimizer):
    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * 0.25
            agent.evaluate(self.problem)
"""
    pyx_path = tmp_path / "from_file_opt.pyx"
    pyx_path.write_text(source, encoding="utf-8")

    cls = compile_file(str(pyx_path))
    assert cls.__name__ == "FromFileOptimizer"

    opt = cls(epoch=4, pop_size=6)
    best = opt.solve(problem, seed=123)
    assert best is not None
    assert np.isfinite(best.fitness)


# ---------------------------------------------------------------------------
# 2. clypto.build_extension tests
# ---------------------------------------------------------------------------


def test_build_extension_single_path(tmp_path: Any) -> None:
    """Test build_extension with a single source file path string."""
    test_pyx = tmp_path / "test_module.pyx"
    test_pyx.write_text("def ping(): return 'pong'\n", encoding="utf-8")

    exts = build_extension(str(test_pyx))
    assert len(exts) == 1
    ext = exts[0]
    assert isinstance(ext, Extension)
    assert ext.name == "test_module"

    # Check clypto and numpy include directories
    clypto_inc = clypto.get_include()
    assert clypto_inc in ext.include_dirs
    assert any("numpy" in d.lower() for d in ext.include_dirs)

    # Check OpenMP flags present by default
    omp_c, omp_l = openmp_flags()
    for flag in omp_c:
        assert flag in ext.extra_compile_args
    for flag in omp_l:
        assert flag in ext.extra_link_args


def test_build_extension_options(tmp_path: Any) -> None:
    """Test build_extension with custom options (openmp=False, custom directives, custom name)."""
    test_pyx = tmp_path / "test_custom.pyx"
    test_pyx.write_text("def ping(): return 'pong'\n", encoding="utf-8")

    exts = build_extension(
        str(test_pyx),
        name="custom_named_ext",
        openmp=False,
        compiler_directives={"boundscheck": True},
        extra_compile_args=["-O2"],
    )
    assert len(exts) == 1
    ext = exts[0]
    assert ext.name == "custom_named_ext"
    assert "-O2" in ext.extra_compile_args

    # When openmp=False, openmp flags should not be added
    omp_c, _ = openmp_flags()
    if omp_c:
        for flag in omp_c:
            assert flag not in ext.extra_compile_args


def test_build_extension_from_extension_object(tmp_path: Any) -> None:
    """Test build_extension when passed an existing setuptools.Extension instance."""
    test_pyx = tmp_path / "test_ext_obj.pyx"
    test_pyx.write_text("def ping(): return 1\n", encoding="utf-8")

    orig_ext = Extension("my_ext_obj", [str(test_pyx)])
    exts = build_extension(orig_ext)
    assert len(exts) == 1
    ext = exts[0]
    assert ext.name == "my_ext_obj"
    assert clypto.get_include() in ext.include_dirs
    assert any("numpy" in d.lower() for d in ext.include_dirs)


def test_build_extension_sequence(tmp_path: Any) -> None:
    """Test build_extension when passed a sequence of source paths."""
    s1 = tmp_path / "s1.pyx"
    s2 = tmp_path / "s2.pyx"
    s1.write_text("def fn1(): pass\n", encoding="utf-8")
    s2.write_text("def fn2(): pass\n", encoding="utf-8")

    exts = build_extension([str(s1), str(s2)])
    assert len(exts) == 2
    names = {ext.name for ext in exts}
    assert names == {"s1", "s2"}


# ---------------------------------------------------------------------------
# 3. clypto.ipython tests
# ---------------------------------------------------------------------------


def test_ipython_registration_and_cell_magic(problem: clypto.Problem) -> None:
    """Test load_ipython_extension and %%clypto cell magic with InteractiveShell user_ns injection."""
    class MockInteractiveShell:
        def __init__(self) -> None:
            self.user_ns: dict[str, Any] = {}
            self.registered_magics: list[Any] = []

        def register_magics(self, magic_cls: Any) -> None:
            self.registered_magics.append(magic_cls(shell=self))

    shell = MockInteractiveShell()
    load_ipython_extension(shell)

    assert len(shell.registered_magics) == 1
    magics = shell.registered_magics[0]
    assert isinstance(magics, ClyptoMagics)

    cell_source = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class NotebookOptimizer(cy.Optimizer):
    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * 0.1
            agent.evaluate(self.problem)

def helper_add(int a, int b):
    return a + b
"""
    magics.clypto(line="", cell=cell_source)

    # Verify exported classes and functions into user_ns
    assert "NotebookOptimizer" in shell.user_ns
    assert "helper_add" in shell.user_ns

    # Call the helper function
    assert shell.user_ns["helper_add"](10, 25) == 35

    # Run the optimizer
    opt_cls = shell.user_ns["NotebookOptimizer"]
    opt = opt_cls(epoch=3, pop_size=5)
    best = opt.solve(problem, seed=77)
    assert best is not None
    assert np.isfinite(best.fitness)


def test_ipython_registration_fallback() -> None:
    """Test load_ipython_extension fallback when register_magic_function is available."""
    registered_fns: dict[str, Any] = {}

    class FallbackShell:
        def __init__(self) -> None:
            self.user_ns: dict[str, Any] = {}

        def register_magic_function(self, func: Any, magic_kind: str, magic_name: str) -> None:
            registered_fns[f"{magic_kind}_{magic_name}"] = func

    shell = FallbackShell()
    load_ipython_extension(shell)
    assert "cell_clypto" in registered_fns
    assert callable(registered_fns["cell_clypto"])


def test_ipython_named_cell_magic() -> None:
    """Test %%clypto cell magic with explicit name argument (-n MyName)."""
    class MockInteractiveShell:
        def __init__(self) -> None:
            self.user_ns: dict[str, Any] = {}

    shell = MockInteractiveShell()
    magics = ClyptoMagics(shell=shell)

    cell_source = """# cython: language_level=3
def square_num(int x):
    return x * x
"""
    magics.clypto(line="-n NamedHelperModule", cell=cell_source)
    assert "square_num" in shell.user_ns
    assert shell.user_ns["square_num"](9) == 81


def test_ipython_positional_name_magic() -> None:
    """Test %%clypto cell magic with positional name."""
    class MockInteractiveShell:
        def __init__(self) -> None:
            self.user_ns: dict[str, Any] = {}

    shell = MockInteractiveShell()
    magics = ClyptoMagics(shell=shell)

    cell_source = """# cython: language_level=3
def triple(int x):
    return x * 3
"""
    magics.clypto(line="MyPositionalModule", cell=cell_source)
    assert "triple" in shell.user_ns
    assert shell.user_ns["triple"](7) == 21


# ---------------------------------------------------------------------------
# 4. clypto.cli tests
# ---------------------------------------------------------------------------


def test_cli_derive_class_name() -> None:
    """Test derivation of valid class names from file stems."""
    assert cli._derive_class_name("my_opt") == "MyOpt"
    assert cli._derive_class_name("123algo") == "Algo123algo"
    assert cli._derive_class_name("") == "MyOptimizer"
    assert cli._derive_class_name("CustomGA") == "CustomGA"


def test_cli_new_scaffolding(tmp_path: Any) -> None:
    """Test 'clypto new' scaffolds a valid .pyx file with appropriate boilerplate."""
    # Test path without .pyx extension
    target = tmp_path / "my_custom_algo"
    rc = cli.new_command(str(target))
    assert rc == 0

    expected_file = tmp_path / "my_custom_algo.pyx"
    assert expected_file.is_file()

    content = expected_file.read_text(encoding="utf-8")
    assert "cdef class MyCustomAlgo(cy.Optimizer):" in content
    assert "cimport clypto.core as cy" in content
    assert "cdef void evolve(self, int epoch):" in content

    # Test file already exists without force fails
    rc_fail = cli.new_command(str(expected_file), force=False)
    assert rc_fail == 1

    # Test file already exists with force succeeds
    rc_force = cli.new_command(str(expected_file), force=True)
    assert rc_force == 0


def test_cli_build_in_place(tmp_path: Any, problem: clypto.Problem) -> None:
    """Test 'clypto build' compiles a .pyx file in-place and runs the resulting extension."""
    pyx_file = tmp_path / "runner_algo.pyx"
    rc_new = cli.new_command(str(pyx_file))
    assert rc_new == 0
    assert pyx_file.is_file()

    # Compile with clypto build
    rc_build = cli.build_command(str(pyx_file), force=True)
    assert rc_build == 0

    # Ensure .so or .pyd exists in tmp_path
    so_files = [f for f in os.listdir(tmp_path) if f.endswith((".so", ".pyd"))]
    assert len(so_files) >= 1

    # Import the compiled extension and test solve()
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("runner_algo")
        algo_cls = mod.RunnerAlgo
        opt = algo_cls(epoch=5, pop_size=6, factor=0.5)
        best = opt.solve(problem, seed=42)
        assert best is not None
        assert np.isfinite(best.fitness)
    finally:
        sys.path.remove(str(tmp_path))


def test_cli_build_nonexistent_file(tmp_path: Any) -> None:
    """Test 'clypto build' on a nonexistent file returns error code."""
    fake_path = str(tmp_path / "nonexistent.pyx")
    rc = cli.build_command(fake_path)
    assert rc != 0


def test_cli_main_entrypoint(tmp_path: Any) -> None:
    """Test cli.main entrypoint with command line arguments."""
    target_pyx = tmp_path / "entrypoint_algo.pyx"
    rc_new = cli.main(["new", str(target_pyx)])
    assert rc_new == 0
    assert target_pyx.is_file()

    rc_build = cli.main(["build", str(target_pyx)])
    assert rc_build == 0
