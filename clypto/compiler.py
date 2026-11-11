#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Just-In-Time (JIT) Cython compilation and setuptools helpers for Clypto."""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from typing import Any, Sequence

import numpy
from Cython.Build import cythonize
from setuptools import Distribution, Extension

import clypto
from clypto.optimizer.native.legacy import LegacyOptimizer
from clypto.optimizer.native.optimizer import NativeOptimizer

Optimizer = LegacyOptimizer

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_COMPILER_DIRECTIVES",
    "openmp_flags",
    "build_extension",
    "compile",
    "compile_file",
    "compile_module",
]

DEFAULT_CACHE_DIR = os.environ.get("CLYPTO_CACHE_DIR") or os.path.expanduser("~/.cache/clypto")

DEFAULT_COMPILER_DIRECTIVES: dict[str, Any] = {
    "language_level": "3",
    "boundscheck": False,
    "cdivision": True,
    "always_allow_keywords": True,
    "cpow": True,
}

_CYTHONIZE_KEYS = {
    "nthreads",
    "quiet",
    "verbose",
    "force",
    "annotate",
    "gdb_debug",
    "build_dir",
    "output_dir",
    "cache",
}


def openmp_flags() -> tuple[list[str], list[str]]:
    """Return compile and link flags for OpenMP.

    Can be disabled by setting the environment variable ``CLYPTO_OPENMP=0``.
    """
    if os.environ.get("CLYPTO_OPENMP", "1") == "0":
        return [], []
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform == "darwin":
        prefix = os.environ.get("LIBOMP_PREFIX")
        if not prefix:
            try:
                prefix = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                return [], []
        return (
            ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include", "-ffp-contract=off"],
            [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"],
        )
    return ["-fopenmp", "-ffp-contract=off"], ["-fopenmp"]


def _augment_extension(
    ext: Extension,
    inc_dirs: list[str],
    extra_compile: list[str],
    extra_link: list[str],
) -> Extension:
    for inc in inc_dirs:
        if inc not in ext.include_dirs:
            ext.include_dirs.append(inc)
    for arg in extra_compile:
        if arg not in ext.extra_compile_args:
            ext.extra_compile_args.append(arg)
    for arg in extra_link:
        if arg not in ext.extra_link_args:
            ext.extra_link_args.append(arg)
    return ext


def _normalize_sources(
    sources: list[str] | list[Extension] | str | Extension,
    ext_name: str | None,
    inc_dirs: list[str],
    extra_compile: list[str],
    extra_link: list[str],
    **kwargs: Any,
) -> list[Extension]:
    if isinstance(sources, str):
        name = ext_name or os.path.splitext(os.path.basename(sources))[0]
        return [
            Extension(
                name,
                [sources],
                include_dirs=inc_dirs,
                extra_compile_args=extra_compile,
                extra_link_args=extra_link,
                **kwargs,
            )
        ]
    if isinstance(sources, Extension):
        return [_augment_extension(sources, inc_dirs, extra_compile, extra_link)]
    if isinstance(sources, Sequence):
        if ext_name is not None:
            return [
                Extension(
                    ext_name,
                    list(sources),  # type: ignore[arg-type]
                    include_dirs=inc_dirs,
                    extra_compile_args=extra_compile,
                    extra_link_args=extra_link,
                    **kwargs,
                )
            ]
        res: list[Extension] = []
        for s in sources:
            if isinstance(s, Extension):
                res.append(_augment_extension(s, inc_dirs, extra_compile, extra_link))
            elif isinstance(s, str):
                s_name = os.path.splitext(os.path.basename(s))[0]
                res.append(
                    Extension(
                        s_name,
                        [s],
                        include_dirs=list(inc_dirs),
                        extra_compile_args=list(extra_compile),
                        extra_link_args=list(extra_link),
                        **kwargs,
                    )
                )
            else:
                raise TypeError(f"Unsupported item in sources list: {type(s)}")
        return res
    raise TypeError(f"Unsupported sources type: {type(sources)}")


def build_extension(
    sources: list[str] | list[Extension] | str | Extension,
    **kwargs: Any,
) -> list[Extension]:
    """Helper for ``setup.py`` to configure Clypto extensions.

    Wraps sources into ``setuptools.Extension`` with Clypto and NumPy include
    directories, compiler directives, and OpenMP flags, then calls ``cythonize``.

    Parameters
    ----------
    sources : list[str] | list[Extension] | str | Extension
        Source file path(s) or pre-configured Extension object(s).
    **kwargs
        Additional arguments passed to Extension or cythonize.
        Notable options:
        - ``name`` (str): Extension module name.
        - ``include_dirs`` (list[str]): Additional include directories.
        - ``compiler_directives`` (dict): Cython compiler directives to merge.
        - ``openmp`` (bool): Whether to include OpenMP flags (default: True).
        - ``extra_compile_args`` (list[str]): Additional compiler arguments.
        - ``extra_link_args`` (list[str]): Additional linker arguments.

    Returns
    -------
    list[Extension]
        The cythonized extensions ready for setuptools.
    """
    inc_dirs = [clypto.get_include(), numpy.get_include()]
    if "include_dirs" in kwargs:
        user_inc = kwargs.pop("include_dirs")
        for inc in user_inc:
            if inc not in inc_dirs:
                inc_dirs.append(inc)

    directives = dict(DEFAULT_COMPILER_DIRECTIVES)
    if "compiler_directives" in kwargs:
        user_directives = kwargs.pop("compiler_directives")
        if user_directives:
            directives.update(user_directives)

    use_openmp = kwargs.pop("openmp", True)
    extra_compile_args = list(kwargs.pop("extra_compile_args", []))
    extra_link_args = list(kwargs.pop("extra_link_args", []))
    if use_openmp:
        omp_compile, omp_link = openmp_flags()
        for arg in omp_compile:
            if arg not in extra_compile_args:
                extra_compile_args.append(arg)
        for arg in omp_link:
            if arg not in extra_link_args:
                extra_link_args.append(arg)

    cythonize_kwargs = {
        k: kwargs.pop(k) for k in list(kwargs.keys()) if k in _CYTHONIZE_KEYS
    }
    ext_name = kwargs.pop("name", None)

    ext_list = _normalize_sources(
        sources,
        ext_name=ext_name,
        inc_dirs=inc_dirs,
        extra_compile=extra_compile_args,
        extra_link=extra_link_args,
        **kwargs,
    )

    return cythonize(ext_list, compiler_directives=directives, **cythonize_kwargs)


def _load_module_from_file(module_name: str, file_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compile_module(source: str, name: str | None = None) -> types.ModuleType:
    """JIT-compile a Cython source snippet and return the compiled module.

    Uses SHA1 digest of the source for caching in ``CLYPTO_CACHE_DIR`` or
    ``~/.cache/clypto``.

    Parameters
    ----------
    source : str
        Cython code to compile.
    name : str, optional
        Optional descriptive name for the module.

    Returns
    -------
    types.ModuleType
        The loaded compiled Cython module.
    """
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    cache_dir = DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    if name:
        clean_name = re.sub(r"\W|^(?=\d)", "_", name)
        mod_name = f"_clypto_{clean_name}_{digest[:12]}"
    else:
        mod_name = f"_clypto_jit_{digest[:16]}"

    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # Search for an existing compiled binary in cache_dir matching mod_name
    pattern = os.path.join(cache_dir, f"{mod_name}.*")
    candidates = [
        f for f in glob.glob(pattern)
        if f.endswith((".so", ".pyd")) and not f.endswith(".py")
    ]
    if candidates:
        return _load_module_from_file(mod_name, candidates[0])

    # Not cached, build extension
    pyx_file = os.path.join(cache_dir, f"{mod_name}.pyx")
    with open(pyx_file, "w", encoding="utf-8") as f:
        f.write(source)

    temp_build = tempfile.mkdtemp(prefix="clypto_build_", dir=cache_dir)
    try:
        exts = build_extension(pyx_file, name=mod_name, quiet=True)
        dist = Distribution({"name": mod_name, "ext_modules": exts})
        cmd = dist.get_command_obj("build_ext")
        cmd.build_lib = cache_dir
        cmd.build_temp = temp_build
        cmd.ensure_finalized()
        cmd.run()
        so_path = cmd.get_ext_fullpath(mod_name)
    finally:
        shutil.rmtree(temp_build, ignore_errors=True)

    if not os.path.isfile(so_path):
        found = [
            f for f in glob.glob(pattern)
            if f.endswith((".so", ".pyd")) and not f.endswith(".py")
        ]
        if not found:
            raise RuntimeError(f"Cython compilation succeeded but output binary was not found for {mod_name}")
        so_path = found[0]

    return _load_module_from_file(mod_name, so_path)


def _extract_target(module: types.ModuleType, name: str | None = None) -> Any:
    """Extract the requested class or discovered optimizer class(es) from module."""
    if name is not None:
        if hasattr(module, name):
            return getattr(module, name)
        raise AttributeError(f"Could not find attribute or class {name!r} in compiled module {module.__name__}")

    # Search for Optimizer subclasses defined in this module
    optimizers: dict[str, type] = {}
    for attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if getattr(obj, "__module__", None) == module.__name__:
            if issubclass(obj, (Optimizer, NativeOptimizer)) and obj not in (Optimizer, NativeOptimizer):
                optimizers[attr_name] = obj

    if len(optimizers) == 1:
        return next(iter(optimizers.values()))
    if len(optimizers) > 1:
        return optimizers

    # If no Optimizer subclass found, look for any class defined in module
    classes: dict[str, type] = {
        attr_name: obj
        for attr_name, obj in inspect.getmembers(module, inspect.isclass)
        if getattr(obj, "__module__", None) == module.__name__
    }
    if len(classes) == 1:
        return next(iter(classes.values()))
    if len(classes) > 1:
        return classes

    return module


def compile(source: str, name: str | None = None) -> Any:
    """JIT-compile a Cython snippet and return the optimizer class or classes.

    Generates a SHA1 hash of the source, checks the cache directory
    (``CLYPTO_CACHE_DIR`` or ``~/.cache/clypto``), builds the extension if not
    cached, imports it, and returns the optimizer class (or dict of classes).

    Parameters
    ----------
    source : str
        Cython code to compile.
    name : str, optional
        Target optimizer class name to extract. If None, automatically
        discovers optimizer classes.

    Returns
    -------
    type | dict[str, type]
        The compiled optimizer class or dict of optimizer classes.
    """
    module = compile_module(source, name=name)
    return _extract_target(module, name=name)


def compile_file(path: str | os.PathLike[str], name: str | None = None) -> Any:
    """Read and JIT-compile a ``.pyx`` file.

    Parameters
    ----------
    path : str or PathLike
        Path to the ``.pyx`` file.
    name : str, optional
        Target optimizer class name to extract.

    Returns
    -------
    type | dict[str, type]
        The compiled optimizer class or dict of optimizer classes.
    """
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return compile(source, name=name)
