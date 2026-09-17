#!/usr/bin/env python
"""Cache location, Cython detection, and pyximport setup for ``precompile``."""
import hashlib
import os
import sys
import tempfile

import numpy

__all__ = [
    "CACHE_DIR",
    "cython_version",
    "install_pyximport",
    "source_digest",
    "pyx_path",
]

CACHE_DIR = os.environ.get(
    "CLYPTO_PRECOMPILE_DIR",
    os.path.join(tempfile.gettempdir(), "clypto-precompiled"),
)
_INSTALLED = False


def cython_version() -> str:
    try:
        import Cython
    except ImportError as exc:
        raise ImportError(
            "clypto.precompile requires Cython at runtime. "
            'Install it with: pip install "clypto[compile]"'
        ) from exc

    return Cython.__version__


def install_pyximport() -> None:
    global _INSTALLED

    if _INSTALLED:
        return

    import pyximport

    pyximport.install(
        build_dir=CACHE_DIR,
        build_in_temp=False,
        language_level="3",
        setup_args={"include_dirs": [numpy.get_include()]},
    )

    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)

    _INSTALLED = True


def source_digest(source: str, version: str) -> str:
    return hashlib.sha1(f"{source}\n{sys.version}\n{version}".encode()).hexdigest()[:16]


def pyx_path(module_name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)

    return os.path.join(CACHE_DIR, module_name + ".pyx")
