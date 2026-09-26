import os
import subprocess
import sys

from Cython.Build import cythonize
from setuptools import setup

# Everything else (name, version, dependencies, classifiers, ...) lives in
# pyproject.toml's [project] table, which uv and pip both read directly.
# setup.py is kept only because cythonize() must run as code -- it can't be
# expressed declaratively in pyproject.toml.
#
# Only native .pyx is compiled: the algorithm collections (clypto/native/collection/{vectorize,legacy})
# and the Cython-only engine they cimport (clypto/optimizer/native). The public optimizer API ships
# as plain Python. CLYPTO_LEGACY=0 skips the frozen legacy collection (development builds).

COMPILER_DIRECTIVES = {
    "language_level": "3",
    "always_allow_keywords": True,
    "boundscheck": False,
    "initializedcheck": False,
    "embedsignature": True,
    "annotation_typing": False,
}


def openmp_flags():
    """Compile/link flags for OpenMP (``prange``); ``CLYPTO_OPENMP=0`` disables it.

    macOS needs libomp: the prefix comes from ``LIBOMP_PREFIX`` or Homebrew, and
    the wheel repair step (delocate) bundles the dylib into the wheel.
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
                print("clypto: libomp not found, building without OpenMP", file=sys.stderr)
                return [], []
        return (
            ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include", "-ffp-contract=off"],
            [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"],
        )
    # -ffp-contract=off keeps a*b+c unfused so results match NumPy bit for bit.
    return ["-fopenmp", "-ffp-contract=off"], ["-fopenmp"]


def get_ext_modules():
    sources = ["clypto/native/collection/vectorize/**/*.pyx", "clypto/optimizer/native/*.pyx"]
    if os.environ.get("CLYPTO_LEGACY", "1") != "0":
        sources.append("clypto/native/collection/legacy/**/*.pyx")
    extensions = cythonize(sources, compiler_directives=COMPILER_DIRECTIVES, quiet=True)
    compile_args, link_args = openmp_flags()
    for ext in extensions:
        if ext.name.startswith("clypto.native.collection.legacy."):
            # classic per-agent code never uses prange: no OpenMP, only the no-FMA flag shared with the golden baseline
            ext.extra_compile_args += [a for a in compile_args if a == "-ffp-contract=off"]
            continue
        ext.extra_compile_args += compile_args
        ext.extra_link_args += link_args
    return extensions


setup(ext_modules=get_ext_modules())
