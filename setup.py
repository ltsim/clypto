import os
import subprocess
import sys

import numpy
from Cython.Build import cythonize
from setuptools import setup

# Everything else (name, version, dependencies, classifiers, ...) lives in
# pyproject.toml's [project] table, which uv and pip both read directly.
# setup.py is kept only because cythonize() must run as code -- it can't be
# expressed declaratively in pyproject.toml.
#
# Everything is compiled: the algorithm collection (clypto/native/collection/**) and the whole
# optimizer package (clypto/optimizer/**, engine and authoring API; .pyi stubs keep the public
# modules typed). CLYPTO_COLLECTION=0 skips the collection (development builds of the core).
# ponytail: the authoring API runs once per solve(), so compiling it buys no speed; it is .pyx only so
# the core is one language.

COMPILER_DIRECTIVES = {
    "language_level": "3",
    "always_allow_keywords": True,
    "boundscheck": False,
    "initializedcheck": False,
    "embedsignature": True,
    "annotation_typing": False,
    # C ``double ** double`` is a real ``pow()`` (Python floats' result); Cython's default types it as
    # complex, which is slower and 1 ulp off whenever evolve's C ``int epoch`` makes the operands C types.
    "cpow": True,
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
    sources = ["clypto/optimizer/**/*.pyx"]
    if os.environ.get("CLYPTO_COLLECTION", "1") != "0":
        sources.append("clypto/native/collection/**/*.pyx")
    extensions = cythonize(sources, compiler_directives=COMPILER_DIRECTIVES, quiet=True)
    compile_args, link_args = openmp_flags()
    for ext in extensions:
        # agents and populations type their arrays as numpy.ndarray (cimport numpy)
        ext.include_dirs.append(numpy.get_include())
        ext.define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
        if ext.name.startswith("clypto.native.collection."):
            # algorithms never use prange (parallel evaluation lives in the core): only the no-FMA flag of the golden baseline
            ext.extra_compile_args += [a for a in compile_args if a == "-ffp-contract=off"]
            continue
        ext.extra_compile_args += compile_args
        ext.extra_link_args += link_args
    return extensions


setup(ext_modules=get_ext_modules())
