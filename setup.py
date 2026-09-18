from Cython.Build import cythonize
from setuptools import setup

# Everything else (name, version, dependencies, classifiers, ...) lives in
# pyproject.toml's [project] table, which uv and pip both read directly.
# setup.py is kept only because cythonize() must run as code -- it can't be
# expressed declaratively in pyproject.toml.
#
# The collection is native .pyx; the core/decorator/utils modules are still .py
# (compiled in pure-Python mode during the transition). Both globs are compiled.

COMPILER_DIRECTIVES = {
    "language_level": "3",
    "always_allow_keywords": True,
    "boundscheck": False,
    "initializedcheck": False,
    "embedsignature": True,
    "annotation_typing": False,
}


def get_ext_modules():
    return cythonize(
        ["clypto/**/*.pyx", "clypto/**/*.py"],
        exclude=[
            "clypto/**/__init__.py",
        ],
        compiler_directives=COMPILER_DIRECTIVES,
        quiet=True,
    )


setup(ext_modules=get_ext_modules())
