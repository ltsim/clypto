from Cython.Build import cythonize
from setuptools import setup

# Everything else (name, version, dependencies, classifiers, ...) lives in
# pyproject.toml's [project] table, which uv and pip both read directly.
# setup.py is kept only because cythonize() must run as code -- it can't be
# expressed declaratively in pyproject.toml.


def get_ext_modules():
    return cythonize(
        "clypto/**/*.py",
        exclude=[
            "clypto/**/__init__.py",
        ],
        compiler_directives={
            "language_level": "3",
            "always_allow_keywords": True,
            "boundscheck": False,
            # wraparound must stay True: the codebase uses negative list indexing
            # (e.g. `pop[-1]`) extensively on plain Python lists, and disabling
            # wraparound corrupts memory on those accesses once compiled.
            "wraparound": True,
            "nonecheck": False,
            "initializedcheck": False,
            "cdivision": True,
            "binding": False,
            "overflowcheck": False,
            "profile": False,
            "linetrace": False,
        },
    )


setup(ext_modules=get_ext_modules())
