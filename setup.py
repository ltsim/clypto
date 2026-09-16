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
            "cdivision": True,
        },
    )


setup(ext_modules=get_ext_modules())
