from setuptools import setup, find_packages
from Cython.Build import cythonize


def get_ext_modules():
    return cythonize(
        ["mealpy/**/*.py"],
        exclude=[
            "mealpy/**/__init__.py",
            "mealpy/collection/**/*.py"
        ],
        compiler_directives={
            "language_level": "3",
            "always_allow_keywords": True
        }
    )

setup(
    ext_modules=get_ext_modules(),
)
