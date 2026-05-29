import os

from setuptools import setup, Extension
from Cython.Build import cythonize


def get_ext_modules():
    if os.name == 'nt':
        extra_compile_args = ["/O2", "/Oi", "/Ot", "/GL", "/fp:fast", "/Qpar"]
        extra_link_args = ["/LTCG"]
    else:
        extra_compile_args = [
            "-O3",
            "-ffast-math",
            "-march=native",
            "-flto",
            "-funroll-loops",
            "-fno-semantic-interposition"
        ]
        extra_link_args = ["-flto"]

    extensions = [
        Extension(
            name="mealpy",
            sources=["mealpy/**/*.py"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return cythonize(
        extensions,
        exclude=[
            "mealpy/**/__init__.py",
        ],
        compiler_directives={
            "language_level": "3",
            "always_allow_keywords": True,
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "initializedcheck": False,
            "cdivision": True,
            "binding": False,
            "overflowcheck": False,
            "profile": False,
            "linetrace": False
        }
    )


setup(
    ext_modules=get_ext_modules(),
)
