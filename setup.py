from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup, find_packages

init_path = Path(__file__).parent / "clypto" / "__init__.py"
init_content = init_path.read_text(encoding="utf-8")

_VERSION = "0.0.0"
for line in init_content.splitlines():
    if line.startswith("__version__"):
        _VERSION = line.split("=")[1].strip().strip("\"'")
        break

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


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


setup(
    name="clypto",
    version=_VERSION,
    description="An Open-source Library for Latest Meta-heuristic Algorithms in Cython",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Thieu, LTSIM",
    author_email="nguyenthieu2102@gmail.com, tsim@cucei.udg.mx",
    url="https://github.com/ltsim/clypto",
    project_urls={
        "Homepage": "https://github.com/ltsim/clypto",
        "Source Code": "https://github.com/ltsim/clypto",
        "Bug Tracker": "https://github.com/ltsim/clypto/issues",
        "Change Log": "https://github.com/ltsim/clypto/blob/master/CHANGELOG.md",
    },
    keywords=[
        "optimization",
        "metaheuristics",
        "MHA",
        "mathematical optimization",
        "nature-inspired algorithms",
        "evolutionary computation",
        "soft computing",
        "population-based algorithms",
        "Stochastic optimization",
        "Global optimization",
        "Convergence analysis",
        "Search space exploration",
        "Local search",
        "Computational intelligence",
        "Black-box optimization",
        "Robust optimization",
        "Hybrid algorithms",
        "Benchmark functions",
        "Metaheuristic design",
        "Performance analysis",
        "Exploration versus exploitation",
        "Self-adaptation",
        "Constrained optimization",
        "Intelligent optimization",
        "Adaptive search",
        "Simulations",
        "Algorithm selection",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "examples*"]),
    include_package_data=True,
    ext_modules=get_ext_modules(),
    install_requires=[
        "numpy>=2.0.2",
        "scipy>=1.15.3",
        "Cython>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "mypy>=1.9.0",
            "twine>=5.0",
            "flake8>=7.0",
            "pandas-stubs>=2.2.3",
            "scipy-stubs>=1.15.0",
            "black>=26.5.1",
        ],
    },
)
