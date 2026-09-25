# Cythonized Library for Yet-another Performance Tool for Optimization 

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
![PyPI - Version](https://img.shields.io/pypi/v/clypto?style=flat-square)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/clypto?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/clypto?style=flat-square)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/clypto?style=flat-square)
![GitHub Release Date](https://img.shields.io/github/release-date/ltsim/clypto.svg?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/clypto?style=flat-square)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ltsim/clypto/publish.yml?style=flat-square&logo=pypi&label=Publish)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ltsim/clypto/test.yml?style=flat-square&logo=pytest&label=Testing)

Is a high-performance, heavily compiled fork of [MEALPY (MEta-Heuristic Algorithms using PYthon)](https://github.com/thieu1995/mealpy), stripped down to its absolute mathematical core and fully optimized via Cython.

The entire original monolithic framework has been aggressively reduced, cutting out all secondary overhead—such as visualization tools, complex logging, and heavy external dependencies—to isolate exclusively the core population-based metaheuristic optimizers (PBM). By translating these raw algorithmic structures and transition operators into native C-extensions, clypto eliminates Python's runtime friction, bypassing the GIL to deliver blazing-fast execution speeds.

This leaves a pure, ultra-lightweight, and bloat-free catalog of compiled legacy implementations. It serves as a high-speed, decoupled foundation engineered specifically to power heavy taxonomic processing and intensive benchmarks within the cmenpy ecosystem.

## Why?

* **Lean Core & Zero Bloat:** Amputated all UI, plotting, and file-writing features to create an ultra-lightweight library, making this massive historical catalog ideal to be consumed as a clean, plug-and-play mathematical dependency.
* **Aggressive Agent Refactoring:** Redesigned the original agent state model to eliminate heavy Python object/dictionary overhead, shifting towards a streamlined, flat population structure that maximizes data locality.
* **Didactic Catalog Architecture:** Re-engineered with a highly pedagogical layout. Students and researchers can open any legacy algorithm, understand its transition rules in pure Python/NumPy in just a few lines of code, and easily use the structure as a mold to study or adapt classic metaheuristics.
* **Ecosystem Ready:** Built to operate as a decoupled, standardized engine, making this collection perfectly tailored to be driven by external benchmarking, evaluation frameworks, and modern optimization wrappers.

## Credits & Citation Request

This project is based is entirely built upon the foundational work, dedication, and effort of the original [MEALPY](https://github.com/thieu1995/mealpy) authors, **Nguyen Van Thieu** and **Seyedali Mirjalili**, as well as the global community of researchers and scientists who originally designed, investigated, and implemented the diverse metaheuristic algorithms contained within this collection. This fork merely restructures their brilliant mathematical work for decoupled, lightweight, and didactic purposes.

If you use this library, its restructured architecture, or the underlying algorithms in your academic research, please ensure proper credit is given to the original creators by citing:
```bibtex 
@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@article{van2023groundwater,
  title={Groundwater level modeling using Augmented Artificial Ecosystem Optimization},
  author={Van Thieu, Nguyen and Barma, Surajit Deb and Van Lam, To and Kisi, Ozgur and Mahesha, Amai},
  journal={Journal of Hydrology},
  volume={617},
  pages={129034},
  year={2023},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.jhydrol.2022.129034}
}

@article{ahmed2021comprehensive,
  title={A comprehensive comparison of recent developed meta-heuristic algorithms for streamflow time series forecasting problem},
  author={Ahmed, Ali Najah and Van Lam, To and Hung, Nguyen Duy and Van Thieu, Nguyen and Kisi, Ozgur and El-Shafie, Ahmed},
  journal={Applied Soft Computing},
  volume={105},
  pages={107282},
  year={2021},
  publisher={Elsevier},
  doi={10.1016/j.asoc.2021.107282}
}
```

## Algorithmic Citations
To honor the individual authors who contributed each method to the field of approximate optimization, we strongly encourage you to cite the specific foundational papers for the algorithms used in your experiments.

You can find the complete list of original scientific papers and their corresponding citations in our dedicated [REFERENCES.md](/REFERENCES.md) file included within this repository.

## Goals
The sole purpose of this repository is to preserve a clean, decoupled collection of both classical and state-of-the-art nature-inspired metaheuristic algorithms. By stripping away all monolithic overhead, our goal is to provide a standardized, raw algorithmic library that can be effortlessly integrated, embedded, and reused across any Python environment—whether for academic research, proprietary commercial software, or high-performance computing clusters. 

### What you can do

- **Universal Reusability:** Import and execute any metaheuristic algorithm across any third-party framework or custom pipeline without dependency conflicts.
- **Pure Logic Inspection:** Open and study the raw mathematical transition operators of a massive catalog of population-based algorithms.
- **Zero-Friction Integration:** Consume the entire collection as a lightweight, plug-and-play mathematical engine.
- **Benchmark-Ready Sourcing:** Use these legacy implementations as standardized baselines for modern benchmarking and comparative evaluation suites.

# Usage

## Installation

### Precompiled wheels (no compiler needed)

Wheels are precompiled per platform and Python version — Linux, macOS (Intel + Apple Silicon), and Windows, for CPython 3.10–3.14 — so a plain `pip install` is all you need:

```bash
$ pip install clypto --upgrade
```

### Compiling from source

Building from source compiles the entire package to native code with Cython, so a C compiler is required. The build dependencies (`Cython`, `numpy`, `setuptools`) are fetched automatically by the PEP 517 build isolation.

* Install directly from the repository:

```bash
$ git clone https://github.com/ltsim/clypto.git
$ cd clypto
$ python -m pip install .
```

* Editable development install with [uv](https://docs.astral.sh/uv/) — compiles the extensions in place and links them into the project:

```bash
$ git clone https://github.com/ltsim/clypto.git
$ cd clypto
$ uv sync --extra dev          # or: make uv-sync
```

* Install the latest development version straight from GitHub:

```bash
$ pip install git+https://github.com/ltsim/clypto.git
```

The [Makefile](Makefile) wraps the uv workflow (`uv-lock`, `uv-sync`, `uv-test`) and offers `make clean` to wipe build, Cython, and bytecode artifacts.

### Running the tests

The algorithm collection is native Cython (`.pyx`), so build the extensions once before importing clypto from a fresh clone:

```bash
$ uv sync --extra dev
$ make build-ext            # python setup.py build_ext --inplace
$ uv run --no-sync pytest tests/   # or: make uv-test
```

The collection is built twice: `clypto/native/collection/vectorize` (whole-population NumPy/C code, the default) and
`clypto/native/collection/legacy` (the frozen classic per-agent algorithms). `CLYPTO_LEGACY=0 make build-ext` skips the legacy
tree for faster development builds; `cy.get_all_optimizers(engine="legacy")` then raises an `ImportError`.

> **Note:** a fresh clone ships `.pyx`/`.pxd` sources, not compiled extensions. Build first (`make build-ext` or `pip install .`); without a build, `import clypto` fails because the compiled collection modules do not exist.

## Writing and compiling custom optimizers

New algorithms use the decorator API. Declare hyper-parameters with
`cy.Argument`, implement `initialize`/`evolve`, and let the base class handle the
population, evaluation and the `solve` loop:

```python
import numpy as np
import clypto as cy


@cy.optimizer
class MyOptimizer:
    alpha: cy.Argument[float, (0.0, 1.0), 0.5]

    def evolve(self, epoch):
        for idx in range(len(self.population)):
            candidate = self.generate_agent()
            if candidate.fitness < self.population[idx].fitness:
                self.population[idx].solution = candidate.solution


problem = cy.Problem(
    obj_func=lambda x: np.sum(x ** 2),
    bounds=cy.FloatVar(lb=[-10.0] * 30, ub=[10.0] * 30),
    minmax="min",
)
g_best = MyOptimizer(epoch=200, pop_size=50).solve(problem, seed=7)
```

Agents can carry their own state with `@cy.agent` + `cy.Attribute`, while
classic MEALPY-style code keeps working untouched through `@cy.legacy`:

```python
import clypto as cy


@cy.legacy
class MyClassicOptimizer:
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(**kwargs)          # LegacyOptimizer is injected
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        ...
```

The classic base class was renamed `Optimizer` → `LegacyOptimizer`; `cy.Optimizer`
is kept as an alias. See the [migration guide](https://ltsim.github.io/clypto/custom-optimizers/migration/).

Compilation is opt-in. Pass `compile=True` to `@cy.optimizer`/`@cy.agent`, or
`precompile=True` to `@cy.legacy`, and install the `compile` extra:

```bash
$ pip install "clypto[compile]"
```

The decorators compile the class to a native extension on import (backed by
`pyximport`) and cache the build by source hash, so unchanged code is not
rebuilt. It fails loudly rather than running uncompiled: `ImportError` if Cython
is missing, `RuntimeError` if the source cannot be read (REPL/notebook) or
compilation fails.

## Optimizer Classification Table

* Meta-heuristic Categories: ([Based on this article](https://doi.org/10.1016/j.procs.2020.09.075))
    + Evolutionary-based: Algorithms inspired by Darwin's law of natural selection and evolutionary computing principles
    + Swarm-based: Algorithms drawing inspiration from the collective movement and interaction of swarms (e.g., birds, social insects).
    + Physics-based: Algorithms derived from physical laws and phenomena (e.g., Newton's law of universal gravitation, black holes, multiverse theory).
    + Human-based: Algorithms inspired by human interactions and behaviors (e.g., queuing search, teaching-learning processes).
    + Biology-based: Algorithms based on biological creatures or microorganisms.
    + System-based:  Algorithms inspired by ecological systems, immune systems, or network systems.
    + Math-based: Algorithms developed from mathematical forms or laws (e.g., sine-cosine functions).
    + Music-based: Algorithms drawing inspiration from musical instruments or compositions.

* Difficulty - Difficulty Level (Personal Opinion): **Objective observation from author**. Depend on the number of 
  parameters, number of equations, the original ideas, time spend for coding, source lines of code (SLOC).
    + Easy: A few paras, few equations, SLOC very short
    + Medium: more equations than Easy level, SLOC longer than Easy level
    + Hard: Lots of equations, SLOC longer than Medium level, the paper hard to read.
    + Hard* - Very hard: Lots of equations, SLOC too long, the paper is very hard to read.

For newbie, we recommend to read the paper of algorithms which difficulty is "easy" or "medium" difficulty level.

### Warning: Algorithms Suspected of Plagiarism

During our implementation and classification of metaheuristic optimization algorithms, we identified a set of methods that raise 
serious concerns regarding **scientific integrity and originality**. These algorithms are typically published under **different names**, 
but they appear to share:

- The **same core mathematical models**, equations, and update rules.
- Only superficial changes in naming, metaphors, or biological analogies.
- Publications authored by **the same or overlapping research groups**.
- **Heavy criticism** on public academic forums such as [PubPeer](https://pubpeer.com), where many of these papers are flagged for **self-plagiarism**, **redundant publication**, or **lack of novelty**.
- Some of these papers may be **withdrawn or retracted in the future**, as investigations unfold.

For these reasons, we strongly advise the **exclusion** of the following algorithms from scientific benchmarking, 
comparative studies, or any applications unless their originality is transparently validated.

**I have personally implemented these algorithms, which is why I can confidently say that they are nearly identical 
and likely cases of plagiarism. For this reason, I will no longer spend time coding such algorithms in the future. 
This warning is intended to help others avoid using or relying on these methods in their work.**

### Ethical Reminder

Researchers and students are urged to **exercise caution** when referencing or applying the algorithms listed above. 
Using unoriginal or unethical work can compromise the **scientific credibility** of any downstream research and introduce **misleading experimental results**.

> **Check [PubPeer1](https://pubpeer.com/publications/1F5DCE5BC42BF2D77A1B0C281A5295)** and [PubPeer2](https://pubpeer.com/publications/D47357D409AE273F9E03C7CBE30EB7) to 
> find ongoing discussions and critiques from the academic community.

For detailed information about the updates in each new version, see the [ChangeLog](/CHANGELOG.md) file.

---

* Maintained by: [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026
* Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=Opfunu_QUESTIONS) @ 2023
