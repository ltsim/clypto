# clypto

**Cythonized Library for Yet-another Performance Tool for Optimization** — a
high-performance, heavily compiled fork of
[MEALPY](https://github.com/thieu1995/mealpy), stripped down to its absolute
mathematical core and optimized with Cython.

The original monolithic framework has been aggressively reduced, cutting out all
secondary overhead — visualization, complex logging, and heavy external
dependencies — to isolate the core population-based metaheuristic optimizers
(PBM). Translating the raw algorithmic structures and transition operators into
native C-extensions removes Python's runtime friction and bypasses the GIL.

What remains is a pure, ultra-lightweight, bloat-free catalog of compiled legacy
implementations: a high-speed, decoupled foundation built to power heavy
taxonomic processing and intensive benchmarks in the `cmenpy` ecosystem.

- **Get running in minutes** — Install from a precompiled wheel and solve your first problem with the [Quickstart](quickstart.md).
- **Learn the API** — Problems, decision variables, termination, and seeding are covered in the [Tutorial](tutorial.md).
- **Write your own optimizer** — Implement a custom metaheuristic from scratch with the [Custom algorithms](custom_algorithms.md) guide and the [Custom optimizers](custom-optimizers/index.md) series.
- **Browse the catalog** — Explore all **244 optimizers across 147 modules**, grouped by [taxonomy category](categories/index.md).
- **Check before you benchmark** — Review [Known Issues](issues.md) — weak, buggy, or plagiarism-suspect algorithms, grouped by category.

## Why clypto?

- **Lean core & zero bloat** — all UI, plotting, and file-writing features were
  amputated, leaving an ultra-lightweight catalog that is easy to consume as a
  clean, plug-and-play mathematical dependency.
- **Aggressive agent refactoring** — the original agent state model was redesigned
  into a streamlined, flat population structure that maximizes data locality.
- **Didactic catalog architecture** — every legacy algorithm reads as a few lines
  of pure Python/NumPy transition rules, making it a good mold for study and
  adaptation.
- **Ecosystem ready** — a decoupled, standardized engine tailored to external
  benchmarking, evaluation frameworks, and modern optimization wrappers.

## Attribution

This project is built entirely upon the foundational work of the original
[MEALPY](https://github.com/thieu1995/mealpy) authors, **Nguyen Van Thieu** and
**Seyedali Mirjalili**, and the global community of researchers who designed and
implemented the algorithms in this collection. Please cite the original work —
see [Citations](reference/citations.md).

:::{warning} Scientific integrity
A number of algorithms ship with concerns about originality. See
[Known Issues](issues.md) before using or benchmarking them.
:::

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
tutorial
parallel-evaluation
```

```{toctree}
:maxdepth: 2
:caption: Custom Optimizers

custom_algorithms
custom-optimizers/index
```

```{toctree}
:maxdepth: 2
:caption: Algorithm Categories

categories/index
```

```{toctree}
:maxdepth: 1
:caption: Project & Quality

issues
audit
releasing
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/core
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/citations
reference/changelog
```
