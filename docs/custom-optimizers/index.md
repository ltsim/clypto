# Custom optimizers

clypto ships two ways to write an optimizer. The **decorator API**
(`@cy.optimizer`) is the recommended route for new algorithms: it is compact,
validates its own hyper-parameters, and can compile itself with Cython. The
**classic API** (subclassing `cy.LegacyOptimizer`) is what the built-in catalog
uses.

This series builds a complete optimizer from scratch with `@cy.optimizer`, gives
its agents extra state with `@cy.agent`, runs it, and compiles it with Cython. It
complements the reference-style [Tutorial](../tutorial.md).

Each part is self-contained but builds on the one before it.

## The problem

Every optimizer solves a `Problem`: an objective function, a search space, and a
direction. We minimize the sphere function in 30 dimensions. This same setup is
reused throughout the series.

```{literalinclude} _snippets/problem.py
:language: python
```

## The episodes

1. [Your first optimizer](01-first-optimizer.md) — `@cy.optimizer`, `evolve`, and
   static typing.
2. [Hyper-parameters](02-parameters.md) — `cy.Argument`, bounds, and validation.
3. [The population](03-population.md) — the `Population` container and derived
   fitness.
4. [Custom agent state](04-agent-state.md) — `@cy.agent` and `cy.Attribute`.
5. [Classic API and compilation](05-legacy-and-compilation.md) — `cy.LegacyOptimizer` and
   the compile flags.
6. [Agent subclasses](06-agent-subclasses.md) — extra per-agent state with
   `duplicate_agent`'s copy protocol.

## Migrating existing code

Porting an existing MEALPY-style or `Optimizer`-based algorithm onto the new API
is covered in the [migration guide](migration.md).

```{toctree}
:maxdepth: 1
:hidden:

01-first-optimizer
02-parameters
03-population
04-agent-state
05-legacy-and-compilation
06-agent-subclasses
migration
```
