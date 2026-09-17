# Custom optimizers

clypto ships two ways to write an optimizer. The **decorator API**
(`@cy.optimizer`) is the recommended route for new algorithms: it is compact,
validates its own hyper-parameters, and can compile itself with Cython. The
**classic API** (`@cy.legacy`) is kept unchanged for the built-in catalog and
for existing MEALPY-style code.

This series builds a complete optimizer from scratch with `@cy.optimizer`, gives
its agents extra state with `@cy.agent`, runs it, and compiles it with Cython. It
complements the reference-style [Tutorial](../tutorial.md).

Each part is self-contained but builds on the one before it.

## The problem

Every optimizer solves a `Problem`: an objective function, a search space, and a
direction. We minimize the sphere function in 30 dimensions. This same setup is
reused throughout the series.

```python
--8<-- "docs/custom-optimizers/_snippets/problem.py"
```

## The episodes

1. [Your first optimizer](01-first-optimizer.md) — `@cy.optimizer`, `evolve`, and
   static typing.
2. [Hyper-parameters](02-parameters.md) — `cy.Argument`, bounds, and validation.
3. [The population](03-population.md) — the `Population` container and derived
   fitness.
4. [Custom agent state](04-agent-state.md) — `@cy.agent` and `cy.Attribute`.
5. [Classic API and compilation](05-legacy-and-compilation.md) — `@cy.legacy` and
   the compile flags.

## Migrating existing code

Porting an existing MEALPY-style or `Optimizer`-based algorithm onto the new API
is covered in the [migration guide](migration.md).
