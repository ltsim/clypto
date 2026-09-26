# Quickstart

This page gets you from zero to a solved optimization problem in a few lines.

## 1. Define the objective

The search space is described with one or more bounds blocks. For a continuous
problem, use `NumberBounds`:

```python
import numpy as np

def objective(solution):
    return np.sum(solution ** 2)
```

## 2. Build a `Problem`

`Problem` binds the objective function, the search space, and the optimization
direction:

```python
import clypto as cy

problem = cy.Problem(
    obj_func=objective,
    bounds=cy.NumberBounds(float, low=[-10.0] * 30, up=[10.0] * 30),
    sense="min",  # "min" to minimize, "max" to maximize
)
```

## 3. Pick an optimizer and solve

Optimizers live in category packages. Import the module (`PSO`) and instantiate
the variant you want (`OriginalPSO`):

```python
from clypto.native.collection.vectorize.swarm_based import PSO

model = PSO.OriginalPSO(epoch=500, pop_size=50)
g_best = model.solve(problem, seed=42)

print(f"Best solution: {g_best.solution}")
print(f"Best fitness:  {g_best.target.fitness}")
```

### Engines: `vectorize` and `legacy`

The collection ships twice, both Cythonized. `cy.get_all_optimizers()` and
`clypto.native.collection.vectorize.*` return the
**vectorized** classes (`clypto/native/collection/vectorize`). The classic per-agent implementations are frozen in
`clypto/native/collection/legacy` and reachable with `cy.get_all_optimizers(engine="legacy")` or
`from clypto.native.collection.legacy.swarm_based import PSO`. Class names are the same in both.

## Complete script

```python
import numpy as np
import clypto as cy
from clypto.native.collection.vectorize.swarm_based import PSO


def objective(solution):
    return np.sum(solution ** 2)


problem = cy.Problem(
    obj_func=objective,
    bounds=cy.NumberBounds(float, low=[-10.0] * 30, up=[10.0] * 30),
    sense="min",
)

model = PSO.OriginalPSO(epoch=500, pop_size=50)
g_best = model.solve(problem, seed=42)

print(f"Best solution: {g_best.solution}")
print(f"Best fitness:  {g_best.target.fitness}")
```

## What next?

- Set `seed=` on every call for reproducible results; the optimizer must be
  seeded explicitly.
- Pass `debug=True` to `solve()` to record a per-epoch history you can inspect
  afterwards.
- Not sure which algorithm to use? Browse the
  [algorithm categories](categories/index.md) or discover them programmatically:

```python
import clypto as cy

optimizers = cy.get_all_optimizers()          # {class_name: class}
optimizers = cy.get_optimizer_by_name("PSO")  # every PSO variant
```

Continue with the [Tutorial](tutorial.md) for decision variables, custom
problems, termination criteria, and more, or with
[Custom optimizers](custom-optimizers/index.md) to implement your own
metaheuristic.
