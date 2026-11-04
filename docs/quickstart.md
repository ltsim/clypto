# Quickstart

This page gets you from zero to a solved optimization problem in seconds.

## 1. Define the objective

Write your objective function as a plain Python/NumPy callable that takes a 1D NumPy array `solution` and returns a scalar numeric value:

```python
import numpy as np

def sphere(solution):
    return np.sum(solution ** 2)
```

## 2. Build a `Problem`

`Problem` binds the objective function, the search space bounds, and the optimization direction (`"min"` or `"max"`):

```python
import clypto as cy

problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.NumberBounds(float, low=[-10.0] * 30, up=[10.0] * 30),
    sense="min",  # "min" to minimize, "max" to maximize
)
```

## 3. Pick an optimizer and solve

Optimizers are organized by category. Import the module (e.g., `PSO` from `swarm_based`) and instantiate the variant you need (such as `OriginalPSO`):

```python
from clypto.native.collection.swarm_based import PSO

model = PSO.OriginalPSO(epoch=200, pop_size=50)
g_best = model.solve(problem, seed=42)

print(f"Best solution: {g_best.solution}")
print(f"Best fitness:  {g_best.fitness}")
```

## 4. Plot the convergence curve

To track and plot convergence over time, pass `debug=True` to `solve()`. Clypto captures performance metrics in-memory inside `model.tracker` as pure NumPy arrays with zero bloat:

```python
import matplotlib.pyplot as plt

# Run with tracking enabled
model.solve(problem, seed=42, debug=True)

# Plot fitness history
history = model.tracker["global_best_fit"]
plt.figure(figsize=(8, 4))
plt.plot(history, label="Best Fitness", color="#2e7d32", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Fitness")
plt.title(f"Convergence of {model.name}")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
```

## Complete runnable script

```python
import numpy as np
import clypto as cy
from clypto.native.collection.swarm_based import PSO
import matplotlib.pyplot as plt

# 1. Define problem
def sphere(solution):
    return np.sum(solution ** 2)

problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.NumberBounds(float, low=[-10.0] * 30, up=[10.0] * 30),
    sense="min",
)

# 2. Instantiate and solve
model = PSO.OriginalPSO(epoch=200, pop_size=50)
g_best = model.solve(problem, seed=42, debug=True)

print(f"Optimizer:     {model.name}")
print(f"Best fitness:  {g_best.fitness:.6e}")
print(f"Best solution: {g_best.solution[:5]}... (dim={len(g_best.solution)})")

# 3. Inspect tracker
print(f"Total epochs:  {model.tracker.meta['n_epochs']}")
```

## Key takeaways & next steps

- **Always seed for reproducibility**: Pass `seed=int` to `solve()`.
- **Fast execution modes**: Use `mode="swarm"` to evaluate candidates as a batch or `mode="parallel"` on OpenMP threads for CPU-bound evaluators. See [Batch and parallel evaluation](parallel-evaluation.md).
- **Programmatic discovery**: Explore the 244 catalog optimizers dynamically:
  ```python
  cy.get_all_optimizers()             # {class_name: class}
  cy.get_optimizer_by_name("PSO")     # All PSO variants
  ```
- **Custom metaheuristics**: Want to write your own algorithm? Use the `@cy.optimizer` decorator with optional Cython JIT compilation in the [Custom optimizers](custom-optimizers/index.md) series.
- **Deep dive**: Read the [Tutorial](tutorial.md) for non-continuous variables, early stopping criteria, and iteration hooks.
