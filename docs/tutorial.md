# Tutorial

This tutorial covers the complete workflow of using Clypto: constructing problems, configuring search space bounds, selecting algorithms, setting stopping criteria, analyzing tracking metrics, and executing evaluations in parallel.

## Defining a problem

You can define optimization problems in four ways:

### 1. Decorator API with `@cy.problem` (Recommended)

Decorate a function or class with `@cy.problem`. Setting `compile=True` automatically JIT-compiles your code to C/Cython for maximum execution speed:

#### Function-based problem
```python
import numpy as np
import clypto as cy
from clypto.native.collection.swarm_based import PSO

@cy.problem(
    bounds=cy.NumberBounds(float, low=[-10.0] * 30, up=[10.0] * 30),
    sense="min",
    compile=True,  # Compiles to native C code via Cython JIT
)
def sphere(x):
    return np.sum(x ** 2)

model = PSO.OriginalPSO(epoch=100, pop_size=50)
g_best = model.solve(sphere, seed=42)
print("Best fitness:", g_best.fitness)
```

#### Class-based problem (with state and custom `evaluate`)
Use class definitions when you need training data, configuration, or state:

```python
import numpy as np
import clypto as cy
from clypto.native.collection.system_based import AEO

@cy.problem(compile=True)
class CurveFitting:
    bounds = cy.NumberBounds(float, low=[-10.0] * 3, up=[10.0] * 3)
    sense = "min"

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def evaluate(self, solution):
        # solution is [a, b, c]
        a, b, c = solution
        predicted = a * (self.x_data ** 2) + b * self.x_data + c
        return float(np.mean((self.y_data - predicted) ** 2))

# Synthetic dataset
x = np.linspace(-5, 5, 50)
y = 2.5 * (x ** 2) - 1.2 * x + 0.5 + np.random.normal(0, 0.1, 50)

problem = CurveFitting(x_data=x, y_data=y)
model = AEO.OriginalAEO(epoch=150, pop_size=40)
best = model.solve(problem, seed=42)
```

#### Automatic Decoding with `decode=True`
For combinatorial (e.g. TSP with `PermutationBounds`), categorical or mixed problems, pass `decode=True`. Your function/method receives the already decoded representations instead of flat continuous random keys:

```python
import clypto as cy

cities = ["Madrid", "Barcelona", "Valencia", "Sevilla"]
dist_matrix = ...  # Distances matrix

@cy.problem(
    bounds=cy.PermutationBounds(cities, name="tour"),
    sense="min",
    decode=True,  # Receives decoded dictionary {'tour': ['Madrid', ...]}
    compile=True,
)
def tsp(decoded_solution):
    tour = decoded_solution["tour"]
    return calculate_route_cost(tour, dist_matrix)
```

### 2. Direct instantiation with `Problem`

Pass the objective function, the bounds, and the direction directly:

```python
import numpy as np
import clypto as cy
from clypto.native.collection.swarm_based import PSO

def sphere(solution):
    return np.sum(solution ** 2)

problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.NumberBounds(float, low=[-10.0] * 30, up=[10.0] * 30),
    sense="min",  # "min" to minimize, "max" to maximize
    name="Sphere",
)

model = PSO.OriginalPSO(epoch=100, pop_size=50)
g_best = model.solve(problem, seed=42)
print("Best fitness:", g_best.fitness)
```

### 3. Subclassing `Problem`

Subclass `Problem` directly and implement `evaluate(self, solution)` or `obj_func`:

```python
import numpy as np
from clypto import Problem, NumberBounds

class MyCustomProblem(Problem):
    def evaluate(self, solution):
        return np.sum(solution ** 2)

problem = MyCustomProblem(
    bounds=NumberBounds(float, low=[-5.0] * 10, up=[5.0] * 10),
    sense="min",
)
```

### 4. Dictionary shorthand

For quick scripts, passing a dictionary is also supported:

```python
problem = {
    "obj_func": lambda x: np.sum(x ** 2),
    "bounds": [NumberBounds(float, low=(-10.0,) * 30, up=(10.0,) * 30)],
    "sense": "min",
    "name": "SphereDict",
}
```

## Decision variables

`Problem(bounds=...)` takes a `Bounds`, one block or a list of blocks; they are
concatenated into `problem.bounds` (`low`, `up`, `n_dims`). Each block maps to a
class of problem:

| Block | Constructor sketch | Typical use |
| --- | --- | --- |
| `NumberBounds(float, ...)` | `NumberBounds(float, low=[-10.]*7, up=[10.]*7, name="delta")` | Continuous problems |
| `NumberBounds(int, ...)` | `NumberBounds(int, low=[-10]*7, up=[10]*7, name="delta")` | LP / IP / NLP / QP / MIP |
| `NumberBounds(int, 0, 1, ...)` | `NumberBounds(int, 0, 1, n_vars=11, name="delta")` | Networks (binary) |
| `NumberBounds(bool, ...)` | `NumberBounds(bool, n_vars=11, name="delta")` | ML / AI, feature toggles |
| `StringBounds` | `StringBounds(valid_sets=(("auto", "forward"),), name="delta")` | Hyper-parameters, categorical (any hashable label) |
| `PermutationBounds` | `PermutationBounds(valid_set=(-10, -4, 10, 6, -2), name="delta")` | Combinatorial optimization |
| `SequenceBounds` | `SequenceBounds(valid_sets=((1,), (2, 3)), return_type=list, name="delta")` | Hyper-parameter tuning |
| `TransferBounds` | `TransferBounds(int, n_vars=11, tf_func="vstf_04", name="delta")` | Feature selection |

The optimizer always works on an encoded, flat `float64` vector
(`problem.bounds.dtype`). Integers and booleans are searched on
`[low - 0.5, up + 0.5]` and rounded, so every value owns an equal share of the
space. Use `problem.decode_solution(x)` to recover `{name: value}` and
`problem.encode_solution(values)` for the reverse.

## Discovering optimizers

```python
import clypto as cy

cy.get_all_optimizers()             # {class_name: class} for the whole catalog
cy.get_optimizer_by_name("PSO")     # every class in the PSO module
cy.get_optimizer_by_class("BaseGA") # a single class by name
```

The same classes are importable directly from their category package:

```python
from clypto.native.collection.swarm_based import PSO
from clypto.native.collection.evolutionary_based import DE, GA
from clypto.native.collection.physics_based import MVO
```

Each algorithm lives in a module (e.g. `PSO`) that exposes one or more variants
(`OriginalPSO`, `C_PSO`, `CL_PSO`, ...). See the
[category pages](categories/index.md) for the full catalog.

## Inspecting an optimizer

```python
model = PSO.OriginalPSO(epoch=100, pop_size=50)

model.name          # "OriginalPSO"
model.parameters    # ordered hyper-parameters for this instance
```

## Writing your own optimizer

This tutorial covers the optimizers that ship with clypto. To implement your own
metaheuristic, see [Custom optimizers](custom-optimizers/index.md), which builds
one from scratch and covers the decorator and classic APIs, custom agent state,
and compilation.

## Stopping criteria

By default an optimizer stops after `epoch` generations. You can also pass a
`Termination` for function-evaluation, time, or early-stopping limits:

```python
from clypto import Termination

model = PSO.OriginalPSO(epoch=1000, pop_size=50)
model.solve(
    problem,
    termination={"max_epoch": 1000, "max_fe": 100_000, "max_time": 10},
    seed=1,
)
```

The four criteria types are **MG** (maximum generations), **FE** (maximum
function evaluations), **TB** (time bound), and **ES** (early stopping on a
fitness plateau).

## Reproducibility

Pass an integer `seed` to `solve()` explicitly — the random number generator is
not seeded otherwise:

```python
g_best_a = PSO.OriginalPSO(epoch=200, pop_size=50).solve(problem, seed=7)
g_best_b = PSO.OriginalPSO(epoch=200, pop_size=50).solve(problem, seed=7)
assert (g_best_a.solution == g_best_b.solution).all()
```

## Tracking

Pass `debug=True` to record a per-epoch history into `model.tracker`, an
in-memory numpy history. Add `track_population=True` to also record a full
population snapshot per epoch. The history lives in RAM and always describes
the most recent `solve()` call: a run that does not track leaves the tracker
empty.

```python
model = PSO.OriginalPSO(epoch=100, pop_size=50)
model.solve(problem, seed=1, debug=True)

model.tracker["global_best_fit"]   # (epoch,) numpy array
model.tracker.n_epochs             # run metadata as typed: optimizer_name, problem_name, seed ...
```

:::{note} Which optimizers track
`debug` and `track_population` belong to the engine the catalog runs on
(`cy.Optimizer`, the classic `.pyx` API). Optimizers written with the
`@cy.optimizer` decorator keep their own loop and record no history: read
the result from the returned best agent, or collect what you need yourself
inside `evolve()`.
:::

### What is recorded

With `debug=True`, `tracker[key]` returns one `(epoch,)` array per metric:

| Key | Contents |
| --- | --- |
| `global_best_fit` | best fitness found so far |
| `current_best_fit` / `current_worst_fit` | best and worst fitness of the epoch's population |
| `mean_fit` / `std_fit` | mean and standard deviation of that population's fitness |
| `diversity` | mean absolute distance from the per-dimension median of the population |
| `exploration` / `exploitation` | `diversity` rescaled to 0–100, and its complement |
| `epoch_time` | wall-clock time of the epoch, in seconds |
| `nfe` | function evaluations used so far |
| `epoch_pop_size` | number of agents in the population that epoch |

`track_population=True` adds three snapshot arrays, NaN-padded to the widest
epoch, so optimizers that grow and shrink their population (SAP_DE, ...) are
recorded without guessing a capacity in advance:

| Key | Shape |
| --- | --- |
| `solution` | `(epoch, pop, n_dims)` |
| `fitness` | `(epoch, pop)` |
| `objectives` | `(epoch, pop, n_objs)`; only when the problem has more than one objective |

The run's metadata lives on typed fields of the tracker: `optimizer_name`,
`problem_name`, `seed`, `pop_size`, `n_dims`, `sense`, `capture_population`,
`n_epochs`, and `n_objs` (`-1` unless a population was captured).
Reading a name that was not recorded raises `KeyError`, so check
`tracker.capture_population` before the snapshot arrays.

### Custom metrics with `@cy.metrics`

By default (`debug=True` without `metrics`), Clypto uses `cy.DefaultMetrics` matching `mealpy-lts` (canonical continuous diversity and exploration/exploitation).

For classical problems (permutations, discrete variables, binary transfer) or custom research metrics, define a class with `@cy.metrics`. Methods accept `positions: np.ndarray`, `fits: np.ndarray`, or both, and must return a `float`. Type annotations are validated at definition time (Pydantic-style):

```python
import numpy as np
import clypto as cy

@cy.metrics
class PermutationMetrics:
    def diversity(self, positions: np.ndarray) -> float:
        """Discrepancy for permutation vectors."""
        return float(np.mean(np.var(positions, axis=0)))

    def entropy(self, positions: np.ndarray) -> float:
        """Distribution entropy across solutions."""
        hist = np.histogram(positions, bins=10)[0]
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def fit_spread(self, fits: np.ndarray) -> float:
        """Difference between best and worst fitness of the epoch."""
        return float(np.max(fits) - np.min(fits))
```

Pass the class or instance directly to `solve(problem, metrics=...)`:

```python
# Passing metrics automatically enables tracking
model.solve(problem, metrics=PermutationMetrics)

entropy_curve = model.tracker["entropy"]
diversity_curve = model.tracker["diversity"]
best_fit_curve = model.tracker["global_best_fit"]
```

To extend the canonical `mealpy-lts` metrics with additional statistics, subclass `cy.DefaultMetrics`:

```python
@cy.metrics
class ExtendedContinuous(cy.DefaultMetrics):
    def spatial_std(self, positions: np.ndarray) -> float:
        return float(np.std(positions))
```

Metrics are resolved once at `start()` into pre-allocated memory columns with zero dictionary allocations in the per-epoch loop.

### Exporting metrics and plotting

Because tracking data is stored in memory as plain NumPy arrays, you can easily load metrics into a pandas DataFrame or plot convergence curves:

```python
import matplotlib.pyplot as plt
import pandas as pd

# Export tracked metrics to a DataFrame
metrics_df = pd.DataFrame({
    "epoch": range(len(model.tracker["global_best_fit"])),
    "best_fitness": model.tracker["global_best_fit"],
    "mean_fitness": model.tracker["mean_fit"],
    "diversity": model.tracker["diversity"],
    "epoch_time": model.tracker["epoch_time"],
})

# Plot fitness convergence and population diversity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(metrics_df["best_fitness"], label="Best Fitness", color="#2e7d32", lw=2)
ax1.plot(metrics_df["mean_fitness"], label="Mean Fitness", color="#81c784", linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Fitness")
ax1.set_title("Fitness Convergence")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

ax2.plot(metrics_df["diversity"], label="Diversity", color="#1976d2", lw=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Diversity Index")
ax2.set_title("Population Diversity")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()
```

### Per-epoch callbacks

Pass `before_iteration=` / `after_iteration=` callbacks to `solve()` to inspect
each population as it evolves. Every callback receives `(population, epoch)`
and runs around each epoch whether or not tracking is enabled:

```python
def log_epoch(population, epoch):
    print(epoch, min(agent.fitness for agent in population))


model.solve(problem, seed=1, after_iteration=log_epoch)
```

The callbacks belong to the call, not to the tracker: they never leak into a
later `solve()` that does not pass them, and the tracker is finalized even when
a callback or `evolve()` raises.

## Parallel execution

The `mode` argument picks how a batch of candidate agents is evaluated:
`mode="sequential"` (the default) evaluates agent by agent, `mode="swarm"`
evaluates the whole batch in one call, and `mode="parallel"` does the same on
OpenMP threads when the problem exposes a nogil evaluator. Algorithms that
never batch (the PSO family, among others) ignore the mode. See
[Batch and parallel evaluation](parallel-evaluation.md) for which algorithms
batch and what each mode costs.
