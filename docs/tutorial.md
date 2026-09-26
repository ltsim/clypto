# Tutorial

This tutorial covers the full optimizer API: defining problems, choosing
decision variables, discovering optimizers, and controlling a run.

## Defining a problem

The quickest way is the dictionary form, which is valid for float decision
variables:

```python
import numpy as np
from clypto import NumberBounds
from clypto.native.collection.vectorize.bio_based import BBO

def objective(solution):
    return np.sum(solution ** 2)

problem = {
    "obj_func": objective,
    "bounds": [NumberBounds(float, low=(-10.0,) * 30, up=(10.0,) * 30)],
    "sense": "min",
    "name": "Sphere",
}

model = BBO.OriginalBBO(epoch=100, pop_size=50)
g_best = model.solve(problem, seed=1)
```

For anything more complex — custom data, non-float variables, extra logic —
subclass `Problem`. The only requirement is that `__init__` passes `bounds` and
`sense` on and that `obj_func` is defined:

```python
import numpy as np
from clypto import Problem, NumberBounds
from clypto.native.collection.vectorize.system_based import AEO

class Squared(Problem):
    def __init__(self, bounds=None, sense="min", data=None, **kwargs):
        super().__init__(bounds, sense, **kwargs)
        self.data = data

    def obj_func(self, solution):
        return np.sum(solution ** 2)

problem = Squared(
    bounds=NumberBounds(float, low=[-10.0] * 20, up=[10.0] * 20, name="my_var"),
    sense="min",
    name="Squared",
    data="anything you need",
)
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
from clypto.native.collection.vectorize.swarm_based import PSO
from clypto.native.collection.vectorize.evolutionary_based import DE, GA
from clypto.native.collection.vectorize.physics_based import MVO
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

Pass `debug=True` to record a per-epoch history into `model.tracker`, a
Zarr-backed store. Metrics include global/current best and worst fitness,
mean/std, diversity, exploration/exploitation, runtime, and function
evaluations. Add `track_population=True` to also stream a full population
snapshot per epoch. By default the store lives in memory; pass
`history_path="run.zarr"` to persist it on disk.

```python
model = PSO.OriginalPSO(epoch=100, pop_size=50)
model.solve(problem, seed=1, debug=True)

model.tracker["global_best_fit"]   # (epoch,) numpy array
model.tracker.group["solution"]    # raw Zarr array (present when track_population=True)
```

Attach per-iteration hooks to inspect each population as it evolves. The
current epoch is available as `model.tracker.epoch` while a hook runs:

```python
model.tracker.before = lambda population: ...


@model.tracker.on_after
def after_iteration(population):
    ...


model.solve(problem, seed=1, debug=True)
```

Hooks can also be passed to `solve()` as `before_iteration=` /
`after_iteration=` and only run while tracking is enabled.

## Parallel execution

The previous parallel agent-update model was removed during the 2026 refactor
because it was inefficient and broke data locality. Optimizers now run a flat,
sequential population. See the [Changelog](reference/changelog.md).
