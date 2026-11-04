# Tutorial

This tutorial covers the full optimizer API: defining problems, choosing
decision variables, discovering optimizers, and controlling a run.

## Defining a problem

The quickest way is the dictionary form, which is valid for float decision
variables:

```python
import numpy as np
from clypto import NumberBounds
from clypto.native.collection.bio_based import BBO

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
from clypto.native.collection.system_based import AEO

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
model.tracker.meta["n_epochs"]     # run metadata: optimizer, problem, seed, ...
```

!!! note "Which optimizers track"

    `debug` and `track_population` belong to the engine the catalog runs on
    (`cy.Optimizer`, the classic `.pyx` API). Optimizers written with the
    `@cy.optimizer` decorator keep their own loop and record no history: read
    the result from the returned best agent, or collect what you need yourself
    inside `evolve()`.

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

`tracker.meta` describes the run: `optimizer`, `problem`, `seed`, `pop_size`,
`n_dims`, `sense`, `capture_population` and `n_epochs`, plus `n_objs` whenever
a population was captured. Reading a name that was not recorded raises
`KeyError`, so check `tracker.meta["capture_population"]` before the snapshot
arrays.

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
