# Tutorial

This tutorial covers the full optimizer API: defining problems, choosing
decision variables, discovering optimizers, writing your own, and controlling a
run.

## Defining a problem

The quickest way is the dictionary form, which is valid for float decision
variables:

```python
import numpy as np
from clypto import FloatVar
from clypto.collection.bio_based import BBO

def objective(solution):
    return np.sum(solution ** 2)

problem = {
    "obj_func": objective,
    "bounds": [FloatVar(lb=(-10.0,) * 30, ub=(10.0,) * 30)],
    "minmax": "min",
    "name": "Sphere",
}

model = BBO.OriginalBBO(epoch=100, pop_size=50)
g_best = model.solve(problem, seed=1)
```

For anything more complex — custom data, non-float variables, extra logic —
subclass `Problem`. The only requirement is that `__init__` sets `bounds` and
`minmax` and that `obj_func` is defined:

```python
import numpy as np
from clypto import Problem, FloatVar
from clypto.collection.system_based import AEO

class Squared(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        super().__init__(bounds, minmax, **kwargs)
        self.data = data

    def obj_func(self, solution):
        return np.sum(solution ** 2)

problem = Squared(
    bounds=FloatVar(lb=[-10.0] * 20, ub=[10.0] * 20, name="my_var"),
    minmax="min",
    name="Squared",
    data="anything you need",
)
```

## Decision variables

`Problem.bounds` accepts a single `*Var` or a list of them. Each variable type
maps to a class of problem:

| Class | Constructor sketch | Typical use |
| --- | --- | --- |
| `FloatVar` | `FloatVar(lb=[-10.]*7, ub=[10.]*7, name="delta")` | Continuous problems |
| `IntegerVar` | `IntegerVar(lb=[-10]*7, ub=[10]*7, name="delta")` | LP / IP / NLP / QP / MIP |
| `StringVar` | `StringVar(valid_sets=(("auto","forward"),), name="delta")` | ML / AI hyper-parameters |
| `BinaryVar` | `BinaryVar(n_vars=11, name="delta")` | Networks |
| `BoolVar` | `BoolVar(n_vars=11, name="delta")` | ML / AI, feature toggles |
| `PermutationVar` | `PermutationVar(valid_set=(-10, -4, 10, 6, -2), name="delta")` | Combinatorial optimization |
| `CategoricalVar` | `CategoricalVar(valid_sets=((...), (...)), name="delta")` | MIP / MILP |
| `SequenceVar` | `SequenceVar(valid_sets=((1,), {2, 3}), return_type=list, name="delta")` | Hyper-parameter tuning |
| `TransferBinaryVar` | `TransferBinaryVar(n_vars=11, tf_func="vstf_04", name="delta")` | Feature selection |
| `TransferBoolVar` | `TransferBoolVar(n_vars=11, tf_func="sstf_02", name="delta")` | Feature selection |

The optimizer always works on an encoded, flat float vector. Use
`problem.decode_solution(x)` to recover real-world values and
`problem.encode_solution(x)` for the reverse.

## Discovering optimizers

```python
import clypto as cy

cy.get_all_optimizers()             # {class_name: class} for the whole catalog
cy.get_optimizer_by_name("PSO")     # every class in the PSO module
cy.get_optimizer_by_class("BaseGA") # a single class by name
```

The same classes are importable directly from their category package:

```python
from clypto.collection.swarm_based import PSO
from clypto.collection.evolutionary_based import DE, GA
from clypto.collection.physics_based import MVO
```

Each algorithm lives in a module (e.g. `PSO`) that exposes one or more variants
(`OriginalPSO`, `C_PSO`, `CL_PSO`, ...). See the
[category pages](categories/index.md) for the full catalog.

## Inspecting an optimizer

```python
model = PSO.OriginalPSO(epoch=100, pop_size=50)

model.get_name()          # "OriginalPSO"
model.get_parameters()    # ordered hyper-parameters for this instance
model.get_attributes()    # the full internal state
```

## Writing your own optimizer

clypto ships two ways to write an optimizer. The **decorator API** is the
recommended route for new algorithms: it is compact, validates its own
hyper-parameters, and can compile itself with Cython. The **classic API** is
kept unchanged for the built-in catalog and for existing MEALPY-style code.
For a hands-on walkthrough see the
[decorator optimizer API tutorial](optimizer-api.md); the full conversion recipe
lives in the [migration guide](migration.md).

### The decorator API (`@cy.optimizer`)

Decorate a plain class with `@cy.optimizer`, declare hyper-parameters with
`cy.Argument`, and implement `initialize` (optional) and `evolve`:

```python
import numpy as np
import clypto as cy


@cy.optimizer
class RandomSearch:
    # name: cy.Argument(type, bound, default)
    alpha: cy.Argument(float, (0.0, 1.0), 0.5)

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

optimizer = RandomSearch(epoch=200, pop_size=50)
g_best = optimizer.solve(problem, seed=7)
```

`epoch` and `pop_size` are built in; every other declared `Argument` becomes a
validated constructor parameter with its default. Bounds follow the validator
convention (`tuple` exclusive, `list` inclusive), and unknown or out-of-range
arguments raise immediately.

Inside the optimizer the base class exposes:

| Attribute | Meaning |
| --- | --- |
| `self.population` | The `Population` container (see below) |
| `self.rng` | A seeded `numpy.random.Generator` (`seed=` from `solve`) |
| `self.problem` | The bound `Problem` |
| `self.bounds` | `lb`/`ub`/`ndim` view of the search space |
| `self.g_best` | Best agent after `solve` returns |

The `Population` gives you the whole solution matrix, the fitness vector, and
the best/worst agents:

```python
self.population.solutions = self.rng.uniform(
    self.bounds.lb, self.bounds.ub, (len(self.population), self.bounds.ndim)
)                                    # assigning re-evaluates every agent
pbest, pworst = self.population.best, self.population.worst
self.population.remove(pworst.id)
self.population.append(self.population.generate())
```

Agents expose `solution`, `fitness`, `target` and `id`. `fitness` is
**read-only** and always derived from the objective: assigning `solution`
recomputes it automatically, so it can never go stale. In-place maths works too,
because `population[n].solution /= 2` routes through the same setter.

### Custom agent attributes (`@cy.agent`)

By default solutions carry no extra state. If an algorithm needs per-agent
attributes (velocity, memory, tags, ...), declare an agent class:

```python
@cy.agent
class MyAgent:
    v: cy.Attribute(float, (0.0, 1.0), 0.5)


@cy.optimizer(agent=MyAgent)
class MyOptimizer:
    def evolve(self, epoch):
        for agent in self.population:
            agent.solution = agent.solution * (1 - agent.v)


optimizer = MyOptimizer(epoch=100, pop_size=40)
```

`cy.Attribute` mirrors `cy.Argument` for agents. To seed attributes from a
distribution, override `generate_agent`:

```python
def generate_agent(self, solution=None):
    agent = super().generate_agent(solution)
    agent.v = self.rng.uniform(0, 1)
    return agent
```

### The classic API (`@cy.legacy`)

Existing MEALPY-style optimizers keep working unchanged. Decorate the class
instead of naming the base; `super().__init__(**kwargs)`, `self.validator`,
`self.pop`, and the whole classic feature set are still available:

```python
import clypto as cy


@cy.legacy
class RandomSearch:
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(self.problem.generate_solution(encoded=True))
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.get_target(pos_new)
            self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
```

`@cy.legacy` injects `LegacyOptimizer` as a base (no inheritance needed) and
also accepts a `precompile` flag.

### Compiling a custom optimizer

Both decorators can compile the class to a native extension at import time (via
`pyximport`): pass `compile=True` to `@cy.optimizer` / `@cy.agent`, or
`precompile=True` to `@cy.legacy`:

```python
@cy.optimizer(agent=MyAgent, compile=True)
class MyOptimizer:
    ...
```

Compilation is opt-in and needs the `compile` extra
(`pip install "clypto[compile]"`) plus a C compiler. It fails loudly rather than
running uncompiled: `ImportError` when Cython is missing, `RuntimeError` when the
source is unavailable (REPL/notebook) or compilation fails. Builds are cached by
source hash, so unchanged code is not rebuilt.

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
