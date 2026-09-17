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

Subclass `Optimizer` and implement `evolve`. The base class already provides
population creation, evaluation, best/worst tracking, and `solve`:

```python
from clypto.optimizer import Optimizer


class RandomSearch(Optimizer):
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True
        self.is_parallelizable = False

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(
                self.problem.generate_solution(encoded=True)
            )
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.get_target(pos_new)
            self.pop[idx] = self.get_better_agent(
                self.pop[idx], agent, self.problem.minmax
            )
```

The execution lifecycle is: `check_problem` → `initialize_variables` →
`before_initialization` → `initialization` → `after_initialization` →
`before_main_loop` → `evolve(epoch)` (repeated) → `track_optimize_process`.

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

!!! warning "Termination is currently unreliable"

    In this build, passing any `termination` triggers a code path that
    references an uninitialized history buffer, so a `Termination` object or
    dict will raise at runtime, and the dict form additionally reads
    `log_to`/`log_file` attributes that `Problem` does not define. Prefer the
    default epoch-based stopping until this is fixed. See
    [Known Issues](issues.md).

## Reproducibility

Pass an integer `seed` to `solve()` explicitly — the random number generator is
not seeded otherwise:

```python
g_best_a = PSO.OriginalPSO(epoch=200, pop_size=50).solve(problem, seed=7)
g_best_b = PSO.OriginalPSO(epoch=200, pop_size=50).solve(problem, seed=7)
assert (g_best_a.solution == g_best_b.solution).all()
```

## Debug flag

`solve(..., debug=True)` is accepted for API compatibility, but per-epoch
history tracking (`track_optimize_step` / `track_optimize_process`) is currently
stubbed out in this stripped build, so no history is recorded. Node-level
results remain available through `model.g_best` and `model.g_worst`.

## Parallel execution

The previous parallel agent-update model was removed during the 2026 refactor
because it was inefficient and broke data locality. Optimizers now run a flat,
sequential population. See the [Changelog](reference/changelog.md).
