# Classic API and compilation

## Write a classic algorithm

The catalog's classic algorithms subclass `cy.Optimizer` (`cy.LegacyOptimizer`
from Python). The constructor registers the hyper-parameters, validates them
with `cy.validator` and declares the population with `cy.population`; `evolve`
works on `self.population`:

```python
import clypto as cy


class RandomSearch(cy.LegacyOptimizer):
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])

    def evolve(self, epoch):
        pop_size = self.population.size()
        candidates = []
        for idx in range(pop_size):
            pos_new = self.population.correct_solution(self.problem.generate_solution())
            candidates.append(self.population.evaluate_solution(pos_new))
        self.population = self.population.greedy(candidates)
```

- `self.population` is a `cy.Population`: `size()` is the configured
  `pop_size`, `len()` the agents it holds right now. `solve()` binds it to the
  problem and fills it before `evolve` runs.
- An agent holds `solution`, `objectives`, `weights` and `fitness`; the last
  three are read-only and change only through `agent.evaluate(problem)` or
  `agent.update_solution(other)`.
- Agent creation, bounds repair and evaluation belong to the population
  (`create_agent`, `generate_agent`, `amend_solution`, `correct_solution`,
  `evaluate_solution`, `evaluate`). An algorithm with its own agents subclasses
  `cy.Population` and passes it as `cy.population(pop_size, range=[5, 10000], cls=MyPopulation)`;
  `cy.ResetPopulation` redraws out-of-bounds values instead of clipping them.

## The compiled form

The catalog writes the same class as a Cython extension type. `cimport
clypto.core as cy` gives `cy.Optimizer`, `cy.Agent`, `cy.Population`,
`cy.validator`, `cy.population` and the helpers (`cy.is_better`,
`cy.get_better_agent`, `cy.sort_agents`, `cy.levy_flight`, ...):

```cython
cimport clypto.core as cy


cdef class RandomSearch(cy.Optimizer):
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])

    def evolve(self, int epoch):
        ...
```

`clypto/native/collection/legacy/swarm_based/PSO/` is the reference: a
`PSOAgent` that moves itself (`update_velocity`, `move`, `update_pbest`) and a
`PSOPopulation` that builds the particles. Build such a module with Cython
(`cythonize(..., include_path=[<clypto source root>])` and NumPy's include
directory).

## Compile the decorator API

Install the `compile` extra once, then opt in per class with `compile=True`:

```bash
pip install "clypto[compile]"
```

```python
@cy.agent(compile=True)
class Particle:
    velocity: cy.Attribute[float, ..., 0.0]


@cy.optimizer(agent=Particle, compile=True)
class MyPSO:
    ...
```

The class is built into a native extension on import and cached by source hash.
Without Cython or a compiler the flags raise (`ImportError`/`RuntimeError`)
instead of silently running uncompiled.

## Next steps

- Convert existing code with the [migration guide](migration.md).
- See the full API surface in the reference [Tutorial](../tutorial.md).
- Use `seed=` on every `solve` call for reproducible runs.
