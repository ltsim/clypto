# Classic API and compilation

## Write a classic algorithm

The catalog's classic algorithms are Cython extension types. `cimport
clypto.core as cy` gives `cy.Optimizer`, `cy.Agent`, `cy.Population`,
`cy.validator`, `cy.population`, the snapshots (`cy.empty_snapshot`,
`cy.snapshot`), the copies (`cy.duplicate_agent`), the bounds repair
(`cy.correct_solution`, `cy.reset_solution`, `cy.opposite_solution`) and the
helpers (`cy.is_better`, `cy.get_better_agent`, `cy.sort_agents`,
`cy.levy_flight`, ...). The constructor registers the hyper-parameters,
validates them and declares the population; `evolve` is the C method
`cdef void evolve(self, int epoch)`:

```cython
cimport clypto.core as cy


cdef class RandomSearch(cy.Optimizer):
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])

    cdef void evolve(self, int epoch):
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        cdef cy.Agent agent
        for agent in self.population.toarray():
            x = cy.correct_solution(self.problem, self.problem.generate_solution())
            n_population.append(self.population.create_agent(x))
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)
```

- `self.population` is a `cy.Population`: `size()` is the configured
  `pop_size`, `len()` the agents it holds right now, `toarray()` the live list
  of agents (a typed `for` over it is a C loop). `solve()` binds it to the
  problem and fills it before `evolve` runs. A list of agents is assigned as
  `self.population.spawn(agents)`.
- An agent holds `solution`, `objectives`, `weights` and `fitness`; the last
  three are read-only and change only through `agent.evaluate(problem)` or
  `agent.update_solution(other)`. There is no dynamic `copy()`/`update()`:
  `cy.duplicate_agent(agent)` calls the agent's static `cdef clone`.
- The pipeline `empty_snapshot` -> `evaluate(..., self.mode)` ->
  `greedy(..., self.mode)` evaluates a whole batch at once in every mode. Use it
  only when each candidate depends on the population at the start of the phase;
  an order-dependent step keeps `if self.mode == "sequential":` (see
  [Batch and parallel evaluation](../parallel-evaluation.md)).
- An algorithm with its own agents subclasses `cy.Agent` with typed
  `cdef public` fields and overrides `cdef cy.Agent clone(self)`; its
  `cy.Population` subclass builds them (`create_agent`, `generate_agent`),
  holds problem-dependent state (`bind`) and overrides `cdef void copy_state`.
  `clypto/native/collection/swarm_based/PSO/` is the reference.
- `evolve` must be a `cdef` method: a Python subclass of `cy.Optimizer` that
  defines `def evolve` is rejected by `solve()`. Write Python optimizers with
  the decorator API.

Build such a module with Cython (`cythonize(..., include_path=[<clypto source
root>], compiler_directives={"cpow": True})` and NumPy's include directory).

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
