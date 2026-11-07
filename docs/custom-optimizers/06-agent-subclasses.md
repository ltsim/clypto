# Agent subclasses and the copy protocol

The single `Agent` you have used so far carries `solution` plus the 41 state
slots of the catalog (`velocity`, `age`, ...) and one copy entry point. When your
own classic algorithm needs *extra* per-agent state beyond those slots, subclass
it and implement the two hooks `cy.duplicate_agent` dispatches on:

```{literalinclude} ../../tests/optimizer/_agent_subclass.pyx
:language: cython
:start-after: cdef class TaggedAgent(Agent):
:end-before: cdef class NaughtyAgent
```

Rules (the guards of `duplicate_agent` enforce the first two):

- `_new_instance(self)` returns a **blank instance of your own type**
  (`Agent.__new__` cannot allocate a subclass safely, so it must be overridden).
- `_copy_state_to(self, other)` calls `Agent._copy_state_to(self, other)`
  **first** — the base fields and the shared `solution` — then copies your
  fields. Skipping the base call loses the solution and is rejected at runtime.
- The `.pxd` must declare both hooks (and the fields if other modules `cimport`
  your class).
- `__cinit__` runs when the blank copy is allocated, and the base `__cinit__`
  does **not**: never draw randomness or depend on the generator there, or
  `duplicate_agent` changes the random streams of the run.
- Extras are copied by reference, like every engine field; `agent.snapshot()`
  (or `cy.duplicate_agent` + a `solution.copy()`) is the evaluation-safe
  independent copy.

The mapping from C++ is direct:

| C++ clone | Cython protocol |
| --- | --- |
| `virtual Base* clone()` | `cy.duplicate_agent(agent)` / `agent.copy()` (one call site, per-type implementation) |
| `new T(*this)` | `cdef Agent _new_instance(self)` |
| `T(const T&) : Base(rhs)` | `cdef void _copy_state_to(self, Agent other)` + the explicit `Agent._copy_state_to(...)` call |
| copy constructor overload | the engine calls `duplicate_agent`, Python callers use `agent.copy()` |

Scaffold a subclass instead of writing the boilerplate:

```bash
python tools/scaffold_agent.py --name PSOAgent \
    --state velocity:object pbest_solution:object pbest_fitness:double \
    --out clypto/native/collection/swarm_based/PSO
```

Then build the agents in the optimizer's own `create_agent` and let
`duplicate_agent` keep the dynamic type everywhere the engine copies.

```{note}
A subclass that forgets a hook fails fast and loudly, never silently:
a missing `_new_instance` raises `NotImplementedError`, a wrong return type
`TypeError`, a missing base call `RuntimeError`.
```

- Cython has no `clone()` and no `virtual` keyword: the two `cdef` hooks above
  are virtual (dispatched through the vtable), and the check at the end is the
  dispatch. Mark `XAgents` (up) an agent factory to add state in the factory instead;
  do **not** override `copy`.

## Where agents come from: the agent factory

`Population` owns the agent lifecycle; the algorithm side provides one factory
subclass and nothing else. `Population.create_agent`/`generate_agent` delegate to
`AgentFactory.create`/`AgentFactory.generate`; the base factory builds the plain
`Agent`, so 200+ algorithms need no factory at all:

```cython
cdef class PSOAgents(AgentFactory):
    cdef Agent create(self, object solution=None):
        cdef Agent agent = AgentFactory.create(self, solution)
        agent.velocity = self.generator.uniform(-self.vscale, self.vscale)
        return agent

    cdef Agent generate(self, object solution=None):
        cdef Agent agent = self.create(solution)
        agent.evaluate(self.problem)
        return agent

cdef class OriginalPSO(cy.Optimizer):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        ...
        self.population = cy.population_with_factory(pop_size, PSOAgents())

    def initialize_variables(self):
        self.population.factory = PSOAgents(self.vscale)  # problem-derived values live here
```

- `self.factory.problem` / `self.factory.generator` arrive when
  `Population.bind` runs (so they are available in `initialize_variables`
  onward, and re-derived values attach there).
- The engine's default `initialization` is `self.population.fill()`; a custom
  `initialization` stays for algorithms with pre-work (groups, reserves).
- Rule of the family: the factory draws randomness **only inside
  `create`/`generate`**, in the same order the algorithm drew it before —
  the golden parity depends on it.

---

Next: [Migrating existing code](migration.md).
