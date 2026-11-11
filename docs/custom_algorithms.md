# Custom Algorithms in Clypto

Clypto is designed to execute population-based metaheuristic optimization algorithms with minimal overhead. While Clypto includes a catalog of built-in optimizers, it also provides a framework for creating, compiling, and reusing **custom optimization algorithms**.

You can develop custom algorithms using:

1. **Pure Python** — via the declarative `@cy.optimizer` decorator or direct inheritance from `cy.Optimizer` for rapid development and experimentation.
2. **Native Cython (`.pyx`)** — by subclassing `cy.Optimizer` with `cimport clypto.core as cy` to achieve C-level execution speed, typed loops, and zero GIL contention.
3. **Just-In-Time (JIT) Compilation** — using `clypto.compile()` and `clypto.compile_file()` with automatic caching.
4. **Interactive Notebooks** — using `%load_ext clypto` and the `%%clypto` cell magic in Jupyter.
5. **Ahead-Of-Time (AOT) Extension Packaging** — using `clypto.build_extension()` in your package's `setup.py`.
6. **Command-Line Interface (CLI)** — using `clypto new` and `clypto build` to scaffold and compile algorithms directly from your terminal.

---

## 1. Writing Custom Algorithms in Pure Python

Clypto provides two ways to author custom algorithms in pure Python without writing Cython or C code: the **Decorator API (`@cy.optimizer`)** and **Subclassing `cy.Optimizer`**.

### 1.1 The Decorator API (`@cy.optimizer`)

The `@cy.optimizer` decorator turns a plain Python class into a full-featured Clypto optimizer. Hyperparameters are declared statically using `cy.Argument`, which automatically validates values and provides constructor defaults.

#### Argument Declarations

`cy.Argument` supports bracketed typing declarations:

- `cy.Argument[dtype]` — Typed argument with no bounds check.
- `cy.Argument[dtype, bound]` — Validates that value lies in `bound` (`[...]` inclusive, `(...)` exclusive).
- `cy.Argument[dtype, bound, default]` — Validates with bound and sets a default value.
- `cy.Argument[dtype, ..., default]` — Unbounded argument with a default value (`...` indicates no bounds).

#### Injected Attributes & Lifecycle

When decorated with `@cy.optimizer`, your class automatically receives:

| Property / Method | Description |
|---|---|
| `self.population` | The active `Population` container holding agents. |
| `self.bounds` | Lower/upper bounds and dimensionality (`n_dims`, `lb`, `ub`). |
| `self.rng` | A seeded `numpy.random.Generator` instance (set via `solve(..., seed=...)`). |
| `self.problem` | The bound `Problem` instance being solved. |
| `self.g_best` | The best `Agent` found during the search. |
| `self.generate_agent()` | Generates a new random agent within problem bounds. |
| `initialize(self)` | *(Optional)* Lifecycle hook executed once before evolution begins. |
| `evolve(self, epoch)` | *(Required)* Algorithm logic executed at each generation/epoch. |

#### Complete Example: Gaussian Mutation Search

```python
import numpy as np
import clypto as cy

# 1. Define the objective problem
def sphere(solution: np.ndarray) -> float:
    return float(np.sum(solution**2))

problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.NumberBounds(float, low=[-5.0] * 5, up=[5.0] * 5),
    sense="min",
)

# 2. Define the custom optimizer with @cy.optimizer
@cy.optimizer
class GaussianMutationSearch:
    step_size: cy.Argument[float, (0.0, 2.0), 0.2]
    mutation_rate: cy.Argument[float, (0.0, 1.0), 0.5]

    def initialize(self):
        """Optional hook called once before the main loop."""
        pass

    def evolve(self, epoch: int):
        """Executed once per iteration."""
        for agent in self.population:
            if self.rng.random() < self.mutation_rate:
                # Perturb solution with Gaussian noise
                noise = self.rng.normal(0, self.step_size, size=self.bounds.n_dims)
                candidate_solution = self.bounds.correct(agent.solution + noise)
                candidate = self.generate_agent(candidate_solution)

                # Greedy selection: updating agent.solution automatically re-evaluates fitness
                if candidate.fitness < agent.fitness:
                    agent.solution = candidate.solution

# 3. Instantiate and run
opt = GaussianMutationSearch(epoch=50, pop_size=20, step_size=0.15)
g_best = opt.solve(problem, seed=42)

print(f"Best solution: {g_best.solution}")
print(f"Best fitness:  {g_best.fitness:.6e}")
```

> [!TIP]
> You can JIT-compile a decorated class directly by adding `compile=True`:
> ```python
> @cy.optimizer(compile=True)
> class FastGaussianSearch:
>     step_size: cy.Argument[float, (0.0, 2.0), 0.2]
>     ...
> ```
> This automatically generates and compiles a Cython C-extension module in the background on first import.

---

### 1.2 Subclassing `cy.Optimizer` in Pure Python

If you prefer classic object-oriented inheritance without class decorators, subclass `cy.Optimizer` (or `cy.DecoratedOptimizer` when using static type checkers like `mypy`):

```python
import numpy as np
import clypto as cy

def sphere(solution: np.ndarray) -> float:
    return float(np.sum(solution**2))

problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.NumberBounds(float, low=[-5.0] * 5, up=[5.0] * 5),
    sense="min",
)

class StepDecayOptimizer(cy.Optimizer):
    """Custom pure-Python optimizer subclassing cy.Optimizer."""

    def __init__(self, epoch: int = 50, pop_size: int = 20, decay: float = 0.85, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.decay = decay

    def initialize(self):
        """Optional hook called once before evolution."""
        pass

    def evolve(self, epoch: int):
        """Contract each agent's vector toward the origin and re-evaluate."""
        for agent in self.population:
            agent.solution = self.bounds.correct(agent.solution * self.decay)
            agent.evaluate(self.problem)

opt = StepDecayOptimizer(epoch=30, pop_size=15, decay=0.9)
g_best = opt.solve(problem, seed=123)

print(f"Best solution: {g_best.solution}")
print(f"Best fitness:  {g_best.fitness:.6e}")
```

#### Custom Agent State in Pure Python

To store extra state per agent (such as velocity or local best), override the `generate(self, agent, solution)` hook or use `@cy.agent` with `@cy.optimizer(agent=CustomAgent)`:

```python
import clypto as cy

@cy.agent
class ParticleAgent:
    velocity: cy.Attribute[float, ..., 0.0]

@cy.optimizer(agent=ParticleAgent)
class ParticleSwarm:
    def generate(self, agent, solution):
        # Initialize custom agent attributes
        agent.velocity = 0.1
        return agent

    def evolve(self, epoch: int):
        for agent in self.population:
            agent.solution = self.bounds.correct(agent.solution + agent.velocity)
            agent.evaluate(self.problem)
```

---

## 2. Writing Custom Algorithms in Cython (`.pyx`)

For maximum execution speed, author your algorithms in Cython (`.pyx` files). Cython eliminates Python interpreter overhead, enables typed C-level iteration over agents, and avoids GIL contention.

### 2.1 The `cimport clypto.core as cy` Interface

Clypto exposes its native Cython API through `clypto/core.pxd`. Import it into your Cython module with:

```cython
# cython: language_level=3
cimport clypto.core as cy
import numpy as np
```

The `cy` namespace provides:
- **`cy.Optimizer`** — Base C extension class for optimizers.
- **`cy.Agent`** — Native agent type holding `agent.solution`, `agent.fitness`, `agent.objectives`.
- **`cy.Population`** — Contiguous population container.
- **`cy.validator(type, value, bounds, name)`** — High-speed C parameter validator.
- **`cy.correct_solution(problem, solution)`** — Boundary repair function.
- **`cy.duplicate_agent(agent)`** — Fast agent duplication protocol.
- **`cy.empty_snapshot(population)`** / **`cy.snapshot(population)`** — Snapshot utilities for synchronous batch updates.

### 2.2 Native Cython Algorithm Template

Create a file named `contraction_algo.pyx`:

```cython
# cython: language_level=3
cimport clypto.core as cy
import numpy as np


cdef class ContractionOptimizer(cy.Optimizer):
    """Custom Cython optimizer performing geometric contraction."""

    cdef public double factor
    cdef public bint initialized

    def __init__(self, epoch=50, pop_size=30, factor=0.7, **kwargs):
        # Register hyperparameters and validate bounds
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")
        self.initialized = False

    def initialize(self):
        """Optional hook executed once before the main loop."""
        self.initialized = True

    cdef void evolve(self, int epoch):
        """Core algorithm logic executed at C-speed for each iteration."""
        cdef cy.Agent agent
        for agent in self.population:
            # Update solution vector in place
            agent.solution = agent.solution * self.factor
            # Re-evaluate agent against objective problem
            agent.evaluate(self.problem)
```

> [!IMPORTANT]
> In Cython `cdef class` optimizers:
> - `evolve` must be declared as `cdef void evolve(self, int epoch):`.
> - Always declare `cdef cy.Agent agent` to ensure C-level iteration over `self.population`.
> - Use `cy.validator(...)` to validate constructor hyperparameters.

---

## 3. Compiling and Reusing Custom Algorithms

Clypto includes a Just-In-Time (JIT) Cython compiler (`clypto.compile` and `clypto.compile_file`) that builds and caches C extensions on the fly.

### 3.1 Inline String Compilation (`clypto.compile`)

You can compile inline Cython code directly from a Python string. Clypto hashes the source code with SHA1 and stores the compiled binary in `~/.cache/clypto` (or the path set by `CLYPTO_CACHE_DIR`). Repeated calls with unchanged source instantly return the cached module.

```python
import numpy as np
import clypto as cy

cython_code = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class JITOptimizer(cy.Optimizer):
    cdef public double factor

    def __init__(self, epoch=30, pop_size=15, factor=0.5, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")

    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * self.factor
            agent.evaluate(self.problem)
"""

# Compile and retrieve the optimizer class
JITOptimizer = cy.compile(cython_code)

# Define problem and solve
problem = cy.Problem(
    obj_func=lambda x: float(np.sum(x**2)),
    bounds=cy.NumberBounds(float, low=[-5.0] * 3, up=[5.0] * 3),
    sense="min",
)

optimizer = JITOptimizer(epoch=20, pop_size=10, factor=0.4)
best = optimizer.solve(problem, seed=42)

print(f"JIT Optimizer Best Fitness: {best.fitness:.6e}")
```

#### Compiling Multiple Classes

When compiling a code snippet containing multiple optimizer classes:
- Without `name`: `cy.compile(source)` returns a `dict` mapping class names to class types:
  ```python
  classes = cy.compile(source)
  AlgoA = classes["AlgoA"]
  AlgoB = classes["AlgoB"]
  ```
- With `name`: `cy.compile(source, name="AlgoA")` extracts and returns that specific class directly.

---

### 3.2 File Compilation (`clypto.compile_file`)

To compile and load an external `.pyx` file from disk:

```python
import clypto as cy

# Compiles 'contraction_algo.pyx' and returns the optimizer class
ContractionOptimizer = cy.compile_file("contraction_algo.pyx")

# Solve problem
opt = ContractionOptimizer(epoch=50, pop_size=20, factor=0.6)
best = opt.solve(problem, seed=99)
print(f"Best: {best.fitness}")
```

---

## 4. Building Packages with `clypto.build_extension`

When distributing custom algorithms as installable Python packages or compiling ahead-of-time (AOT) for production, use `clypto.build_extension` inside your `setup.py`.

### 4.1 What `build_extension` Configures

`clypto.build_extension` takes care of:
1. Adding Clypto's include directory (`clypto.get_include()`) containing `clypto/core.pxd`.
2. Adding NumPy's C include directory (`numpy.get_include()`).
3. Setting optimal Cython compiler directives:
   - `language_level = "3"`
   - `boundscheck = False`
   - `cdivision = True`
   - `cpow = True`
   - `always_allow_keywords = True`
4. Injecting platform-specific OpenMP compile and link flags (`-fopenmp` or `/openmp`).

### 4.2 Example `setup.py`

```python
from setuptools import setup
from clypto.compiler import build_extension

setup(
    name="my_optimizer_package",
    version="0.1.0",
    description="Custom Clypto algorithms",
    ext_modules=build_extension(
        sources=["my_algorithms/advanced_pso.pyx", "my_algorithms/mutation_search.pyx"],
        openmp=True,  # Set False or export CLYPTO_OPENMP=0 to disable OpenMP
    ),
)
```

Compile extensions in place:
```bash
python setup.py build_ext --inplace
```

Once built, import your compiled algorithms like standard Python modules:
```python
from my_algorithms.advanced_pso import AdvancedPSO
```

---

## 5. Using Clypto in Jupyter Notebooks (`%%clypto`)

Clypto integrates directly with Jupyter Notebooks and IPython environments via the `%load_ext clypto` extension and `%%clypto` cell magic.

### 5.1 Setting Up the Notebook

In the first notebook cell, load the Clypto extension:

```python
%load_ext clypto
```

### 5.2 The `%%clypto` Cell Magic

Prefix any cell with `%%clypto` to compile native Cython code in-memory. The cell magic automatically adds Clypto and NumPy include directories, compiles the code with OpenMP support, and imports all defined classes and functions directly into your notebook's user namespace:

```python
%%clypto
# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class NotebookSwarm(cy.Optimizer):
    """Custom Cython optimizer compiled interactively inside Jupyter."""
    cdef public double contraction

    def __init__(self, epoch=50, pop_size=25, contraction=0.8, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.contraction = cy.validator(float, contraction, (0.0, 1.0), "contraction")

    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            # Update solution vector
            agent.solution = agent.solution * self.contraction
            # Re-evaluate
            agent.evaluate(self.problem)
```

In the next cell, immediately use `NotebookSwarm`:

```python
import numpy as np
import clypto as cy

problem = cy.Problem(
    obj_func=lambda x: float(np.sum(x**2)),
    bounds=cy.NumberBounds(float, low=[-5.0] * 5, up=[5.0] * 5),
    sense="min",
)

model = NotebookSwarm(epoch=30, pop_size=20, contraction=0.7)
g_best = model.solve(problem, seed=42)

print("Best solution found in notebook:", g_best.solution)
print("Best fitness:", g_best.fitness)
```

> [!TIP]
> You can pass an optional name parameter to the magic:
> ```python
> %%clypto -n MyCustomModule
> ```

---

## 6. Command-Line Interface (`clypto` CLI)

Clypto includes a command-line interface with tools for scaffolding and compiling custom algorithms.

### 6.1 `clypto new` — Scaffold an Algorithm Template

The `clypto new` command generates a ready-to-use Cython algorithm `.pyx` file with proper typing, constructor validation, and evolution hooks:

```bash
clypto new my_custom_pso.pyx
```

Output:
```text
Created template at 'my_custom_pso.pyx' (class: MyCustomPso)
```

Generated `my_custom_pso.pyx`:
```cython
# cython: language_level=3
cimport clypto.core as cy
import numpy as np


cdef class MyCustomPso(cy.Optimizer):
    """Custom optimization algorithm implemented in Cython.

    Subclasses Clypto's native optimizer core for high-performance execution.
    """

    cdef public double factor

    def __init__(self, epoch=100, pop_size=30, factor=0.5, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")

    def initialize(self):
        """Optional hook executed once before the main optimization loop."""
        pass

    cdef void evolve(self, int epoch):
        """Core algorithm logic executed at each iteration/epoch."""
        cdef cy.Agent agent
        for agent in self.population:
            # Update solution vector
            agent.solution = agent.solution * self.factor
            # Re-evaluate agent against the problem
            agent.evaluate(self.problem)
```

Flags:
- `-f, --force` — Overwrite target file if it already exists.

---

### 6.2 `clypto build` — Compile In-Place

Compile any `.pyx` algorithm file in-place into a loadable shared library (`.so` on Linux/macOS, `.pyd` on Windows):

```bash
clypto build my_custom_pso.pyx
```

Output:
```text
Compiling 'my_custom_pso.pyx' in-place...
Successfully compiled: /path/to/my_custom_pso.cpython-314-x86_64-linux-gnu.so
```

Once compiled, you can import and run the optimizer in Python:

```python
import my_custom_pso
import clypto as cy

problem = cy.Problem(
    obj_func=lambda x: float((x**2).sum()),
    bounds=cy.NumberBounds(float, low=[-5.0] * 3, up=[5.0] * 3),
    sense="min",
)

optimizer = my_custom_pso.MyCustomPso(epoch=20, pop_size=10, factor=0.5)
best = optimizer.solve(problem, seed=42)
print("Result:", best.fitness)
```

Flags:
- `-f, --force` — Force recompilation even if binary exists.

---

## 7. Best Practices & Performance Optimization

1. **Type Agent Variables in Loops**
   In Cython `.pyx` files, always declare `cdef cy.Agent agent` prior to looping over `self.population`. Untyped iteration falls back to Python object calls and slows down execution.

2. **Always Seed the RNG**
   Clypto algorithms use deterministic random number generators. Always pass `seed=...` to `model.solve(problem, seed=42)` for reproducible runs. In algorithm code, use `self.rng` (decorator API) or `self.generator` (classic API), not global `np.random`.

3. **Check Bounds Repairs**
   Keep agent solutions valid inside problem bounds. Use `self.bounds.correct(solution)` in Python, or `cy.correct_solution(self.problem, solution)` in Cython.

4. **Environment Variables**
   - `CLYPTO_CACHE_DIR` — Path for storing JIT-compiled binaries (default: `~/.cache/clypto`).
   - `CLYPTO_OPENMP=0` — Disable OpenMP compiler and linker flags if OpenMP is not available on your platform.
