# Cython Memory-Leak Audit — Plan (clypto)

**Status:** planned. Starts when the Phase 4 (hot-path profiling) review is closed.
**Owner:** `cy-audit-advisor-code` supervises; GLM programmer subagent executes; `cy-implementation` handles env/infra.
**Branch:** `refactoring/audit-simplifications`.

## Why this audit, now

Phase 2 (`population.pyx`) introduced raw pointer access (`<double*>cnp.PyArray_DATA`) and
`memcpy` on NumPy buffers; Phase 3 removed precompile's `pyximport` runtime. Those are exactly
the places where Cython leaks or dangles references. The current evidence (420 tests, ruff,
mypy) proves *behaviour*, not *memory safety*.

## Scope

- `clypto/agents/_core.py` + `_core.pyi` (cdef classes, `__dealloc__`, public attrs)
- `clypto/optimizer/native/{population,utils,nogil}.pyx`
- `clypto/optimizer/precompile/*.pyx` (compiler, decorator, base, declaration, decoration, problem)
- hot algorithms: PSO, GA, DE (plus any file touched by later phases)
- `clypto/utils/target.py`

Out of scope: untouched optimizers (smoke test only).

## Leak classes to hunt (checklist)

1. **Refcounts:** `PyObject*`/`object` field overwritten without DECREF; borrowed ref stored as
   owned; `object` used inside `nogil`; refs leaked on exception paths (early `return` before
   `Py_DECREF`).
2. **C heap:** `malloc`/`strdup`/`new` without matching free in `__dealloc__`; `realloc` result
   unchecked; buffers grown per epoch.
3. **NumPy ownership:** `PyArray_DATA` pointer retained after the array is released (dangling);
   `PyArray_SimpleNew` result never DECREF'd; `memcpy` size/stride mismatch; writes to
   non-contiguous arrays.
4. **Lifetime:** `Population` holding pointers into arrays freed by the caller; `Agent` arrays
   outliving their `Population`; views into freed buffers.
5. **Python retention:** module/class-level caches, `History` growth, decorator closures,
   per-epoch list accumulation.

## Tooling

Python level (fast; gate for every change):

```bash
.venv/bin/python benchmarks/memleak_probe.py rss --algo PSO --iters 300
.venv/bin/python benchmarks/memleak_probe.py tracemalloc --algo GA --iters 200
```

- RSS from `/proc/self/status` after `gc.collect()`; PASS only if the linear slope over the last
  200 iterations is ~0 (< 1 MB / 100 iterations).
- `tracemalloc` snapshot diff, top-20 filtered to `clypto/` frames.
- Refcount probes: `sys.getrefcount` before/after N epochs on `pop`, `pop.agents[0]`,
  `pop.fitness`; `weakref.ref` death checks for `Population`/`Agent`/`History`.

C level (slower; on demand):

```bash
# Memcheck: short runs only (20-50x slowdown)
PYTHONMALLOC=malloc valgrind --leak-check=full --show-leak-kinds=definite \
  --errors-for-leak-kinds=definite --num-callers=30 \
  --suppressions=reports/suppressions/python-numpy.supp \
  --log-file=/tmp/opencode/valgrind.%p.log \
  .venv/bin/python benchmarks/memleak_probe.py rss --algo PSO --iters 50

# ASan/LSan: rebuild extensions instrumented
CFLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  .venv/bin/python setup.py build_ext --inplace -j 8
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
  ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1:halt_on_error=1 \
  .venv/bin/python benchmarks/memleak_probe.py rss --algo PSO --iters 50
```

Reference skills available locally: `valgrind`, `address-sanitizer`.

**Prerequisite (missing today):** `valgrind` is not installed on this host (openSUSE Leap 16).
Needs `sudo zypper install -y valgrind` before the Memcheck step; `libasan` is already present at
`/usr/lib64/gcc/x86_64-suse-linux/15/libasan.so`, so the ASan path works without extra packages.
If Valgrind cannot be installed, the audit runs on ASan/LSan + RSS/tracemalloc and says so
explicitly in the report.

## Workloads

- `OriginalPSO` pop 200 / d30 / 300 epochs, plain Python objective (Phase 2 path).
- `GA` and `DE`, same shape.
- `cy.compile(...)` decorator path, 200 iterations (exercises compile + teardown).
- Batch/swarm mode with a vectorized objective (hits the batch buffers).
- Population churn: create and destroy 5,000 `Population(20)` objects.

## Acceptance criteria

- RSS plateau over 300 iterations; no monotonic growth.
- Valgrind: 0 `definitely lost` blocks attributable to clypto frames; every remaining block is
  classified as CPython/NumPy noise with a suppression entry.
- ASan: 0 errors (no use-after-free, overflow or double-free); LSan delta between iteration 10
  and 300 ≈ 0.
- Refcounts stable across epochs; weakrefs die after `del`.
- Fixes keep 420 tests + ruff + mypy green and numerics bit-identical.

## Deliverables

- `reports/memory-audit-baseline.md` — raw numbers, log paths, top leaks, false positives.
- `reports/suppressions/{python-numpy.supp,lsan.supp}` — documented suppressions.
- `benchmarks/memleak_probe.py` — reproducer (not committed until reviewed).
- One commit per leak fixed, each with before/after evidence.

## Risks and cost

- Valgrind on CPython is noisy: expect exit-time allocations from CPython/NumPy. Compare
  *growth across iterations*, never exit-time totals.
- ASan plus a non-instrumented interpreter can be unstable; fallback is Valgrind + 
  `PYTHONMALLOC=debug`.
- Budget: ~1 session for the baseline, ~1 per leak family fixed.

## Sequence

1. Baseline harness + measurements (no fixes).
2. Triage: real leaks vs false positives.
3. Fixes, one family at a time, each with evidence.
4. Re-run baseline; update `reports/memory-audit-baseline.md`.
