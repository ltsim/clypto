# Memory audit — baseline (clypto, HEAD 7654d659 → 67c51eca)

**Supervisor:** `cy-audit-advisor-code`. **Auditor session:** `opencode_2eddfb44` (GLM-5.3-flash).
**Plan:** `reports/AUDIT-MEMORY-LEAKS-PLAN.md`. **Probe:** `benchmarks/memleak_probe.py` (untracked).
**This file supersedes the earlier consolidated draft** (backup: `/tmp/opencode/reports/baseline-v1-consolidated.md.bak`).

## Verdict

No leak in the compiled C paths. RSS is flat over 300 iterations for every workload; Valgrind
attributes zero `definitely lost` blocks to clypto frames; ASan/LSan finds no use-after-free,
overflow or double-free and its exit-time leaks are all interpreter/scipy-side. The audit did
find:

1. **Memory-safety bugs** in `Population` (unchecked `<Agent>` casts on duck-typed agents) —
   fixed and committed as `67c51eca`.
2. **A bounded, caller-controlled retention in the decorator API** (not the native engine):
   `LegacyOptimizer` writes `g_best` only as `snapshot_agent(...)`, so the native `solve()`
   returns an independent copy that pins nothing — the `PSOAgents.owner` cycle is pure garbage
   and the cycle collector breaks it. The decorator engine snapshots through
   `RuntimeAgent.copy()`, which carries `_evaluator` (a bound method of the optimizer): a caller
   that keeps that agent pins one optimizer + one population and costs +3 refs on `problem`
   until the agent is dropped. Verified by the Phase 0 reproduction below (2026-11-17).

## Results

### RSS slope (native build, gc.collect per iteration; PASS threshold < 1 MB / 100 iters)

| Workload | Iters | Slope kB/iter | MB / 100 iters | Log |
|---|---|---|---|---|
| PSO pop200 d30 e60 | 300 | +0.054 | **+0.005** | `/tmp/opencode/reports/rss_pso_300.log` |
| GA pop200 d30 e60 | 300 | +0.020 | **+0.002** | `rss_ga_300.log` |
| DE pop200 d30 e60 | 300 | +0.019 | **+0.002** | `rss_de_300.log` |
| compile path (JIT + solve) | 200 | +0.006 | **+0.001** | `rss_compile_200.log` |
| decorator path (`@cy.optimizer(compile=True)`) | 200 | +0.030 | **+0.003** | `rss_decorated_200.log` |
| churn 5000×Population(20)/iter | 50 | +0.012 | **+0.001** | `rss_churn_50.log` |

All plateau; HWM ≈ base RSS (~118–120 MB). Under Valgrind (no pymalloc) the PSO probe shows
+0.94 MB/100 iters from arena growth noise (`vg_pso_stdout.log`), still under threshold ×1.

### tracemalloc (snapshot iter10 vs final, top diffs filtered to clypto SOURCE frames)

- PSO 40 iters and churn 30 iters: **no trace passing through clypto sources in the diff**;
  total net diff all frames ≈ +5 KiB (`tm_pso_40c.log`, `tm_churn_30c.log`).
- Caveat documented in the probe header: allocations made inside cdef code appear as
  `[None]` frames and are invisible to tracemalloc; it proves *Python-object* retention only.

### Refcounts / weakrefs (`rc_pso_head_final.log`, `rc_churn_8c.log` on 67c51eca)

- Stable objects across repeated solves: `delta=[0,0,0,0,0]` — problem rc does NOT grow when
  both `model` and the returned `best` are deleted each iteration (probe part-b shape).
- Weakref death: Tracker dies after `del`; `Population`/`Agent` are cdef classes without a
  dict → not weak-referenceable by design; equivalent death-by-counter used instead:
  residual `Population=0`, `Agent=0` after churn of 500 batches (earlier `Agent residual 4`
  was the probe's own last-loop local `p` pinning one population — fixed in the probe).
- Retention reproduction (the "+3/solve" row): with `res = model.solve(...)` rebound-but-not
  deleted and `gc.collect()` per iteration, OLD code showed rc 5→8→…→26 with 8 live
  `OriginalPSO` objects accumulating mid-loop; deleting the last `res` collapses everything
  back to base rc=2. Per-generation cost measured directly: +3 refs on `problem`
  (population, factory, generator-holder chain), +1 Population, +~pop_size Agents pinned.
- Same pattern on NEW code (67c51eca): identical numbers — the fix changed cast safety, not
  ownership. GA/DE control (default AgentFactory, no owner field): flat at all times.

### Valgrind Memcheck (PYTHONMALLOC=malloc, definite+indirect, suppressions active)

| Run | definitely lost | classification |
|---|---|---|
| pso 50 iters (`vg_pso.log`) | 40,705 B / 168 blocks | **0 frames of clypto.** All stacks terminate in `xsf::numpy::ufunc` / `PyUFunc_FromFuncAndData*` called from `_special_ufuncs_module_exec` / `_gufuncs_module_exec` — i.e. scipy.special import-time ufunc registration leaking libc malloc. Identical totals in the compile run (`vg_compile.log`) → constant one-time cost, not per-iteration. |
| ga 50 iters (`vg_ga.log`) | 0 B | clean |
| de 50 iters (`vg_de.log`) | 40 B / 1 block | `range_iter.warm` ← `SeedSequence.generate_state` ← `np.random.default_rng` triggered from `BaseBounds.seed` setter — CPython/numpy one-off; size independent of iteration count → not a clypto leak |
| churn 50 iters (`vg_churn.log`) | 0 B | clean |
| compile 50 iters (`vg_compile.log`) | 40,705 B / 168 blocks | same scipy.signature as pso run (identical bytes) |

Suppression file: `reports/suppressions/python-numpy.supp` (CPython arenas/codecache/finalize,
ld.so TLS, numpy alloc-cache entries; format note inside). The 168-block scipy bucket was NOT
suppressed because it never touches clypto frames and is constant-size; filing it upstream or
adding `fun:xsf::numpy::ufunc` entries is optional follow-up.

### ASan/LSan (instrumented copy ONLY: `/tmp/opencode/asan-build`, worktree untouched)

- Build: 269 extensions rebuilt with `-fsanitize=address -fno-omit-frame-pointer -g`
  (`asan_build.log`); plain rebuild also works there (`plain_rebuild.log`).
- Runs: pso (`asan_pso.log`, exit 1 due to halt_on_error on leaks), churn, compile —
  **0 errors** of type heap-use-after-free / overflow / double-free / stack-use-after-return.
- LSan exit totals ≈ 1.5 MB / ~4,100 allocations, essentially IDENTICAL between workloads
  (interpreter + pybind11/highspy + numpy module-init retained; first non-asan frames:
  `_PyObject_Malloc`, `pybind11::cpp_function::make_function_record`, `strdup` in highspy).
  **Zero frames from clypto sources.** No per-iteration delta signal.

## Findings (prioritized list of REAL items to fix / decide)

1. **[FIXED in 67c51eca] Unchecked `<Agent>` casts** in `Population.evaluate/snapshot/fitness/
   solutions` segfaulted on duck-typed agents mixed into the public `agents` list. Reproduced
   (exit 139), covered by regression tests. Numeric parity kept (golden sha match against
   7654d659, 434 tests green).
2. **[CLOSED 2026-11-17 — misdiagnosed in the first pass] generation retention**:
   the original claim (`PSOAgents.owner` strong ref + `solve()` returning a population agent
   pins a generation) is **wrong for the native engine**: `LegacyOptimizer.after_initialization`
   /`after_evolve` write `g_best` as `snapshot_agent(ranked[0])`, a copy with no edge back to
   the population, so an algorithm that assigns a live member inside `evolve` is re-snapshotted
   before `solve()` returns. Phase 0 measured: `best is population[i]` **False** and no shared
   solution memory for `OriginalPSO`, `P_PSO`, `OriginalGA`; holding the return pins nothing
   (rc(problem) flat, colonies collected). The real `+3/solve` signature belongs to the
   **decorator API**: `RuntimeAgent.copy()` carries `_evaluator = self.evaluate_agent`, a bound
   method of the optimizer, so keeping that agent pins the optimizer + population (+3 refs on
   `problem`) until dropped — bounded, one generation per kept agent. Decision: **document**
   (option c), with corrected docstrings; a weakref here would change re-evaluation semantics.
   Repro: `/tmp/opencode/fixmem/PHASE0-REPORT.md`, scripts `phase0_*.py`.
3. **[NO ACTION] scipy.special ufunc-registration definite-loss** (40.7 KB/run, constant):
   upstream issue candidate, not ours; suppressed-adjacent but attributed.
4. **[NO ACTION] Bounds seed-setter default_rng one-off** (40 B): CPython/numpy internal.

## False positives closed this session

- "problem +3/solve monotonic growth": measurement-shape artifact — appears only when the
  returned agent is kept rooted (finding 2); disappears with `del best` + any collect.
- "Agent residual 4 / Population residual 1 after churn": the probe's own loop locals;
  corrected in `memleak_probe.py` (drop `p`, explicit `del agent0/fit/sols`, fresh-frame scans).
- tracemalloc "+5 KiB at clypto filter": filter matched site-packages paths containing
  `/clypto/`; fixed to require repo source prefix.

## What was not run

- Valgrind on decorated workload (time budget; compile-run covers the JIT/module-exec surface).
- A/B cost table of the 67c51eca guards on an idle host (contaminated by concurrent valgrinds;
  prior bound ≤ 0.15 µs/call stands as indicative).

## Evidence index

- Probe & decorated-workload modules: `benchmarks/memleak_probe.py`, `benchmarks/memleak_decorated_mod.py` (both untracked, not committed).
- Logs: `/tmp/opencode/reports/{rss_*,tm_*,rc_*,vg_*,asan_*}.log`; frozen pre-fix tree for A/B: `/tmp/opencode/asan-build` (now rebuilt WITHOUT asan).
- Suppressions: `reports/suppressions/python-numpy.supp` (syntax validated with valgrind-3.27.1).
