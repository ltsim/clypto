#!/usr/bin/env python
# Created by "LTSIM" at 2026-11-17 ----------%
"""Memory-leak probe for the Cython hot paths (memory audit, 2026-11).

Modes over the same workloads:

* ``rss``         -- N iterations of the workload; RSS from /proc/self/status is
                     sampled after ``gc.collect()`` each iteration; least-squares
                     slope over the last ``--regress`` samples (MB / 100 iters).
* ``tracemalloc`` -- snapshot at iteration 10 and at the end; top diff frames
                     whose traceback passes through a clypto SOURCE frame.
                     NOTE: compiled Cython allocates report Python-level frames
                     only (numpy helpers, dict/list machinery); allocations made
                     inside cdef code appear as ``[None]`` and are invisible here.
                     tracemalloc therefore proves Python-object retention, not C
                     refcounts.
* ``refcount``    -- (a) growth of gc.get_objects()/total dict count across churn
                     iterations (catches module/dict-level retention that object
                     identity-based probes miss); (b) ``sys.getrefcount`` of stable
                     objects (problem, bounds, objective, classes) across repeated
                     solves; (c) weakref death checks for Population/Agent/Tracker.
* ``timed``       -- one solve with step timings and post-solve retention marks
                     (population length, tracker buffer) — epoch-retention probe.

Workloads (``--algo``):
    pso / ga / de   -- fresh model + ``solve`` per iteration (pop 200, d 30).
    compile         -- ``cy.compile(...)`` JIT + short solve per iteration
                       (stable digest: first build compiles, rest hit the cache).
    decorated       -- @cy.optimizer/@cy.agent(compile=True) class + short solve.
    churn           -- create/bind/fill/destroy 5000 ``Population(20)`` per iteration.

Examples:
    .venv/bin/python benchmarks/memleak_probe.py rss --algo pso --iters 300
    .venv/bin/python benchmarks/memleak_probe.py tracemalloc --algo ga --iters 40
    .venv/bin/python benchmarks/memleak_probe.py refcount --algo churn --iters 10
"""
import argparse
import gc
import importlib
import os
import sys
import time
import tracemalloc

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # repo root
import clypto as cy  # noqa: E402

# ``cy.optimizer`` is the decorator function (it shadows the subpackage attribute),
# so the native classes are reached through importlib.
_population_module = importlib.import_module("clypto.optimizer.native.population")
_agent_module = importlib.import_module("clypto.optimizer.native.agent")
Population = _population_module.Population
Agent = _agent_module.Agent
from clypto.optimizer.history import Tracker  # noqa: E402
from clypto.native.collection.swarm_based import PSO  # noqa: E402
from clypto.native.collection.evolutionary_based import GA, DE  # noqa: E402

DIMS = 30
CYTHON_SOURCE = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class NativeContractionSearch(cy.Optimizer):
    cdef public double factor

    def __init__(self, epoch=40, pop_size=20, factor=0.8, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")

    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * self.factor
            agent.evaluate(self.problem)
"""


def _sphere(x):
    return float(np.sum(x**2))


def make_problem():
    bounds = cy.NumberBounds(float, low=[-10.0] * DIMS, up=[10.0] * DIMS)
    return cy.Problem(bounds=bounds, sense="min", obj_func=_sphere)


def make_optimizer(algo, epochs, pop_size):
    module = {"pso": PSO, "ga": GA, "de": DE}[algo]
    cls = {"pso": "OriginalPSO", "ga": "OriginalGA", "de": "OriginalDE"}[algo]
    return getattr(module, cls)(epoch=epochs, pop_size=pop_size)


def solve_workload(algo, epochs, pop_size, problem):
    """One full solve: builds the model, runs the optimizer, drops everything."""
    model = make_optimizer(algo, epochs, pop_size)
    best = model.solve(problem, seed=7)
    fitness = float(best.fitness)
    del model, best
    return fitness


def timed_solve_workload(algo, epochs, pop_size, problem):
    """A solve with explicit step timings and post-solve retention marks."""
    marks = {}
    t0 = time.perf_counter()
    model = make_optimizer(algo, epochs, pop_size)
    marks["t_construct"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    best = model.solve(problem, seed=7)
    marks["t_solve"] = time.perf_counter() - t1
    marks["pop_len_after"] = len(model.population.agents)
    marks["tracker_n_epochs"] = getattr(model.tracker, "n_epochs", None)
    buf = getattr(model.tracker, "buffer", None)
    marks["tracker_buffer_mb"] = round(buf.nbytes / 1e6, 3) if buf is not None else None
    t2 = time.perf_counter()
    del model, best
    gc.collect()
    marks["t_del_gc"] = time.perf_counter() - t2
    for k, v in marks.items():
        print(f"  {k}={v}")


def compile_workload(problem):
    """JIT compile (cached after the first call) + instantiate + short solve."""
    cls = cy.compile(CYTHON_SOURCE)
    model = cls(epoch=3, pop_size=20)
    model.solve(problem, seed=7)
    del model, cls


def churn_workload(problem):
    """Create/bind/fill/destroy populations: agent cdef-class churn."""
    for _ in range(5000):
        pop = Population(20)
        pop.bind(problem, np.random.default_rng(7))
        pop.fill()
        del pop


# --- decorator path (@cy.optimizer / @cy.agent), compiled variant -------------
# The class lives in benchmarks/memleak_decorated_mod.py: ``inspect.getsource``
# needs a real file, and the stable JIT digest means each iteration hits the
# compile cache and exercises only the decorated-class + solve + teardown path.
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)


def decorated_workload(problem):
    """Decorated class (re)built + short solve, per iteration."""
    mod = importlib.import_module("memleak_decorated_mod")
    model = mod.DecaySearch(epoch=3, pop_size=20)
    model.solve(problem, seed=7)
    del model


WORKLOADS = {
    "pso": lambda p, cfg: solve_workload("pso", cfg.epochs, cfg.pop_size, p),
    "ga": lambda p, cfg: solve_workload("ga", cfg.epochs, cfg.pop_size, p),
    "de": lambda p, cfg: solve_workload("de", cfg.epochs, cfg.pop_size, p),
    "compile": lambda p, cfg: compile_workload(p),
    "decorated": lambda p, cfg: decorated_workload(p),
    "churn": lambda p, cfg: churn_workload(p),
}


# --------------------------------------------------------------------------- rss

def rss_kb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])  # kB
    return 0


def hwm_kb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def _linfit_slope(xs, ys):
    """Slope of the least-squares line y = a + b x (manual: numpy.linalg.lstsq can
    raise LinAlgError on short, near-constant windows)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    return sxy / sxx if sxx else float("nan")


def run_rss(cfg):
    workload = WORKLOADS[cfg.algo]
    problem = make_problem()
    samples = []
    for i in range(cfg.iters):
        t_iter0 = time.perf_counter()
        workload(problem, cfg)
        del problem
        gc.collect()
        problem = make_problem()  # noqa: F841 - read by the next iteration's workload()
        if cfg.verbose:
            print(f"  iter {i + 1}/{cfg.iters} took {time.perf_counter() - t_iter0:.2f}s rss={rss_kb()} kB", flush=True)
        if i >= cfg.warmup:  # exclude warm-up from the sample window
            samples.append(rss_kb())
    gc.collect()

    n = len(samples)
    window = samples[-cfg.regress:]
    slope_kb_iter = _linfit_slope(list(range(len(window))), [float(v) for v in window])
    print(f"[rss] algo={cfg.algo} iters={cfg.iters} warmup={cfg.warmup} samples={n}")
    print(f"[rss] window_last={min(n, cfg.regress)} slope={slope_kb_iter:+.3f} kB/iter "
          f"({slope_kb_iter * 100.0 / 1024.0:+.3f} MB per 100 iters)")
    if n >= 2:
        print(f"[rss] rss_first_sample={samples[0]} kB rss_last={samples[-1]} kB "
              f"growth_over_window={(samples[-1] - samples[0]) / 1024.0:+.2f} MB")
    else:
        print(f"[rss] samples too few for a window (n={n})")
    print(f"[rss] hwm={hwm_kb()} kB ({hwm_kb() / 1024.0:.1f} MB)")
    if n >= 10:
        # Coarse trend view: mean RSS over each of 10 equal slices of the window.
        edges = np.linspace(0, n, 11).astype(int)
        means = [round(float(np.mean(samples[a:b])), 1) for a, b in zip(edges[:-1], edges[1:], strict=False) if b > a]
        print(f"[rss] slice means (kB, 10 slices): {means}")


# --------------------------------------------------------------------- tracemalloc

def _is_clypto_source(fr):
    name = fr.filename
    # only the repo's clypto SOURCE tree counts: not site-packages paths that
    # contain "clypto" (the worktree is literally .../clypto/<branch>/...).
    return name.startswith(os.path.join(REPO_ROOT, "clypto") + os.sep)


def run_tracemalloc(cfg):
    workload = WORKLOADS[cfg.algo]
    problem = make_problem()
    tracemalloc.start()
    for _ in range(10):
        workload(problem, cfg)
    gc.collect()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(10, cfg.iters):
        workload(problem, cfg)
        del problem
        gc.collect()
        problem = make_problem()  # noqa: F841 - read by the next iteration's workload()
    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diff = sorted(snap2.compare_to(snap1, "traceback"), key=lambda s: abs(s.size_diff), reverse=True)
    print(f"[tracemalloc] algo={cfg.algo} iters={cfg.iters} (snapshot1 at iter 10)")
    print("[tracemalloc] top-20 diff traces passing through a clypto source frame:")
    n_shown = 0
    for stat in diff:
        tb = stat.traceback
        if not any(_is_clypto_source(fr) for fr in tb):
            continue
        n_shown += 1
        anchor = next(fr for fr in tb if _is_clypto_source(fr))
        print(f"  {stat.size_diff / 1024.0:+10.2f} KiB  {stat.count_diff:+7d} objs  "
              f"{tb[0].filename}:{tb[0].lineno}  [clypto: {anchor.filename}:{anchor.lineno}]")
        for fr in tb[1:4]:
            print(f"                 <- {fr.filename}:{fr.lineno}")
        if n_shown >= 20:
            break
    if n_shown == 0:
        print("  (no trace passing through clypto/ sources in the diff)")
    total = sum(stat.size_diff for stat in diff)
    print(f"[tracemalloc] total net diff (all frames) = {total / 1024.0:+.2f} KiB")


# ---------------------------------------------------------------------- refcount

def refcount_table(objs, names):
    return {name: sys.getrefcount(obj) for name, obj in zip(names, objs, strict=False)}


def _obj_stats():
    """Count tracked objects and Python dicts (the per-Agent-slot dict leak shows
    up here even though the cdef objects themselves are invisible to gc)."""
    n_obj = n_dict = 0
    for o in gc.get_objects():
        n_obj += 1
        if type(o) is dict:
            n_dict += 1
    return n_obj, n_dict


def run_refcount(cfg):
    # (a) global object/dict growth under churn — catches retention by identity-miss.
    print(f"[refcount] gc-growth workload={cfg.algo} iters={cfg.iters}")
    workload = WORKLOADS[cfg.algo]
    problem = make_problem()
    gc.collect()
    base_obj, base_dict = _obj_stats()
    rows = []
    for i in range(cfg.iters):
        workload(problem, cfg)
        del problem
        gc.collect()
        problem = make_problem()
        rows.append(_obj_stats())
        if cfg.verbose:
            print(f"  iter {i + 1}: objects={rows[-1][0]} (+{rows[-1][0] - base_obj}) "
                  f"dicts={rows[-1][1]} (+{rows[-1][1] - base_dict})", flush=True)
    last_obj, last_dict = rows[-1]
    slope_obj = _linfit_slope(list(range(len(rows))), [float(r[0]) for r in rows])
    slope_dict = _linfit_slope(list(range(len(rows))), [float(r[1]) for r in rows])
    print(f"[refcount] gc.objects base={base_obj} last={last_obj} delta={last_obj - base_obj}"
          f" slope={slope_obj:+.1f}/iter")
    print(f"[refcount] dicts      base={base_dict} last={last_dict} delta={last_dict - base_dict}"
          f" slope={slope_dict:+.1f}/iter")
    del problem
    gc.collect()
    rest_obj, rest_dict = _obj_stats()
    print(f"[refcount] after final del+gc: objects={rest_obj} dicts={rest_dict} "
          f"(residual vs base: {rest_obj - base_obj}, {rest_dict - base_dict})")

    # (b) refcounts of stable objects across repeated solves.
    problem = make_problem()
    bounds = problem.bounds
    obj_func = problem.obj_func
    names = ["problem", "bounds", "obj_func", "AgentClass", "PopulationClass"]
    objs = [problem, bounds, obj_func, Agent, Population]
    print(f"[refcount] stable-object refcounts across solves (algo={cfg.algo})")
    print("  iter  " + "  ".join(f"{n:>16s}" for n in names))
    first = last = None
    n_rc_iters = min(cfg.iters, 20)
    for i in range(n_rc_iters):
        model = make_optimizer("pso", cfg.epochs, cfg.pop_size)
        model.solve(problem, seed=7)
        pop = model.population
        rc = refcount_table(objs, names)
        row = [rc[n] for n in names]
        if i == 0:
            first = row
        if i == n_rc_iters - 1:
            last = row
        print(f"  {i:4d}  " + "  ".join(f"{v:16d}" for v in row))
        if i in (0, n_rc_iters - 1):
            agent0 = pop.agents[0]
            fit = pop.fitness
            sols = pop.solutions
            print(f"    returned: pop@{id(pop):x} rc={sys.getrefcount(pop)} "
                  f"agent0@{id(agent0):x} rc={sys.getrefcount(agent0)} "
                  f"fitness_len={len(fit)} fitness_rc={sys.getrefcount(fit)} "
                  f"solutions_len={len(sols)} solutions_rc={sys.getrefcount(sols)}")
            print(f"    agent0.solution rc={sys.getrefcount(agent0.solution)} "
                  f"solution_id={id(agent0.solution):x} fitness={agent0.fitness!r}")
            # NOTE: agent0/fit/sols stay bound in this frame; they pin the last
            # iteration's population (the probe holds them, not clypto). The live
            # count that follows separates the debug refs from real residuals.
            gc.collect()
            n_debug = sum(1 for o in gc.get_objects() if type(o) is Agent)
            print(f"    live Agents with agent0/fit/sols pinned by this frame: {n_debug}")
        del model, pop
        gc.collect()

    # drop the debug locals so the residual counts below mean something
    try:
        del agent0, fit, sols
    except NameError:
        pass
    gc.collect()
    n_live_a = sum(1 for o in gc.get_objects() if type(o) is Agent)
    print(f"[refcount] after dropping locals: live Agents={n_live_a} (0 => no leak)")

    print(f"[refcount] stable objects: first={first} last={last} "
          f"delta={[b - a for a, b in zip(first, last, strict=False)]}")

    # (c) weakref death checks. Population/Agent are cdef classes without a dict:
    # not weak-referenceable by design — the equivalent death check is done with
    # an external counter around gc.collect() over a batch of freshly built objects.
    print("[refcount] weakref death checks:")
    import weakref
    t = Tracker()
    wr_t = weakref.ref(t)
    del t
    gc.collect()
    print(f"  Tracker alive after del? {wr_t() is not None}")
    pop = Population(3)
    ag = Agent(np.zeros(DIMS))
    try:
        wr_pop = weakref.ref(pop)
        wr_ag = weakref.ref(ag)
    except TypeError as exc:
        wr_pop = wr_ag = None
        print(f"  Population/Agent NOT weakref-able: {exc}")
    if wr_pop is not None:
        del pop, ag
        gc.collect()
        print(f"  Population alive after del? {wr_pop() is not None}")
        print(f"  Agent alive after del? {wr_ag() is not None}")

    def _type_count(tp):
        return sum(1 for o in gc.get_objects() if type(o) is tp)

    n0p, n0a = _type_count(Population), _type_count(Agent)
    objs = []
    p = None
    for _ in range(500):
        p = Population(4)
        p.bind(problem, np.random.default_rng(7))
        p.fill()
        objs.append(p)
    peak_p = _type_count(Population)
    del objs, p          # the loop's last local keeps one population alive otherwise
    gc.collect()
    rest_p, rest_a = _type_count(Population), _type_count(Agent)
    print(f"[refcount] cdef death-by-counter: Population base={n0p} peak={peak_p} "
          f"after del+gc={rest_p} residual={rest_p - n0p}; Agent base={n0a} after={rest_a} "
          f"residual={rest_a - n0a}")


# -------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["rss", "tracemalloc", "refcount", "timed"])
    p.add_argument("--algo", default="pso", choices=list(WORKLOADS))
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--epochs", type=int, default=60, help="epochs per solve (pso/ga/de)")
    p.add_argument("--pop-size", type=int, default=200)
    p.add_argument("--warmup", type=int, default=None,
                   help="rss: iterations excluded before sampling (default: 10, or iters//3 when iters < 30)")
    p.add_argument("--regress", type=int, default=200, help="rss: window for the slope regression")
    p.add_argument("--verbose", action="store_true")
    cfg = p.parse_args()

    np.seterr(all="ignore")
    if cfg.warmup is None:
        cfg.warmup = 10 if cfg.iters >= 30 else max(1, cfg.iters // 3)
    if cfg.mode == "timed":
        algo = cfg.algo if cfg.algo in ("pso", "ga", "de") else "pso"
        timed_solve_workload(algo, cfg.epochs, cfg.pop_size, make_problem())
    else:
        {"rss": run_rss, "tracemalloc": run_tracemalloc, "refcount": run_refcount}[cfg.mode](cfg)


if __name__ == "__main__":
    main()
