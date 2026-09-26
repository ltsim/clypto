#!/usr/bin/env python
"""Compare PSO speed across clypto checkouts (each built in place).

    python benchmarks/bench_pso.py --repo original=/path/a --repo current=.

Every checkout runs in its own interpreter (``PYTHONPATH=<checkout>``). Each
scenario reports the best of ``--reps`` runs and whether every checkout found
the same fitness (the algorithms must stay bit-identical).

Scenarios a checkout cannot run (a vectorized objective or a nogil evaluator on
a version without them) fall back to the plain Python objective and are marked
with the objective actually used.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# (label, algorithm, pop_size, n_dims, epochs, objective)
#   objective: "python" (per-row sphere), "vectorized" (batch sphere),
#              "heavy" (expensive Rastrigin; nogil + OpenMP when available)
SCENARIOS = [
    ("n50 d30", "OriginalPSO", 50, 30, 300, "python"),
    ("n50 d30", "AIW_PSO", 50, 30, 300, "python"),
    ("n50 d30", "C_PSO", 50, 30, 300, "python"),
    ("n50 d30", "CL_PSO", 50, 30, 100, "python"),
    ("n500 d100", "OriginalPSO", 500, 100, 60, "python"),
    ("n500 d100", "OriginalPSO", 500, 100, 60, "vectorized"),
    ("n200 d30 heavy", "OriginalPSO", 200, 30, 20, "heavy"),
]
HEAVY_REPEATS = 200

_EVALUATOR_PYX = """
from libc.math cimport cos, M_PI
from clypto.optimizer.native.nogil cimport _NogilEvaluator

cdef void heavy(const double* x, Py_ssize_t n, double* out) noexcept nogil:
    cdef Py_ssize_t i, r
    cdef double s = 0.0
    for r in range(%d):
        for i in range(n):
            s += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]) + 10.0
    out[0] = s / %d

cdef class HeavyEvaluator(_NogilEvaluator):
    def __cinit__(self):
        self._func = heavy
        self._n_objs = 1
""" % (HEAVY_REPEATS, HEAVY_REPEATS)


def _heavy_python(x):
    import numpy as np

    s = 0.0
    for _ in range(HEAVY_REPEATS):
        s += float(np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))
    return s / HEAVY_REPEATS


def _evaluator():
    import pyximport

    tmp = Path(tempfile.mkdtemp(prefix="clypto-bench-"))
    (tmp / "_clypto_bench_heavy.pyx").write_text(_EVALUATOR_PYX)
    pyximport.install(language_level=3)
    sys.path.insert(0, str(tmp))
    import _clypto_bench_heavy

    return _clypto_bench_heavy.HeavyEvaluator()


def worker(reps):
    import numpy as np

    import clypto as cy
    from clypto.native.collection.vectorize.swarm_based import PSO

    # Vectorized objectives and the OpenMP evaluator in PSO arrived together.
    new_api = hasattr(cy.Problem(bounds=cy.NumberBounds(float, low=[0.0], up=[1.0])), "vectorized")
    results = []
    for label, algo, n, d, epochs, objective in SCENARIOS:
        bounds = cy.NumberBounds(float, low=[-5.12] * d, up=[5.12] * d)
        kwargs, used, mode = {}, objective, None
        if objective == "vectorized" and new_api:
            kwargs = {"obj_func": lambda X: np.sum(X**2, axis=1), "vectorized": True}
        elif objective == "heavy":
            kwargs = {"obj_func": _heavy_python}
            if new_api:
                kwargs["evaluator"] = _evaluator()
                mode, used = "parallel", "heavy nogil"
            else:
                used = "heavy python"
        else:
            kwargs = {"obj_func": lambda x: float(np.sum(x**2))}
            used = "python"
        best, fitness = float("inf"), None
        for _ in range(reps):
            problem = cy.Problem(bounds=bounds, sense="min", **kwargs)
            model = getattr(PSO, algo)(epoch=epochs, pop_size=n, mode=mode)
            start = time.perf_counter()
            g_best = model.solve(problem, seed=7)
            best = min(best, time.perf_counter() - start)
            fitness = float(g_best.target.fitness)
        results.append({"label": label, "algo": algo, "objective": objective, "used": used,
                            "seconds": best, "fitness": fitness})
    json.dump(results, sys.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", default=None, help="OMP_NUM_THREADS for every run")
    args = parser.parse_args()

    runs = {}
    for spec in args.repo:
        name, path = spec.split("=", 1)
        env = {**os.environ, "PYTHONPATH": str(Path(path).resolve())}
        if args.threads:
            env["OMP_NUM_THREADS"] = args.threads
        out = subprocess.run(
            [sys.executable, __file__, "--worker", str(args.reps)],
            env=env, capture_output=True, text=True, check=True,
        )
        runs[name] = json.loads(out.stdout)

    names = list(runs)
    base = names[0]
    header = f"{'scenario':16s} {'algorithm':12s} {'objective':11s}" + "".join(
        f" {n:>12s}" for n in names
    ) + "".join(f" {'x vs ' + base:>12s}" for n in names[1:]) + "  same fitness"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(runs[base]):
        times = [runs[n][i]["seconds"] for n in names]
        fits = {runs[n][i]["fitness"] for n in names if runs[n][i]["used"] == runs[base][i]["used"]}
        line = f"{row['label']:16s} {row['algo']:12s} {row['objective']:11s}"
        line += "".join(f" {t * 1e3:10.1f}ms" for t in times)
        line += "".join(f" {times[0] / t:11.2f}x" for t in times[1:])
        used = {runs[n][i]["used"] for n in names}
        line += "  " + ("yes" if len(fits) == 1 else "NO") + (
            "" if len(used) == 1 else f"  (objective used: {', '.join(runs[n][i]['used'] for n in names)})"
        )
        print(line)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
