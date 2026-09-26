#!/usr/bin/env python
"""Time every optimizer on the legacy and vectorize collections (and optionally mealpy-lts).

    python benchmarks/bench_engines.py --out bench.json
    python benchmarks/bench_engines.py --only "PSO|GWO" --epoch 100 --mealpy-python /path/to/venv/bin/python

Each class runs ``--repeats`` times per engine on the same problem and seed; the best time is kept.
The vectorize collection also runs with a batch (``vectorized=True``) objective. The last lines
summarize the speedup (legacy time / vectorize time) overall and per ``--bins`` bucket.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import clypto as cy

MEALPY_SNIPPET = """
import json, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import importlib, inspect, pkgutil
import mealpy
from mealpy import NumberBounds, Problem
from mealpy.optimizer import Optimizer
cfg = json.loads(sys.argv[1])
def sphere(x):
    return float(np.sum(x ** 2))
out = {}
opts = {}
for cat in pkgutil.iter_modules(mealpy.__path__):
    if not cat.name.endswith("_based"):
        continue
    pkg = importlib.import_module(f"mealpy.{cat.name}")
    for mod in pkgutil.iter_modules(pkg.__path__):
        try:
            m = importlib.import_module(f"mealpy.{cat.name}.{mod.name}")
        except Exception:
            continue
        for cname, cobj in inspect.getmembers(m, inspect.isclass):
            if issubclass(cobj, Optimizer) and cobj is not Optimizer:
                opts[cname] = cobj
for name in cfg["names"]:
    cls = opts.get(name)
    if cls is None:
        continue
    best = None
    try:
        for _ in range(cfg["repeats"]):
            prob = Problem(obj_func=sphere, bounds=NumberBounds(float, low=[-5.0] * cfg["dim"], up=[5.0] * cfg["dim"]), sense="min", log_to=None)
            t = time.perf_counter()
            cls(epoch=cfg["epoch"], pop_size=cfg["pop"]).solve(prob, seed=cfg["seed"])
            dt = time.perf_counter() - t
            best = dt if best is None else min(best, dt)
    except Exception:
        best = None  # not runnable with these parameters in mealpy-lts
    if best is not None:
        out[name] = best
print(json.dumps(out))
"""


def sphere(x):
    return float(np.sum(x**2))


def sphere_batch(X):
    return np.sum(X**2, axis=1)


def timed(cls, problem, args):
    best = None
    for _ in range(args.repeats):
        t = time.perf_counter()
        cls(epoch=args.epoch, pop_size=args.pop).solve(problem, seed=args.seed)
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best


def optimizers(engine):
    try:
        return cy.get_all_optimizers(engine=engine)
    except TypeError:  # older API without engines: single collection
        return cy.get_all_optimizers()


def run_mealpy(python, names, args):
    cfg = {"names": names, "repeats": args.repeats, "dim": args.dim, "epoch": args.epoch, "pop": args.pop, "seed": args.seed}
    proc = subprocess.run([python, "-c", MEALPY_SNIPPET, json.dumps(cfg)], capture_output=True, text=True, check=False)
    if proc.returncode:
        print("mealpy run failed:", proc.stderr[-400:], file=sys.stderr)
        return {}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def bucket(speedup, edges):
    for lo, hi in zip(edges, edges[1:] + [float("inf")], strict=True):
        if lo <= speedup < hi:
            return f"{lo:g}x" if hi == float("inf") else f"{lo:g}-{hi:g}x"
    return "<%gx" % edges[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="", help="regex on the class name")
    ap.add_argument("--legacy", default="legacy", help="engine name of the classic collection")
    ap.add_argument("--vectorize", default="vectorize", help="engine name of the vectorized collection")
    ap.add_argument("--mealpy-python", default="", help="python of an env with mealpy-lts installed")
    ap.add_argument("--bins", default="0.9,1.5,3,10", help="speedup bucket edges")
    ap.add_argument("--from-json", default="", help="reuse the rows of a previous run and only add the mealpy timings")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    legacy, vec = optimizers(args.legacy), optimizers(args.vectorize)
    names = sorted(n for n in vec if n in legacy and re.search(args.only, n))
    scalar = cy.Problem(obj_func=sphere, bounds=cy.NumberBounds(float, low=[-5.0] * args.dim, up=[5.0] * args.dim), sense="min")
    batch = cy.Problem(obj_func=sphere_batch, bounds=cy.NumberBounds(float, low=[-5.0] * args.dim, up=[5.0] * args.dim), sense="min", vectorized=True)

    rows = json.loads(Path(args.from_json).read_text())["rows"] if args.from_json else {}
    for name in [] if args.from_json else names:
        try:
            row = {"legacy": timed(legacy[name], scalar, args), "vectorize": timed(vec[name], scalar, args)}
        except Exception as exc:  # a class that fails in one engine is reported, not fatal
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"[:120]}
            continue
        try:
            row["vectorize_batch"] = timed(vec[name], batch, args)
        except Exception:
            row["vectorize_batch"] = None
        rows[name] = row
        print(f"{name:26s} legacy {row['legacy'] * 1e3:8.1f} ms  vectorize {row['vectorize'] * 1e3:8.1f} ms  x{row['legacy'] / row['vectorize']:.2f}", flush=True)

    if args.mealpy_python:
        mp = run_mealpy(args.mealpy_python, names, args)
        for name, sec in mp.items():
            if name in rows and "error" not in rows[name]:
                rows[name]["mealpy"] = sec

    good = {n: r for n, r in rows.items() if "error" not in r}
    edges = [float(x) for x in args.bins.split(",")]
    speed = {n: r["legacy"] / r["vectorize"] for n, r in good.items()}
    hist = {}
    for n, s in speed.items():
        hist.setdefault(bucket(s, edges), []).append(n)
    print(f"\n{len(good)} classes, median speedup x{statistics.median(speed.values()):.2f} "
          f"(min x{min(speed.values()):.2f}, max x{max(speed.values()):.2f}); slower than legacy: {sum(s < 1 for s in speed.values())}")
    for key in sorted(hist, key=lambda k: float(re.findall(r"[\d.]+", k)[0])):
        print(f"  {key:>10s}: {len(hist[key])}")
    slower = sorted((s, n) for n, s in speed.items() if s < 1)
    if slower:
        print("slower than legacy:", ", ".join(f"{n} x{s:.2f}" for s, n in slower))
    if any("mealpy" in r for r in good.values()):
        ms = [r["mealpy"] / r["vectorize"] for r in good.values() if "mealpy" in r]
        print(f"vs mealpy-lts ({len(ms)} classes): median x{statistics.median(ms):.2f}")
    errors = {n: r["error"] for n, r in rows.items() if "error" in r}
    if errors:
        print("errors:", errors)
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
