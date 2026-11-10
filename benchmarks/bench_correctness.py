#!/usr/bin/env python
"""Statistical fitness-correctness check of every optimizer against mealpy-lts.

    python benchmarks/bench_correctness.py --mealpy-python /path/to/venv/bin/python --out correctness.json
    python benchmarks/bench_correctness.py --mealpy-python ... --names-file swarm_based.json --out swarm.json

Bit-identical comparison against mealpy-lts is not meaningful: clypto and mealpy-lts run
independent RNG call sequences even under a matched seed (different call order, different
generator). Instead, each class runs ``--seeds`` independent seeds on a sphere and a Rastrigin
problem (same functions/bounds as ``tests/test_all_optimizer.py``) in both libraries, at the
epoch/pop budget the test suite already trusts as the safe floor (every optimizer accepts it),
and the two final-fitness distributions (mean/std) are compared within ``--tol`` pooled standard
deviations. Classes in ``NONDETERMINISTIC`` (defined below: they draw from the scipy global RNG,
which no ``seed`` can reach) get a wider tolerance instead of being flagged for high variance
alone. Classes in ``NO_EVOLVE`` (no real
evolve step, raise ``NotImplementedError``) are skipped.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

import clypto as cy
from tests._collection import NO_EVOLVE

NONDETERMINISTIC = {"AAO", "JADE", "L_SHADE", "OriginalLSHADEcnEpSin", "OriginalPSS", "OriginalSHADE"}


def sphere(x):
    return float(np.sum(x**2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


PROBLEMS = {
    "sphere": {"f": sphere, "d": 5, "low": -1.0, "up": 1.0},
    "rastrigin": {"f": rastrigin, "d": 10, "low": -5.0, "up": 5.0},
}

MEALPY_SNIPPET = """
import json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import importlib, inspect, pkgutil
import mealpy.optimizer as _mopt

class _FakeTqdm:
    def __init__(self, it, *a, **k):
        self._it = it
    def __iter__(self):
        return iter(self._it)
    def set_postfix(self, *a, **k):
        pass

_mopt.tqdm = _FakeTqdm  # mealpy's per-epoch tqdm bar would otherwise flood the subprocess pipe

import mealpy
from mealpy import FloatVar, Problem
from mealpy.optimizer import Optimizer
cfg = json.loads(sys.argv[1])

def sphere(x):
    return float(np.sum(x ** 2))

def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

FUNCS = {"sphere": sphere, "rastrigin": rastrigin}
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

out = {}
for name in cfg["names"]:
    cls = opts.get(name)
    if cls is None:
        continue
    row = {}
    for pname, p in cfg["problems"].items():
        fits = []
        for seed in cfg["seeds"]:
            try:
                prob = Problem(bounds=FloatVar(lb=(p["low"],) * p["d"], ub=(p["up"],) * p["d"]), minmax="min", obj_func=FUNCS[pname], log_to=None)
                g_best = cls(epoch=cfg["epoch"], pop_size=cfg["pop"]).solve(prob, seed=seed)
                fits.append(float(g_best.target.fitness))
            except Exception as exc:
                fits.append(None)
        row[pname] = fits
    out[name] = row
print(json.dumps(out))
"""


def run_clypto(names, args):
    classes = cy.get_all_optimizers()
    out = {}
    for name in names:
        if name in NO_EVOLVE:
            continue
        cls = classes.get(name)
        if cls is None:
            continue
        row = {}
        for pname, p in PROBLEMS.items():
            fits = []
            for seed in range(1, args.seeds + 1):
                try:
                    problem = cy.Problem(obj_func=p["f"], bounds=cy.NumberBounds(float, low=[p["low"]] * p["d"], up=[p["up"]] * p["d"]), sense="min")
                    g_best = cls(epoch=args.epoch, pop_size=args.pop).solve(problem, seed=seed)
                    fits.append(float(g_best.fitness))
                except Exception:
                    fits.append(None)
            row[pname] = fits
        out[name] = row
    return out


def run_mealpy(python, names, args):
    cfg = {
        "names": names,
        "seeds": list(range(1, args.seeds + 1)),
        "epoch": args.epoch,
        "pop": args.pop,
        "problems": {pname: {"d": p["d"], "low": p["low"], "up": p["up"]} for pname, p in PROBLEMS.items()},
    }
    proc = subprocess.run([python, "-c", MEALPY_SNIPPET, json.dumps(cfg)], capture_output=True, text=True, check=False)
    if proc.returncode:
        print("mealpy run failed:", proc.stderr[-800:], file=sys.stderr)
        return {}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _stats(fits):
    clean = [f for f in fits if f is not None and np.isfinite(f)]
    n_bad = len(fits) - len(clean)
    if not clean:
        return None
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
        "n_bad": n_bad,
    }


def _paired(c_fits, m_fits):
    """Paired per-seed diff (clypto - mealpy): clypto's Cython port largely mirrors mealpy-lts's
    RNG call order (verified on OriginalGWO: bit-identical fitness per seed), so a same-seed pair
    is far more sensitive to a real divergence than comparing independent distributions -- a
    consistent one-directional gap (e.g. OriginalWOA, clypto worse on every one of 5 seeds) can
    hide inside each library's own run-to-run variance under an unpaired comparison."""
    diffs = [c - m for c, m in zip(c_fits, m_fits, strict=True) if c is not None and m is not None and np.isfinite(c) and np.isfinite(m)]
    n = len(diffs)
    if n == 0:
        return None
    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.pstdev(diffs) if n > 1 else 0.0
    se = std_diff / (n**0.5) if n > 1 else 0.0
    # falls back to the raw gap (vs. a small absolute floor) when every paired diff is identical,
    # so a real, consistent divergence with zero variance still trips the threshold
    t = mean_diff / se if se > 1e-12 else mean_diff / 1e-9
    return {"mean_diff": mean_diff, "std_diff": std_diff, "n": n, "t": t}


def compare(clypto_rows, mealpy_rows, args):
    flags = {}
    for name in sorted(set(clypto_rows) | set(mealpy_rows)):
        c_row = clypto_rows.get(name)
        m_row = mealpy_rows.get(name)
        if c_row is None:
            flags[name] = {"flag": "clypto_error_or_missing"}
            continue
        if m_row is None:
            flags[name] = {"flag": "mealpy_missing_or_failed"}
            continue
        tol = args.tol * (2.0 if name in NONDETERMINISTIC else 1.0)
        entry = {}
        for pname in PROBLEMS:
            c_fits, m_fits = c_row.get(pname, []), m_row.get(pname, [])
            cs, ms = _stats(c_fits), _stats(m_fits)
            if cs is None:
                entry[pname] = {"flag": "clypto_all_invalid"}
                continue
            if ms is None:
                entry[pname] = {"flag": "mealpy_all_invalid", "clypto": cs}
                continue
            pair = _paired(c_fits, m_fits)
            entry[pname] = {"clypto": cs, "mealpy": ms, "paired": pair}
            if cs["n_bad"] > 0:
                entry[pname]["flag"] = f"clypto_invalid_runs:{cs['n_bad']}/{args.seeds}"
            elif pair is not None and abs(pair["t"]) > tol:
                entry[pname]["flag"] = "distribution_mismatch"
        if any("flag" in v for v in entry.values()):
            flags[name] = entry
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--pop", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=30, help="run seeds 1..N in each library")
    ap.add_argument("--tol", type=float, default=3.0, help="allowed |mean diff| in pooled std-devs before flagging")
    ap.add_argument("--only", default="", help="regex on the class name")
    ap.add_argument("--names-file", default="", help="JSON list of class names to restrict to")
    ap.add_argument("--mealpy-python", required=True, help="python of an isolated env with mealpy-lts installed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    all_names = sorted(cy.get_all_optimizers())
    if args.names_file:
        restrict = set(json.loads(Path(args.names_file).read_text()))
        all_names = [n for n in all_names if n in restrict]
    if args.only:
        all_names = [n for n in all_names if re.search(args.only, n)]
    names = [n for n in all_names if n not in NO_EVOLVE]

    print(f"running {len(names)} classes x {len(PROBLEMS)} problems x {args.seeds} seeds ...", file=sys.stderr)
    clypto_rows = run_clypto(names, args)
    mealpy_rows = run_mealpy(args.mealpy_python, names, args)
    flags = compare(clypto_rows, mealpy_rows, args)

    print(f"{len(names)} classes compared, {len(flags)} flagged")
    for name, entry in flags.items():
        print(f"  {name}: {entry.get('flag') or {p: v.get('flag') for p, v in entry.items() if 'flag' in v}}")

    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "clypto": clypto_rows, "mealpy": mealpy_rows, "flags": flags}, indent=1))


if __name__ == "__main__":
    main()
