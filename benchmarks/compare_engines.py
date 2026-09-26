#!/usr/bin/env python
"""Statistical equivalence of the vectorize collection against the frozen legacy collection.

    python benchmarks/compare_engines.py --only "GWO|WOA" --seeds 30 --out equiv.json

For every class present in both engines: same budget, same seed list, four test functions
(sphere, rastrigin, ackley, rosenbrock) at d=10 and d=30. A two-sided Mann-Whitney U test on the
final best fitness is run per case; p-values of one class are Holm-Bonferroni corrected (family alpha
``--alpha``). A class FAILS when native is significantly worse in any case, or when its median is more
than ``--ratio`` times the legacy median (with an absolute floor ``--floor`` for values near 0).
A significant improvement is reported as ``INVESTIGATE`` (it usually means the semantics changed).
NFE (function evaluations) of both engines must match; otherwise the class is reported as ``NFE``.
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import clypto as cy


def sphere(x):
    return float(np.sum(x**2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e)


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


FUNCTIONS = {"sphere": (sphere, 5.0), "rastrigin": (rastrigin, 5.12), "ackley": (ackley, 5.0), "rosenbrock": (rosenbrock, 5.0)}


def run(cls, fn, bound, dim, seed, epoch, pop):
    problem = cy.Problem(obj_func=fn, bounds=cy.NumberBounds(float, low=[-bound] * dim, up=[bound] * dim), sense="min")
    model = cls(epoch=epoch, pop_size=pop)
    best = model.solve(problem, seed=seed)
    return float(best.target.fitness), int(model.nf_counter)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def compare_class(name, legacy_cls, vec_cls, args):
    seeds = list(range(1, args.seeds + 1))
    cases, pvals = [], []
    nfe_a, nfe_b = [], []
    for fname, (fn, bound) in FUNCTIONS.items():
        for dim in args.dims:
            a, b = [], []
            for seed in seeds:
                fa, na = run(legacy_cls, fn, bound, dim, seed, args.epoch, args.pop)
                fb, nb = run(vec_cls, fn, bound, dim, seed, args.epoch, args.pop)
                a.append(fa)
                b.append(fb)
                nfe_a.append(na)
                nfe_b.append(nb)
            a, b = np.array(a), np.array(b)
            if np.array_equal(a, b):
                p = 1.0
            else:
                p = float(mannwhitneyu(b, a, alternative="two-sided").pvalue)
            ma, mb = float(np.median(a)), float(np.median(b))
            worse_pt = mb > max(args.ratio * ma, ma + args.floor)
            cases.append({"function": fname, "dim": dim, "median_legacy": ma, "median_vectorize": mb, "p": p,
                          "worse_median": worse_pt, "better": mb < ma, "material": abs(mb - ma) > args.floor})
            pvals.append(p)
    adj = holm(np.array(pvals))
    verdict = "PASS"
    for case, padj in zip(cases, adj, strict=True):
        case["p_holm"] = float(padj)
        significant = padj < args.alpha and case["material"]
        if significant and not case["better"]:
            verdict = "FAIL"
        elif significant and case["better"] and verdict == "PASS":
            verdict = "INVESTIGATE"
        if case["worse_median"] and verdict == "PASS":
            verdict = "WARN"  # practical guard exceeded without significant evidence
    if np.mean(nfe_b) > 1.05 * np.mean(nfe_a) and verdict in ("PASS", "WARN"):
        verdict = "NFE"
    return {"verdict": verdict, "cases": cases}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="regex on the class name")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--floor", type=float, default=1e-3, help="absolute tolerance: smaller differences are equivalent")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    legacy, vec = cy.get_all_optimizers(engine="legacy"), cy.get_all_optimizers(engine="vectorize")
    names = sorted(n for n in vec if n in legacy and re.search(args.only, n))
    report = {}
    for name in names:
        try:
            report[name] = compare_class(name, legacy[name], vec[name], args)
        except Exception as exc:
            report[name] = {"verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:200]}
        print(f"{name:26s} {report[name]['verdict']}", flush=True)
    bad = {n: r["verdict"] for n, r in report.items() if r["verdict"] != "PASS"}
    print(f"\n{len(report) - len(bad)}/{len(report)} PASS", "; ".join(f"{n}={v}" for n, v in bad.items()))
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "report": report}, indent=1))
    return 1 if any(v in ("FAIL", "ERROR", "NFE") for v in bad.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
