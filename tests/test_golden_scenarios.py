#!/usr/bin/env python
"""Bit-identity of every optimizer against the classic implementation in more scenarios.

``baseline_scenarios_2026a0.json`` was captured from the classic (pre-native) collection for a maximization
problem, a 12-D Rastrigin and both problems in ``mode="swarm"``. Scenarios where the classic code raised
are recorded as errors and skipped.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import clypto as cy
from tests._engines import CHANGED, ENGINES, skip_if_not_bit_identical

BASELINE_PATH = Path(__file__).parent / "golden" / "baseline_scenarios_2026a0.json"

pytestmark = pytest.mark.skipif(not BASELINE_PATH.exists(), reason="scenario baseline not available")


def _sphere(x):
    return float(-np.sum(x**2))


def _rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


SCENARIOS = {
    "max8": {"f": _sphere, "d": 8, "mm": "max", "n": 20, "ep": 30, "seed": 2, "mode": None},
    "swarm6": {"f": _rastrigin, "d": 6, "mm": "min", "n": 20, "ep": 30, "seed": 3, "mode": "swarm"},
    "min12": {"f": _rastrigin, "d": 12, "mm": "min", "n": 30, "ep": 40, "seed": 4, "mode": None},
    "swarm-max": {"f": _sphere, "d": 7, "mm": "max", "n": 20, "ep": 25, "seed": 5, "mode": "swarm"},
}


def _cases():
    baseline = json.loads(BASELINE_PATH.read_text())
    return [(name, label) for name in sorted(baseline.keys() - CHANGED.keys()) for label in SCENARIOS if "fitness" in baseline[name].get(label, {})]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name,label", _cases())
def test_matches_classic_scenario(engine, name, label):
    skip_if_not_bit_identical(engine, name)
    expected = json.loads(BASELINE_PATH.read_text())[name][label]
    s = SCENARIOS[label]
    problem = cy.Problem(obj_func=s["f"], bounds=cy.NumberBounds(float, low=[-5.0] * s["d"], up=[5.0] * s["d"]), sense=s["mm"])
    kwargs = {"epoch": s["ep"], "pop_size": s["n"]}
    if s["mode"]:
        kwargs["mode"] = s["mode"]
    g_best = cy.get_all_optimizers(engine=engine)[name](**kwargs).solve(problem, seed=s["seed"])
    digest = hashlib.sha1(np.asarray(g_best.solution, dtype=np.float64).tobytes()).hexdigest()
    assert float(g_best.target.fitness) == pytest.approx(expected["fitness"], abs=1e-9)
    assert digest == expected["solution_sha1"]
