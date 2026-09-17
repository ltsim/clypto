"""Regression: the public bases keep the injected API visible to mypy.

``@cy.optimizer``/``@cy.legacy`` inject a base at runtime, which a type checker
cannot see. Typed algorithms should inherit ``cy.DecoratedOptimizer`` /
``cy.LegacyOptimizer`` instead; this test runs mypy over the fixtures in
``typed_optimizer_api.py`` and asserts those members resolve.
"""
import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import clypto as cy

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = pathlib.Path(__file__).resolve().parent / "typed_optimizer_api.py"


def _load_fixture():
    spec = importlib.util.spec_from_file_location("typed_optimizer_api", FIXTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_typed_optimizer_api_has_no_mypy_errors():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            str(FIXTURE),
            "--config-file",
            str(ROOT / "pyproject.toml"),
            "--follow-imports=silent",
            "--no-error-summary",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def problem():
    return cy.Problem(
        obj_func=lambda solution: float(np.sum(solution**2)),
        bounds=cy.FloatVar(lb=[-5.0] * 3, ub=[5.0] * 3),
        minmax="min",
    )


def test_typed_patterns_solve(problem):
    fixture = _load_fixture()

    new = fixture.NewStyleSearch(epoch=5, pop_size=6)
    assert new.solve(problem, seed=1).fitness is not None

    classic = fixture.ClassicSearch(epoch=5, pop_size=6)
    assert classic.solve(problem, seed=1).target.fitness is not None
