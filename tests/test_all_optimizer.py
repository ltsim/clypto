import numpy as np
import pytest

import clypto as cy

from tests._engines import ENGINES, NO_EVOLVE

CASES = [
    pytest.param(cls, id=f"{engine}-{name}", marks=[pytest.mark.xfail(raises=NotImplementedError, strict=True)] if name in NO_EVOLVE else [])
    for engine in ENGINES for name, cls in cy.get_all_optimizers(engine=engine).items()
]
OPTIMIZERS = cy.get_all_optimizers()
N_DIMS = 5


def objective(solution):
    return np.sum(solution**2)


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.NumberBounds(float, low=[-1] * N_DIMS, up=[1] * N_DIMS),
        sense="min",
    )


@pytest.mark.smoke
def test_all_optimizers_discovered():
    assert len(OPTIMIZERS) > 0


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", CASES)
def test_all_optimizers_run(optimizer, problem):
    # epoch=50 / pop_size=25 is the smallest budget every optimizer accepts:
    # several validators scale their ranges with the budget (e.g. CHIO.max_age,
    # CRO.restart_count), and tiny populations crash some solvers.
    model = optimizer(epoch=50, pop_size=25)
    g_best = model.solve(problem, seed=1)

    solution = g_best.solution
    assert isinstance(solution, np.ndarray)
    assert solution.shape == (problem.n_dims,)
    assert np.all(np.isfinite(solution))
    # Short runs can overshoot bounds by ~2% (OriginalHBO), so allow a small slack.
    assert np.all(problem.bounds.low - 0.05 <= solution) and np.all(solution <= problem.bounds.up + 0.05)

    target = np.asarray(g_best.target.objectives).flatten()
    assert target.size == 1
    assert np.all(np.isfinite(target))
