#!/usr/bin/env python
# Created by "LTSIM" at 07:57, 16/03/2022 ----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
import numpy as np
import pytest

import clypto as cy

all_optimizer = cy.get_all_optimizers()


@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution**2)

    return cy.Problem(
        obj_func=objective_function,
        bounds=cy.FloatVar(lb=[-1, -1, -1, -1, -1], ub=[1, 1, 1, 1, 1]),
        minmax="min",
    )


@pytest.mark.parametrize("optimizer", all_optimizer.values(), ids=all_optimizer.keys())
def test_all_optimizer_test(optimizer, problem):
    assert optimizer is not None

    model = optimizer(epoch=101, pop_size=25)
    g_best = model.solve(problem)

    assert isinstance(model, cy.Optimizer)
    assert isinstance(g_best.solution, np.ndarray)
