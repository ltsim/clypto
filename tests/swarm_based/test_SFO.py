#!/usr/bin/env python
# Created by "Thieu" at 11:50, 21/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest

from clypto import FloatVar, ClassicOptimizer
from clypto.collection.swarm_based import SFO


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
        "log_to": None
    }
    return problem


def test_SFO_results(problem):
    models = [
        SFO.OriginalSFO(epoch=10, pop_size=50, pp=0.1, AP=4, epxilon=0.0001),
        SFO.ImprovedSFO(epoch=10, pop_size=50, pp=0.1)
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, ClassicOptimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
