import numpy as np
import clypto as cy


def sphere(solution):
    return np.sum(solution ** 2)


problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.FloatVar(lb=[-10.0] * 30, ub=[10.0] * 30),
    minmax="min",
)
