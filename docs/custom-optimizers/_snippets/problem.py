import numpy as np
import clypto as cy


def sphere(solution):
    return np.sum(solution ** 2)


problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.NumberBounds(float, low=[-10.0] * 30, up=[10.0] * 30),
    sense="min",
)
