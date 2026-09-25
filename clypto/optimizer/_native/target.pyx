"""Native ``Target`` used by the Cython collection (see ``clypto.optimizer.target``)."""
import numbers

import numpy as np

cdef tuple SUPPORTED_ARRAY = (tuple, list, np.ndarray)


cdef class NativeTarget:
    def __init__(self, objectives=None, weights=None):
        if type(objectives) not in SUPPORTED_ARRAY:
            if not isinstance(objectives, numbers.Number):
                raise ValueError(
                    "Invalid objectives. It should be a list, tuple, np.ndarray, int or float."
                )
            objectives = [objectives]
        self.objectives = np.array(objectives).flatten()

        cdef Py_ssize_t n = len(self.objectives)
        if weights is None:
            self.weights = n
        elif type(weights) in SUPPORTED_ARRAY:
            self.weights = np.array(weights).flatten()
        elif isinstance(weights, numbers.Number):
            self.weights = np.array([weights] * n).flatten()
        else:
            raise ValueError("Invalid weights. It should be a list, tuple, np.ndarray.")

        fitness_weights = self.weights
        if not (type(fitness_weights) in SUPPORTED_ARRAY and len(fitness_weights) == n):
            fitness_weights = n * (1.0,)
        self.fitness = np.dot(fitness_weights, self.objectives)

    cpdef NativeTarget copy(self):
        # Fields are already normalized; skip the constructor's re-validation.
        cdef NativeTarget new = NativeTarget.__new__(NativeTarget)
        new.objectives = self.objectives.copy()
        new.weights = self.weights
        new.fitness = self.fitness
        return new

    def __str__(self):
        return f"Objectives: {self.objectives}, Fitness: {self.fitness}"
