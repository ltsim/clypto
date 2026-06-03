import numpy as np


def init_buffer(n_pop: int, ndim: int) -> np.ndarray:
    return np.full((n_pop, ndim + 1), np.nan)


class Population:
    def __init__(self, n_pop: int, ndim: int):
        self.__buffer = init_buffer(n_pop, ndim)

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass
