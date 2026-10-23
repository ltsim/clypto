from _typeshed import Incomplete
from clypto.optimizer.native.utils import validator as validator

class Termination:
    '''
    Define custom single/multiple Stopping Conditions (termination criteria) for the Optimizer.

    Notes
    ~~~~~
    + By default, the stopping condition in the Optimizer class is based on the maximum number of generations (epochs/iterations).
    + Using this class allows you to override the default termination criteria. If multiple stopping conditions are specified, the first one that occurs will be used.

    + In general, there are four types of termination criteria: FE, MG, TB, and ES.
        + MG: Maximum Generations / Epochs / Iterations
        + FE: Maximum Number of Function Evaluations
        + TB: Time Bound - If you want your algorithm to run for a fixed amount of time (e.g., K seconds), especially when comparing different algorithms.
        + ES: Early Stopping -  Similar to the idea in training neural networks (stop the program if the global best solution has not improved by epsilon after K epochs).

    + Parameters for Termination class, set it to None if you don\'t want to use it
        + max_epoch (int): Indicates the maximum number of generations for the MG type.
        + max_fe (int): Indicates the maximum number of function evaluations for the FE type.
        + max_time (float): Indicates the maximum amount of time for the TB type.
        + max_early_stop (int): Indicates the maximum number of epochs for the ES type.
            + epsilon (float): (Optional) This is used for the ES termination type (default value: 1e-10).
        + termination (dict): (Optional) A dictionary of termination criteria.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import BBO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> p1 = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="C-params"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function,
    >>>     "name": "Test Function"
    >>> }
    >>>
    >>> term_dict = {
    >>>     "max_epoch": 1000,
    >>>     "max_fe": 100000,  # 100000 number of function evaluation
    >>>     "max_time": 10,     # 10 seconds to run the program
    >>>     "max_early_stop": 15    # 15 epochs if the best fitness is not getting better we stop the program
    >>> }
    >>> model1 = BBO.OriginalBBO(epoch=1000, pop_size=50)
    >>> model1.solve(p1, termination=term_dict)
    '''
    max_epoch: Incomplete
    max_fe: Incomplete
    max_time: Incomplete
    max_early_stop: Incomplete
    epsilon: float
    def __init__(self, max_epoch=None, max_fe=None, max_time=None, max_early_stop=None, **kwargs) -> None: ...
    def get_name(self): ...
    start_epoch: Incomplete
    start_fe: Incomplete
    start_time: Incomplete
    start_threshold: Incomplete
    def set_start_values(self, start_epoch, start_fe, start_time, start_threshold) -> None: ...
    message: str
    def should_terminate(self, current_epoch, current_fe, current_time, current_threshold): ...
