import abc
import numpy as np
from _typeshed import Incomplete

__all__ = ['BaseBounds', 'NumberBounds', 'TransferBounds', 'StringBounds', 'SequenceBounds', 'PermutationBounds', 'Bounds']

class BaseBounds(abc.ABC, metaclass=abc.ABCMeta):
    """One block of ``n_vars`` decision variables with its own random stream."""
    value_type: type
    low: Incomplete
    up: Incomplete
    name: Incomplete
    def __init__(self, low, up, name=None) -> None: ...
    @property
    def n_vars(self): ...
    generator: Incomplete
    @property
    def seed(self): ...
    @seed.setter
    def seed(self, seed) -> None: ...
    def encode(self, value):
        """Values (decoded type) -> ``float64`` slice of the search vector."""
    def correct(self, x):
        """A moved slice -> a valid slice."""
    @abc.abstractmethod
    def decode(self, x):
        """Slice of the search vector -> values of ``value_type``."""
    @abc.abstractmethod
    def generate(self):
        """Random values of ``value_type`` drawn from this block's stream."""

class NumberBounds(BaseBounds):
    """``n_vars`` numbers of ``value_type`` (``float``, ``int`` or ``bool``) in ``[low, up]``.

    ``low``/``up`` are scalars (broadcast to ``n_vars``, default 1) or sequences.
    ``bool`` defaults to ``low=0, up=1``.
    """
    value_type: Incomplete
    def __init__(self, value_type=..., low=None, up=None, n_vars=None, name=None) -> None: ...
    def decode(self, x): ...
    def generate(self): ...

class TransferBounds(BaseBounds):
    """``n_vars`` binary values (``int`` 0/1 or ``bool``) moved on ``[low, up]``.

    ``correct`` maps each coordinate through ``tf_func`` (a name from
    ``clypto.optimizer.transfer``) to the probability of a 1 and samples it.
    ``all_zeros=False`` forbids the all-zero vector.
    """
    value_type: Incomplete
    tf_func: Incomplete
    all_zeros: Incomplete
    def __init__(self, value_type=..., n_vars: int = 1, tf_func: str = 'vstf_01', low: float = -8.0, up: float = 8.0, all_zeros: bool = True, name=None) -> None: ...
    def correct(self, x): ...
    def decode(self, x): ...
    def generate(self): ...

class StringBounds(BaseBounds):
    """One label per variable, chosen from that variable's set (any hashable labels).

    ``valid_sets`` is one set (one variable) or a sequence of sets (one each).
    """
    valid_sets: Incomplete
    def __init__(self, valid_sets, name=None) -> None: ...
    value_type = object
    def encode(self, value): ...
    def decode(self, x): ...
    def generate(self): ...

class SequenceBounds(StringBounds):
    """One variable whose value is one of ``valid_sets`` (sequences), returned as ``return_type``."""
    return_type: Incomplete
    def __init__(self, valid_sets, return_type=..., name=None) -> None: ...
    def decode(self, x): ...

class PermutationBounds(BaseBounds):
    """An ordering of ``valid_set``: random keys in ``[0, n - 1]``, decoded by ``argsort``."""
    value_type = object
    valid_set: Incomplete
    def __init__(self, valid_set, name=None) -> None: ...
    def encode(self, value): ...
    def decode(self, x): ...
    def generate(self): ...

class Bounds:
    """The search space: blocks concatenated into one ``float64`` vector.

    ``Bounds(block, [block, ...], ...)``; unnamed blocks are named ``x0``,
    ``x1``, ... by position, and names must be unique. ``low``/``up`` span the
    whole vector and ``dtype`` is always ``float64``.
    """
    dtype = np.float64
    blocks: Incomplete
    low: Incomplete
    up: Incomplete
    def __init__(self, *blocks) -> None: ...
    @property
    def n_dims(self): ...
    @property
    def seed(self): ...
    @seed.setter
    def seed(self, seed) -> None: ...
    def encode(self, values):
        """One value (or sequence of values) per block -> search vector."""
    def correct(self, x):
        """Valid search vector (or ``(n, n_dims)`` matrix of them)."""
    def decode(self, x):
        """Search vector -> ``{name: value}``; a one-variable block gives a scalar."""
    def generate(self):
        """A random search vector (every block draws from its own stream)."""
