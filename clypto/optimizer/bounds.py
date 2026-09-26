"""Search-space blocks and the ``Bounds`` that concatenates them.

Every metaheuristic moves a continuous ``float64`` vector, so each block maps
its values to a slice of that vector (``encode``), keeps a moved slice valid
(``correct``) and maps it back (``decode``):

* ``NumberBounds(float, low, up)``: the values themselves.
* ``NumberBounds(int | bool, low, up)``: searched on ``[low - 0.5, up + 0.5]``
  and rounded, so every value, both ends included, owns a cell of width 1.
* ``TransferBounds(int | bool, n_vars, tf_func)``: a transfer function turns
  each coordinate into the probability of a 1.
* ``StringBounds(valid_sets)``: one label per variable (any hashable), encoded
  as its index in the variable's set; ``SequenceBounds`` picks one sequence.
* ``PermutationBounds(valid_set)``: random keys, decoded by ``argsort``.
"""
import abc

import numpy as np

from clypto.optimizer import transfer

__all__ = ["BaseBounds", "Bounds", "NumberBounds", "PermutationBounds", "SequenceBounds", "StringBounds", "TransferBounds"]


class BaseBounds(abc.ABC):
    """One block of ``n_vars`` decision variables with its own random stream."""

    value_type: type = float

    def __init__(self, low, up, name=None):
        self.low = np.asarray(low, dtype=np.float64).ravel()
        self.up = np.asarray(up, dtype=np.float64).ravel()
        if self.low.shape != self.up.shape or self.low.size == 0:
            raise ValueError("low and up must be non-empty and of the same length.")
        if np.any(self.low > self.up):
            raise ValueError("Every low must be <= its up.")
        self.name = name
        self.seed = None

    @property
    def n_vars(self):
        return self.low.size

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.generator = np.random.default_rng(seed)

    def encode(self, value):
        """Values (decoded type) -> ``float64`` slice of the search vector."""
        return np.asarray(value, dtype=np.float64).ravel()

    def correct(self, x):
        """A moved slice -> a valid slice."""
        return np.clip(x, self.low, self.up)

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

    def __init__(self, value_type=float, low=None, up=None, n_vars=None, name=None):
        if value_type not in (float, int, bool):
            raise TypeError(f"value_type must be float, int or bool, got {value_type!r}.")
        if value_type is bool:
            low, up = 0 if low is None else low, 1 if up is None else up
        if low is None or up is None:
            raise ValueError("NumberBounds needs low and up.")
        low, up = np.asarray(low, dtype=np.float64), np.asarray(up, dtype=np.float64)
        if n_vars is not None:
            low, up = np.broadcast_to(low, (n_vars,)), np.broadcast_to(up, (n_vars,))
        self.value_type = value_type
        self.values_low, self.values_up = low.ravel(), up.ravel()
        half = 0.0 if value_type is float else 0.5
        super().__init__(low - half, up + half, name)

    def decode(self, x):
        x = self.correct(x)
        if self.value_type is float:
            return x
        return np.clip(np.floor(x + 0.5), self.values_low, self.values_up).astype(self.value_type)

    def generate(self):
        if self.value_type is float:
            return self.generator.uniform(self.low, self.up)
        return self.generator.integers(self.values_low, self.values_up + 1).astype(self.value_type)


class TransferBounds(BaseBounds):
    """``n_vars`` binary values (``int`` 0/1 or ``bool``) moved on ``[low, up]``.

    ``correct`` maps each coordinate through ``tf_func`` (a name from
    ``clypto.optimizer.transfer``) to the probability of a 1 and samples it.
    ``all_zeros=False`` forbids the all-zero vector.
    """

    def __init__(self, value_type=int, n_vars=1, tf_func="vstf_01", low=-8.0, up=8.0, all_zeros=True, name=None):
        if value_type not in (int, bool):
            raise TypeError(f"value_type must be int or bool, got {value_type!r}.")
        if not hasattr(transfer, tf_func):
            raise ValueError(f"Unknown transfer function {tf_func!r}.")
        super().__init__(np.full(n_vars, low), np.full(n_vars, up), name)
        self.value_type = value_type
        self.tf_func = getattr(transfer, tf_func)
        self.all_zeros = all_zeros

    def _ensure_one(self, x):
        if not self.all_zeros and not np.any(x):
            x[self.generator.integers(0, len(x))] = 1
        return x

    def correct(self, x):
        p = self.tf_func(np.clip(x, self.low, self.up))
        return self._ensure_one(np.where(self.generator.random(len(p)) < p, 1.0, 0.0))

    def decode(self, x):
        return (np.asarray(x) >= 0.5).astype(self.value_type)

    def generate(self):
        return self._ensure_one(self.generator.integers(0, 2, self.n_vars)).astype(self.value_type)


class StringBounds(BaseBounds):
    """One label per variable, chosen from that variable's set (any hashable labels).

    ``valid_sets`` is one set (one variable) or a sequence of sets (one each).
    """

    def __init__(self, valid_sets, name=None):
        valid_sets = list(valid_sets)
        if not valid_sets:
            raise ValueError("valid_sets must not be empty.")
        if not all(isinstance(s, (list, tuple, np.ndarray)) for s in valid_sets):
            valid_sets = [valid_sets]
        self.valid_sets = [tuple(s) for s in valid_sets]
        if any(len(s) < 1 for s in self.valid_sets):
            raise ValueError("Every set needs at least one label.")
        self._index = [{label: i for i, label in enumerate(s)} for s in self.valid_sets]
        sizes = np.array([len(s) for s in self.valid_sets], dtype=np.float64)
        super().__init__(np.full(sizes.size, -0.5), sizes - 0.5, name)

    value_type = object

    def encode(self, value):
        return np.array([index[label] for index, label in zip(self._index, value, strict=True)], dtype=np.float64)

    def decode(self, x):
        idx = np.clip(np.floor(self.correct(x) + 0.5), 0, self.up - 0.5).astype(int)
        return [s[i] for s, i in zip(self.valid_sets, idx, strict=True)]

    def generate(self):
        return [s[self.generator.integers(len(s))] for s in self.valid_sets]


class SequenceBounds(StringBounds):
    """One variable whose value is one of ``valid_sets`` (sequences), returned as ``return_type``."""

    def __init__(self, valid_sets, return_type=tuple, name=None):
        super().__init__([[tuple(v) for v in valid_sets]], name)
        self.return_type = return_type

    def decode(self, x):
        return [self.return_type(v) for v in super().decode(x)]


class PermutationBounds(BaseBounds):
    """An ordering of ``valid_set``: random keys in ``[0, n - 1]``, decoded by ``argsort``."""

    value_type = object

    def __init__(self, valid_set, name=None):
        self.valid_set = tuple(valid_set)
        if len(self.valid_set) < 2:
            raise ValueError("valid_set needs at least two items.")
        self._index = {item: i for i, item in enumerate(self.valid_set)}
        n = len(self.valid_set)
        super().__init__(np.zeros(n), np.full(n, n - 1.0), name)

    def encode(self, value):
        keys = np.empty(self.n_vars)
        for position, item in enumerate(value):
            keys[self._index[item]] = position
        return keys

    def decode(self, x):
        return [self.valid_set[i] for i in np.argsort(self.correct(x), kind="stable")]

    def generate(self):
        return [self.valid_set[i] for i in self.generator.permutation(self.n_vars)]


class Bounds:
    """The search space: blocks concatenated into one ``float64`` vector.

    ``Bounds(block, [block, ...], ...)``; unnamed blocks are named ``x0``,
    ``x1``, ... by position, and names must be unique. ``low``/``up`` span the
    whole vector and ``dtype`` is always ``float64``.
    """

    dtype = np.float64

    def __init__(self, *blocks):
        flat = []
        for block in blocks:
            if isinstance(block, Bounds):
                flat.extend(block.blocks)
            elif isinstance(block, BaseBounds):
                flat.append(block)
            elif isinstance(block, (list, tuple)):
                flat.extend(Bounds(*block).blocks)
            else:
                raise TypeError(f"Expected bounds blocks, got {type(block).__name__}.")
        if not flat:
            raise ValueError("Bounds needs at least one block.")
        for i, block in enumerate(flat):
            if block.name is None:
                block.name = f"x{i}"
        names = [block.name for block in flat]
        if len(set(names)) != len(names):
            raise ValueError(f"Block names must be unique, got {names}.")
        self.blocks = tuple(flat)
        self.low = np.concatenate([b.low for b in flat])
        self.up = np.concatenate([b.up for b in flat])
        self._slices = np.cumsum([0] + [b.n_vars for b in flat])
        # np.clip is element-wise, so a float-only space corrects a row or a matrix at once.
        self._all_float = all(type(b) is NumberBounds and b.value_type is float for b in flat)
        self.seed = None

    @property
    def n_dims(self):
        return self.low.size

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        for block in self.blocks:
            block.seed = seed

    def _parts(self, x):
        return ((b, x[..., lo:hi]) for b, lo, hi in zip(self.blocks, self._slices[:-1], self._slices[1:], strict=True))

    def encode(self, values):
        """One value (or sequence of values) per block -> search vector."""
        return np.concatenate([b.encode(v) for b, v in zip(self.blocks, values, strict=True)])

    def correct(self, x):
        """Valid search vector (or ``(n, n_dims)`` matrix of them)."""
        if self._all_float:
            return np.clip(x, self.low, self.up)
        if np.ndim(x) == 2:
            return np.array([self.correct(row) for row in x])
        return np.concatenate([b.correct(part) for b, part in self._parts(np.asarray(x, dtype=np.float64))])

    def decode(self, x):
        """Search vector -> ``{name: value}``; a one-variable block gives a scalar."""
        out = {}
        for b, part in self._parts(np.asarray(x, dtype=np.float64)):
            value = b.decode(part)
            out[b.name] = value[0] if b.n_vars == 1 else value
        return out

    def generate(self):
        """A random search vector (every block draws from its own stream)."""
        return self.encode([b.generate() for b in self.blocks])
