#!/usr/bin/env python
# Created by "Thieu" at 21:32, 14/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numbers
import operator
import typing

import numpy as np

T = typing.TypeVar("T")

Number = typing.Union[int, float]
Bound = typing.Union[tuple[T, T], list[T]]

SEQUENCE_TYPES: tuple[type, ...] = (list, tuple, np.ndarray)
DIGIT_TYPES: tuple[type, ...] = (int, np.integer)
REAL_TYPES: tuple[type, ...] = (float, np.floating)

INF = float("inf")
NEG_INF = float("-inf")


def _is_in_bound(value: Number, bound: Bound[Number]) -> bool:
    """Check whether ``value`` lies inside ``bound``.

    A ``tuple`` bound is treated as exclusive ``(<)`` and a ``list`` bound as
    inclusive ``(<=)``. Infinite endpoints are handled explicitly.
    """
    if isinstance(bound, tuple):
        ops: typing.Callable[[typing.Any, typing.Any], bool] = operator.lt
    elif isinstance(bound, list):
        ops = operator.le
    else:
        raise TypeError(
            f"'bound' must be a tuple (exclusive) or list (inclusive), "
            f"got {type(bound).__name__}."
        )

    if len(bound) != 2:
        raise ValueError(f"'bound' must have exactly 2 elements, got {len(bound)}.")

    low, high = bound

    low_ok = low == NEG_INF or ops(low, value)
    high_ok = high == INF or ops(value, high)
    return low_ok and high_ok


def _is_str_in_list(value: str, my_list: typing.Optional[typing.Sequence[str]]) -> bool:
    """Return ``True`` if ``value`` is a string contained in ``my_list``."""
    if isinstance(value, str) and my_list is not None:
        return value in my_list
    return False


def _is_real_number(value: typing.Any) -> bool:
    """Numeric check that excludes ``bool`` (which is a subclass of ``int``)."""
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


class Validator:
    @staticmethod
    def check_int(
        name: str,
        value: Number,
        bound: typing.Optional[Bound[int]] = None,
    ) -> int:
        if _is_real_number(value):
            if bound is None or _is_in_bound(value, bound):
                return int(value)

        suffix = "" if bound is None else f" and value should be in range: {bound}"
        raise TypeError(f"'{name}' should be an integer{suffix}.")

    @staticmethod
    def check_float(
        name: str,
        value: Number,
        bound: typing.Optional[Bound[float]] = None,
    ) -> float:
        if _is_real_number(value):
            if bound is None or _is_in_bound(value, bound):
                return float(value)

        suffix = "" if bound is None else f" and value should be in range: {bound}"
        raise TypeError(f"'{name}' should be a float{suffix}.")

    @staticmethod
    def check_str(
        name: str,
        value: str,
        bound: typing.Optional[typing.Sequence[str]] = None,
    ) -> str:
        if isinstance(value, str):
            if bound is None or _is_str_in_list(value, bound):
                return value

        suffix = "" if bound is None else f" and value should be one of: {bound}"
        raise TypeError(f"'{name}' should be a string{suffix}.")

    @staticmethod
    def check_bool(
        name: str,
        value: bool,
        bound: tuple[bool, ...] = (True, False),
    ) -> bool:
        # `is` comparisons avoid 1/0 being treated as True/False.
        if isinstance(value, bool) and any(value is b for b in bound):
            return value

        raise TypeError(f"'{name}' should be a boolean, one of: {bound}.")

    @staticmethod
    def check_tuple_int(
        name: str,
        values: typing.Sequence[int],
        bounds: typing.Optional[typing.Sequence[Bound[int]]] = None,
    ) -> typing.Sequence[int]:
        if isinstance(values, SEQUENCE_TYPES) and len(values) > 1:
            if all(isinstance(item, DIGIT_TYPES) for item in values):
                if bounds is None:
                    return values
                if len(bounds) == len(values) and all(
                    _is_in_bound(item, b) for item, b in zip(values, bounds)
                ):
                    return values

        suffix = "" if bounds is None else f" and values should be in range: {bounds}"
        raise TypeError(f"'{name}' should be a sequence of integers{suffix}.")

    @staticmethod
    def check_tuple_float(
        name: str,
        values: typing.Sequence[float],
        bounds: typing.Optional[typing.Sequence[Bound[float]]] = None,
    ) -> typing.Sequence[float]:
        if isinstance(values, SEQUENCE_TYPES) and len(values) > 1:
            if all(_is_real_number(item) for item in values):
                if bounds is None:
                    return values
                if len(bounds) == len(values) and all(
                    _is_in_bound(item, b) for item, b in zip(values, bounds)
                ):
                    return values

        suffix = "" if bounds is None else f" and values should be in range: {bounds}"
        raise TypeError(f"'{name}' should be a sequence of floats{suffix}.")

    @staticmethod
    def check_list_tuple(
        name: str,
        value: typing.Union[list[T], tuple[T, ...]],
        data_type: str,
    ) -> list[T]:
        if isinstance(value, (list, tuple)) and len(value) >= 1:
            return list(value)

        raise TypeError(
            f"'{name}' should be a list or tuple of {data_type}, and length >= 1."
        )

    @staticmethod
    def check_is_instance(name: str, value: T, class_type: type) -> T:
        if isinstance(value, class_type):
            return value

        raise TypeError(
            f"'{name}' should be an instance of {class_type.__name__} class."
        )

    @staticmethod
    def check_is_int_and_float(
        name: str,
        value: Number,
        bound_int: typing.Optional[Bound[int]] = None,
        bound_float: typing.Optional[Bound[float]] = None,
    ) -> Number:
        # Order matters: bool is a subclass of int, so reject it up front.
        if isinstance(value, bool):
            pass
        elif isinstance(value, DIGIT_TYPES):
            if bound_int is None or _is_in_bound(value, bound_int):
                return int(value)
        elif isinstance(value, REAL_TYPES):
            if bound_float is None or _is_in_bound(value, bound_float):
                return float(value)

        int_suffix = "" if bound_int is None else f" in range: {bound_int}"
        float_suffix = "" if bound_float is None else f" in range: {bound_float}"
        raise TypeError(f"'{name}' can be int{int_suffix}, or float{float_suffix}.")
