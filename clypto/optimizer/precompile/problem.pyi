import typing

from clypto.optimizer.native.problem import Problem

_T = typing.TypeVar("_T")


@typing.overload
def problem(
    *,
    bounds: typing.Any = None,
    sense: str = "min",
    vectorized: bool = False,
    vectorize: typing.Optional[bool] = None,
    decode: bool = False,
    compile: bool = True,
    name: typing.Optional[str] = None,
    evaluator: typing.Optional[typing.Any] = None,
    obj_weights: typing.Optional[typing.Any] = None,
    seed: typing.Optional[int] = None,
    **kwargs: typing.Any,
) -> typing.Callable[[_T], _T]: ...


@typing.overload
def problem(
    target: typing.Callable,
    *,
    bounds: typing.Any = None,
    sense: str = "min",
    vectorized: bool = False,
    vectorize: typing.Optional[bool] = None,
    decode: bool = False,
    compile: bool = True,
    name: typing.Optional[str] = None,
    evaluator: typing.Optional[typing.Any] = None,
    obj_weights: typing.Optional[typing.Any] = None,
    seed: typing.Optional[int] = None,
    **kwargs: typing.Any,
) -> Problem: ...


@typing.overload
def problem(target: _T) -> _T: ...
