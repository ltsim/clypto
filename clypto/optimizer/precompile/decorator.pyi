import typing

_C = typing.TypeVar("_C")

__all__ = ['optimizer']

@typing.overload
def optimizer(cls: _C) -> _C: ...
@typing.overload
def optimizer(*, agent: typing.Any | None = None, compile: bool = False) -> typing.Callable[[_C], _C]: ...
