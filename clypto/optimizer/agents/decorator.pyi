import typing

_C = typing.TypeVar("_C")

__all__ = ['agent']

@typing.overload
def agent(cls: _C) -> _C: ...
@typing.overload
def agent(*, compile: bool = False) -> typing.Callable[[_C], _C]: ...
