#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""Shared building blocks for the ``@cy.agent``/``@cy.optimizer``/``@cy.legacy``
decorators.

Everything here is class-level glue rather than algorithm code: parsing the
``[...]`` declaration syntax, injecting a base class, collecting declarations
from a class hierarchy, rewriting class source for the Cython build, and
dispatching that build.
"""

import re
import typing

__all__ = [
    "Declaration",
    "parse_declaration_params",
    "decorate_with_base",
    "collect_declarations",
    "class_source_with_base",
    "maybe_compile",
]


def parse_declaration_params(params: typing.Any) -> tuple[typing.Any, typing.Any, typing.Any]:
    """Normalize ``Attribute[...]`` / ``Argument[...]`` parameters.

    Accepts ``[dtype]``, ``[dtype, bound]`` and ``[dtype, bound, default]``.
    ``...`` (Ellipsis) in the bound position means "no bound".
    """
    if not isinstance(params, tuple):
        params = (params,)

    if len(params) > 3:
        raise TypeError(
            f"Expected at most 3 declaration parameters (type, bound, default), got {len(params)}."
        )

    dtype = params[0] if len(params) > 0 else None
    bound = params[1] if len(params) > 1 else None
    default = params[2] if len(params) > 2 else None

    if bound is Ellipsis:
        bound = None

    return dtype, bound, default


class Declaration:
    """Base for the ``Attribute``/``Argument`` bracket declarations.

    Concrete subclasses exist only so ``isinstance`` can tell an agent
    attribute from an optimizer argument; the storage and ``[...]`` parsing
    are identical.
    """

    __slots__ = ("type", "bound", "default")

    def __init__(
        self,
        dtype: typing.Any = None,
        bound: typing.Any = None,
        default: typing.Any = None,
    ) -> None:
        self.type = dtype
        self.bound = bound
        self.default = default

    def __class_getitem__(cls, params: typing.Any) -> "Declaration":
        return cls(*parse_declaration_params(params))


def decorate_with_base(cls: typing.Any, base: typing.Any) -> typing.Any:
    """Return ``cls`` with ``base`` injected as an *additional* base.

    CPython refuses to reassign ``__bases__`` when the only base is ``object``
    ("deallocator differs from 'object'"), so the decorator returns a new class
    that inherits from both the user class and the base. Keeping the original
    class in the MRO preserves zero-arg ``super()`` calls such as
    ``super().__init__(**kwargs)`` in a ``@cy.legacy`` class.
    """
    if issubclass(cls, base):
        return cls

    namespace: dict[str, typing.Any] = {
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__": cls.__doc__,
        "__annotations__": {},
    }
    return type(cls.__name__, (cls, base), namespace)


def collect_declarations(cls: typing.Any, declaration_types: tuple[type, ...]) -> dict:
    """Collect declaration instances from a class and its bases.

    Both annotations (``name: cy.Argument[...]``) and class attributes (the form
    the compiled source is materialized into) are recognized.
    """
    declarations: dict[str, typing.Any] = {}

    for base in reversed(cls.__mro__):
        for name, declaration in (getattr(base, "__annotations__", {}) or {}).items():
            if isinstance(declaration, declaration_types):
                declarations[name] = declaration
        for name, value in vars(base).items():
            if isinstance(value, declaration_types):
                declarations[name] = value

    return declarations


def class_source_with_base(
    cls: typing.Any, base_name: str, declaration_types: tuple[type, ...]
) -> str:
    """Return the class source with ``base_name`` inserted into its header.

    ``... name: cy.Argument[...]`` annotations are also turned into ordinary
    assignments, because Cython compiles class-body annotations as type
    annotations and drops them, which would erase the declarations.
    """
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(cls))
    pattern = re.compile(rf"^class\s+{re.escape(cls.__name__)}\s*:", re.M)

    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"class {cls.__name__}({base_name}):"

    source = pattern.sub(replace, source, count=1)
    if count == 0:
        raise RuntimeError(f"Could not locate class {cls.__name__} in its own source.")

    declaration_names = "|".join(sorted(re.escape(t.__name__) for t in declaration_types))
    declarations = [
        name
        for name, value in (getattr(cls, "__annotations__", {}) or {}).items()
        if isinstance(value, declaration_types)
    ]
    for name in declarations:
        # Accept both ``cy.Argument[...]`` and a directly imported ``Argument[...]``.
        assignment = re.compile(
            rf"^(\s*){re.escape(name)}\s*:\s*((?:[A-Za-z_]\w*\.)?(?:{declaration_names})\b.*)$",
            re.M,
        )
        source = assignment.sub(rf"\g<1>{name} = \g<2>", source, count=1)

    return source


def maybe_compile(
    decorated: typing.Any,
    enabled: bool,
    *,
    source_cls: typing.Any = None,
    base_name: typing.Optional[str] = None,
    import_line: typing.Optional[str] = None,
    declaration_types: tuple[type, ...] = (),
) -> typing.Any:
    """Return the JIT-compiled class when ``enabled``, else ``decorated``.

    ``source_cls`` is the original user class (needed because ``decorated`` may
    be an injected subclass with no source of its own).
    """
    if not enabled:
        return decorated

    from clypto.precompile import compile_decorated, is_precompiling

    # While the compiled module imports, its decorator runs again; bail out
    # before touching inspect.getsource (unavailable for a compiled module).
    if is_precompiling():
        return decorated

    original = source_cls or decorated
    source = class_source_with_base(original, base_name, declaration_types) if base_name else None
    return compile_decorated(original, source=source, imports=[import_line] if import_line else None)
