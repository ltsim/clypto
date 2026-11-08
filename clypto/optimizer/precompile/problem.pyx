import inspect
import typing

import numpy as np

from clypto.optimizer.bounds import Bounds
from clypto.optimizer.native.problem import Problem
from clypto.optimizer.precompile.decoration import decorate_with_base, maybe_compile
from clypto.optimizer.precompile.compiler import compile_decorated, is_precompiling

__all__ = ["problem"]


def problem(
    target=None,
    *,
    bounds=None,
    sense="min",
    vectorized=False,
    vectorize=None,
    decode=False,
    compile=True,
    name=None,
    evaluator=None,
    obj_weights=None,
    seed=None,
    **kwargs,
):
    """Define and optionally JIT-compile an optimization problem.

    Usable as a decorator on classes or functions, or as a direct factory:

        @cy.problem(bounds=..., sense="min", compile=True)
        def sphere(x):
            return np.sum(x**2)

        @cy.problem(compile=True, decode=True)
        class MyProblem:
            bounds = ...
            def evaluate(self, solution):
                ...

        p = cy.problem(sphere, bounds=...)
    """
    if vectorize is not None:
        vectorized = vectorize

    def decorate(user_target):
        if inspect.isclass(user_target):
            cls_bounds = getattr(user_target, "bounds", bounds)
            cls_sense = getattr(user_target, "sense", sense)
            cls_decode = getattr(user_target, "decode", decode)
            cls_vectorized = getattr(user_target, "vectorize", getattr(user_target, "vectorized", vectorized))

            coerced_bounds = None
            if cls_bounds is not None:
                coerced_bounds = cls_bounds if isinstance(cls_bounds, Bounds) else Bounds(cls_bounds)

            decorated = decorate_with_base(user_target, Problem)
            if coerced_bounds is not None:
                decorated.bounds = coerced_bounds
            decorated.sense = cls_sense
            decorated.decode = bool(cls_decode)
            decorated.vectorized = bool(cls_vectorized)

            res_cls = maybe_compile(
                decorated,
                compile,
                source_cls=user_target,
                base_name="Problem",
                import_line="from clypto.optimizer.native.problem import Problem",
            )

            if coerced_bounds is not None:
                res_cls.bounds = coerced_bounds
            res_cls.sense = cls_sense
            res_cls.decode = bool(cls_decode)
            res_cls.vectorized = bool(cls_vectorized)

            orig_init = getattr(user_target, "__init__", None)

            def init_wrapper(self, *args, **call_kwargs):
                init_bounds = call_kwargs.pop("bounds", coerced_bounds)
                if init_bounds is None and hasattr(type(self), "bounds"):
                    init_bounds = getattr(type(self), "bounds")
                init_sense = call_kwargs.pop("sense", cls_sense)
                init_vectorized = call_kwargs.pop("vectorize", call_kwargs.pop("vectorized", cls_vectorized))
                init_decode = call_kwargs.pop("decode", cls_decode)
                Problem.__init__(
                    self,
                    bounds=init_bounds,
                    sense=init_sense,
                    vectorized=init_vectorized,
                    decode=init_decode,
                    name=call_kwargs.pop("name", getattr(res_cls, "__name__", "Problem")),
                    evaluator=call_kwargs.pop("evaluator", evaluator),
                    obj_weights=call_kwargs.pop("obj_weights", obj_weights),
                    seed=call_kwargs.pop("seed", seed),
                )
                if orig_init is not None and orig_init is not object.__init__:
                    orig_init(self, *args, **call_kwargs)

            res_cls.__init__ = init_wrapper
            return res_cls

        elif callable(user_target):
            fn = user_target
            if compile and not is_precompiling():
                try:
                    fn = compile_decorated(user_target)
                except Exception:
                    fn = user_target

            if bounds is not None:
                return Problem(
                    bounds=bounds,
                    sense=sense,
                    obj_func=fn,
                    name=name or getattr(user_target, "__name__", "Problem"),
                    evaluator=evaluator,
                    obj_weights=obj_weights,
                    vectorized=vectorized,
                    seed=seed,
                    decode=decode,
                    **kwargs,
                )

            def wrapper(*args, **call_kwargs):
                has_bounds = "bounds" in call_kwargs or (
                    len(args) > 0 and not isinstance(args[0], (list, tuple, np.ndarray)) and hasattr(args[0], "low")
                )
                if has_bounds:
                    call_bounds = call_kwargs.pop("bounds", args[0] if args else None)
                    call_sense = call_kwargs.pop("sense", sense)
                    call_vectorized = call_kwargs.pop("vectorize", call_kwargs.pop("vectorized", vectorized))
                    call_decode = call_kwargs.pop("decode", decode)
                    return Problem(
                        bounds=call_bounds,
                        sense=call_sense,
                        obj_func=fn,
                        name=name or getattr(user_target, "__name__", "Problem"),
                        evaluator=evaluator,
                        obj_weights=obj_weights,
                        vectorized=call_vectorized,
                        seed=seed,
                        decode=call_decode,
                        **call_kwargs,
                    )
                return fn(*args, **call_kwargs)

            wrapper.__name__ = getattr(user_target, "__name__", "wrapper")
            wrapper.compiled_func = fn
            return wrapper

        else:
            raise TypeError(f"@cy.problem expects a class or callable, got {type(user_target).__name__}")

    return decorate(target) if target is not None else decorate
