#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Command-line interface for Clypto."""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import tempfile
from typing import Sequence

from setuptools import Distribution

from clypto.compiler import build_extension

__all__ = ["build_command", "new_command", "main"]

_ALGO_TEMPLATE = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np


cdef class {class_name}(cy.Optimizer):
    \"\"\"Custom optimization algorithm implemented in Cython.

    Subclasses Clypto's native optimizer core for high-performance execution.
    \"\"\"

    cdef public double factor

    def __init__(self, epoch=100, pop_size=30, factor=0.5, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")

    def initialize(self):
        \"\"\"Optional hook executed once before the main optimization loop.\"\"\"
        pass

    cdef void evolve(self, int epoch):
        \"\"\"Core algorithm logic executed at each iteration/epoch.

        Parameters
        ----------
        epoch : int
            The current epoch index (0-based).
        \"\"\"
        cdef cy.Agent agent
        for agent in self.population:
            # Update solution vector
            agent.solution = agent.solution * self.factor
            # Re-evaluate agent against the problem
            agent.evaluate(self.problem)
"""


def _derive_class_name(stem: str) -> str:
    parts = [p for p in re.split(r"[^a-zA-Z0-9]+", stem) if p]
    if not parts:
        return "MyOptimizer"
    if len(parts) == 1 and parts[0][0].isupper() and not parts[0].isupper():
        class_name = parts[0]
    else:
        class_name = "".join(p.capitalize() for p in parts)
    if class_name[0].isdigit():
        class_name = "Algo" + class_name
    return class_name


def new_command(filepath: str, force: bool = False) -> int:
    """Scaffold a new Cython optimizer algorithm template."""
    if not filepath.endswith(".pyx"):
        filepath = filepath + ".pyx"

    if os.path.exists(filepath) and not force:
        print(f"Error: File '{filepath}' already exists. Use -f/--force to overwrite.", file=sys.stderr)
        return 1

    stem = os.path.splitext(os.path.basename(filepath))[0]
    class_name = _derive_class_name(stem)
    content = _ALGO_TEMPLATE.format(class_name=class_name)

    parent_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(parent_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created template at '{filepath}' (class: {class_name})")
    return 0


def build_command(filepath: str, force: bool = False) -> int:
    """Compile a .pyx algorithm file in-place to a shared library (.so/.pyd)."""
    if not os.path.isfile(filepath):
        print(f"Error: Source file '{filepath}' does not exist.", file=sys.stderr)
        return 1

    abs_path = os.path.abspath(filepath)
    out_dir = os.path.dirname(abs_path)
    stem = os.path.splitext(os.path.basename(abs_path))[0]

    print(f"Compiling '{filepath}' in-place...")
    temp_build = tempfile.mkdtemp(prefix="clypto_cli_")
    try:
        exts = build_extension(abs_path, name=stem, force=force, quiet=True)
        dist = Distribution({"name": stem, "ext_modules": exts})
        cmd = dist.get_command_obj("build_ext")
        cmd.build_lib = out_dir
        cmd.build_temp = temp_build
        cmd.ensure_finalized()
        cmd.run()
        so_path = cmd.get_ext_fullpath(stem)
        if not os.path.isfile(so_path):
            candidates = [
                f for f in glob.glob(os.path.join(out_dir, f"{stem}.*"))
                if f.endswith((".so", ".pyd")) and not f.endswith(".pyx")
            ]
            if candidates:
                so_path = candidates[0]
        print(f"Successfully compiled: {so_path}")
        return 0
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temp_build, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for clypto command."""
    parser = argparse.ArgumentParser(
        prog="clypto",
        description="Clypto: High-performance Cython optimization framework CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    build_parser = subparsers.add_parser("build", help="Compile a .pyx file in-place to .so")
    build_parser.add_argument("file", help="Path to .pyx file")
    build_parser.add_argument("-f", "--force", action="store_true", help="Force rebuild")

    # new
    new_parser = subparsers.add_parser("new", help="Scaffold a new optimizer algorithm template")
    new_parser.add_argument("file", help="Path or filename of the .pyx file to create")
    new_parser.add_argument("-f", "--force", action="store_true", help="Overwrite if file exists")

    args = parser.parse_args(argv)

    if args.command == "build":
        return build_command(args.file, force=args.force)
    elif args.command == "new":
        return new_command(args.file, force=args.force)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
