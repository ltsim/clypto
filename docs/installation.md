# Installation

clypto supports CPython 3.11–3.14 and depends on `numpy`, `scipy`, and `zarr` at
runtime.

## Precompiled wheels (no compiler needed)

Wheels are precompiled per platform and Python version — Linux, macOS (Intel +
Apple Silicon), and Windows — so a plain `pip install` is all you need:

```bash
pip install clypto --upgrade
```

## Compiling from source

Building from source compiles the entire package to native code with Cython, so
a C compiler is required. The build dependencies (`Cython`, `numpy`,
`setuptools`) are fetched automatically by PEP 517 build isolation.

=== "From the repository"

    ```bash
    git clone https://github.com/ltsim/clypto.git
    cd clypto
    python -m pip install .
    ```

=== "Editable dev install (uv)"

    [uv](https://docs.astral.sh/uv/) compiles the extensions in place and links
    them into the project:

    ```bash
    git clone https://github.com/ltsim/clypto.git
    cd clypto
    uv sync --extra dev          # or: make uv-sync
    ```

=== "Latest from GitHub"

    ```bash
    pip install git+https://github.com/ltsim/clypto.git
    ```

The [Makefile](https://github.com/ltsim/clypto/blob/master/Makefile) wraps the uv
workflow (`uv-lock`, `uv-sync`, `uv-test`) and offers `make clean` to wipe build,
Cython, and bytecode artifacts.

## Running the tests

```bash
uv sync --extra dev
uv run pytest tests/          # or: make uv-test
```

## Just-in-time compilation (`compile` extra)

The `@cy.optimizer`, `@cy.agent` and `@cy.legacy` decorators can compile a
user-defined class on import via `pyximport`, which needs Cython and
`setuptools`. Install the `compile` extra:

```bash
pip install "clypto[compile]"
```

With uv, add it alongside the dev extra:

```bash
uv sync --extra dev --extra compile
```

## Building these docs

The documentation is built with MkDocs + Material. The docs toolchain does not
need the compiled package (`--no-install-project` skips it), so install and serve
with:

```bash
uv sync --extra docs --no-install-project
uv run --no-sync mkdocs serve
```

!!! note "Compile before benchmarking"

    A fresh clone ships only its `.py` sources. Compile first
    (`uv sync --extra dev` or `pip install .`), otherwise clypto runs as
    un-compiled Python and its compiled-by-design performance is lost.
