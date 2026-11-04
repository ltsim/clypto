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

## Git hooks

[Husky](https://typicode.github.io/husky/) runs the checks CI runs anyway, before
your code reaches a pull request. Hooks live in `.husky/` and need one setup per
clone (Node.js 18 or newer):

```bash
uv sync --extra dev           # the hooks call into this environment
npm install                   # activates .husky/pre-commit and .husky/pre-push
```

| Hook | Runs |
| --- | --- |
| `pre-commit` | the release version check, then `ruff` and `mypy` over the Python files the commit stages |
| `pre-push` | the release version check, then the full `pytest tests/` suite |

Both hooks scope to what you changed, not to the whole repository: the tree still
carries pre-existing lint and typing debt, so `pre-commit` checks only staged
files (`mypy` runs with `--follow-imports=silent` so it reports the files you
touched, not the modules they import). A file is expected to be clean from the
moment you edit it. `make lint` and `make typecheck` run the tools over the
repository when you want the full picture, and `make check-version` runs the
version check against `HEAD`.

Skip a hook when you need to (`HUSKY=0 git commit`, or `git commit --no-verify`).
`pre-push` needs the compiled package *and* the `dev` extra — a plain `uv run`
syncs only the runtime dependencies, so `pytest` will be missing. Build the
environment with `uv sync --extra dev` before pushing. Once it is built the suite
itself takes about a minute; the slow part is the first build, which compiles every
Cython extension. See [Releasing](releasing.md) for the version check the hooks
enforce.

## Just-in-time compilation (`compile` extra)

The `@cy.optimizer` and `@cy.agent` decorators can compile a
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
