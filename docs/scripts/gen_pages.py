"""Generate the algorithm-category and core-API reference pages at build time.

This is a ``mkdocs-gen-files`` hook. It reads the clypto sources with the
standard library ``ast`` module only, so no Cython compilation or import of the
package is required. Every page written here is virtual: nothing is added to the
working tree, and ``mkdocs serve`` / ``mkdocs build`` regenerate it on the fly.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from mkdocs_gen_files import open as gen_open

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_ROOT = REPO_ROOT / "clypto" / "collection"
REPO_URL = "https://github.com/ltsim/clypto/blob/master"


def _pythonize_pyx(text: str) -> str:
    """Rewrite native Cython constructs into plain Python for ``ast`` parsing.

    The collection is native ``.pyx`` (``cdef class`` / ``cimport`` / ``cpdef``),
    which the stdlib ``ast`` cannot read. Docstrings and ``__init__`` signatures
    are unaffected, so the rest of this script still works on the parsed tree.
    """
    out: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*(from\s+\S+\s+cimport\s+.+|cimport\s+.+)$", line):
            out.append("pass")
        elif re.match(r"^\s*cdef class ", line):
            out.append(re.sub(r"^(\s*)cdef class ", r"\1class ", line))
        elif re.match(r"^\s*cdef\s+", line):
            out.append(re.sub(r"^(\s*)cdef\s+.*$", r"\1pass", line))
        elif re.match(r"^\s*cpdef\s+", line):
            out.append(re.sub(r"^(\s*)cpdef\s+(?:\w+\s+)?(\w+\s*\()", r"\1def \2", line))
        else:
            out.append(line)
    return "\n".join(out)


def _parse(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".pyx":
        text = _pythonize_pyx(text)
    return ast.parse(text)

CATEGORY_LABELS = {
    "bio_based": "Bio-based",
    "evolutionary_based": "Evolutionary-based",
    "game_based": "Game-based",
    "human_based": "Human-based",
    "math_based": "Math-based",
    "music_based": "Music-based",
    "physics_based": "Physics-based",
    "sota_based": "State-of-the-art (SOTA)",
    "swarm_based": "Swarm-based",
    "system_based": "System-based",
}

CATEGORY_ORDER = list(CATEGORY_LABELS)

CORE_MODULES = [
    "clypto/optimizer/base.py",
    "clypto/optimizer/legacy.py",
    "clypto/optimizer/precompile/declaration.py",
    "clypto/optimizer/precompile/base.py",
    "clypto/optimizer/precompile/decorator.py",
    "clypto/optimizer/agents/declaration.py",
    "clypto/optimizer/agents/runtime.py",
    "clypto/optimizer/agents/decorator.py",
    "clypto/optimizer/precompile/compiler.py",
    "clypto/optimizer/precompile/decoration.py",
    "clypto/optimizer/problem.py",
    "clypto/optimizer/termination.py",
    "clypto/optimizer/validator.py",
    "clypto/optimizer/target.py",
    "clypto/optimizer/population.py",
    "clypto/optimizer/space/base.py",
    "clypto/optimizer/space/floats.py",
    "clypto/optimizer/space/integers.py",
    "clypto/optimizer/space/strings.py",
    "clypto/optimizer/space/binary.py",
    "clypto/optimizer/space/boolean.py",
    "clypto/optimizer/space/categorical.py",
    "clypto/optimizer/space/sequence.py",
    "clypto/optimizer/space/permutation.py",
    "clypto/hints/array.py",
    "clypto/hints/primitives.py",
    "clypto/hints/sense.py",
]


def _clean_doc(doc: str | None) -> list[str]:
    if not doc:
        return []
    lines = textwrap.dedent(doc).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [line.rstrip() for line in lines]


def _first_paragraph(lines: list[str]) -> str:
    paragraph: list[str] = []
    for line in lines:
        if not line.strip():
            break
        paragraph.append(line.strip())
    return " ".join(paragraph)


def _extract_hyperparameters(lines: list[str]) -> list[str]:
    params: list[str] = []
    capturing = False
    for line in lines:
        if "Hyper-parameters" in line:
            capturing = True
            continue
        if capturing:
            if not line.strip():
                break
            if line.strip().startswith("+"):
                params.append(line.strip()[1:].strip())
    return params


def _extract_references(lines: list[str]) -> str:
    try:
        idx = next(i for i, line in enumerate(lines) if line.strip().startswith("References"))
    except StopIteration:
        return ""
    refs = " ".join(line.strip() for line in lines[idx + 1 :] if line.strip())
    refs = refs.replace("~~~~~~~~~~", "").strip()
    return refs


def _format_signature(name: str, node: ast.ClassDef | ast.FunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    for arg in args.posonlyargs + args.args:
        parts.append(arg.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for arg in args.kwonlyargs:
        parts.append(arg.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return f"{name}({', '.join(parts)})"


def _doc_markdown(lines: list[str]) -> list[str]:
    """Render a docstring as Markdown without breaking the page.

    Docstrings use ``~~~~~`` separator lines which Python-Markdown treats as
    code fences, and indented ``Args:`` blocks which render as code. The summary
    paragraph is kept as prose and the remainder is fenced as plain text.
    """
    cleaned = [line for line in lines if not re.fullmatch(r"\s*~{3,}\s*", line)]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    if not cleaned:
        return []

    try:
        split = next(i for i, line in enumerate(cleaned) if not line.strip())
    except StopIteration:
        return [" ".join(line.strip() for line in cleaned), ""]

    summary = " ".join(line.strip() for line in cleaned[:split] if line.strip())
    rest = cleaned[split + 1 :]
    output: list[str] = []
    if summary:
        output += [summary, ""]
    if rest:
        output += ["```text", *rest, "```", ""]
    return output


def _format_class_signature(node: ast.ClassDef) -> str:
    init = next(
        (
            member
            for member in node.body
            if isinstance(member, ast.FunctionDef) and member.name == "__init__"
        ),
        None,
    )
    if init is None:
        return node.name
    return _format_signature(node.name, init)


def _source_link(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return f"{REPO_URL}/{rel}"


def _collect_classes(path: Path) -> list[dict]:
    tree = _parse(path)
    classes: list[dict] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        lines = _clean_doc(ast.get_docstring(node, clean=False))
        classes.append(
            {
                "name": node.name,
                "summary": _first_paragraph(lines),
                "params": _extract_hyperparameters(lines),
                "references": _extract_references(lines),
            }
        )
    return classes


def _generate_category(category: str) -> tuple[int, int]:
    label = CATEGORY_LABELS[category]
    directory = COLLECTION_ROOT / category
    modules = sorted(p for p in directory.glob("*.pyx") if p.name != "__init__.py")

    catalog: list[tuple[str, dict]] = []
    for module in modules:
        for cls in _collect_classes(module):
            catalog.append((module.stem, cls))

    lines: list[str] = [
        f"# {label} algorithms",
        "",
        f"`clypto.collection.{category}` ships **{len(catalog)} optimizer "
        f"{'class' if len(catalog) == 1 else 'classes'}** across "
        f"**{len(modules)} {'module' if len(modules) == 1 else 'modules'}**.",
        "",
        "| Module | Class | Summary | Source |",
        "| --- | --- | --- | --- |",
    ]
    for module_name, cls in catalog:
        link = _source_link(COLLECTION_ROOT / category / f"{module_name}.pyx")
        summary = cls["summary"].replace("|", "\\|") or "-"
        lines.append(f"| [{module_name}]({link}) | `{cls['name']}` | {summary} | [source]({link}) |")

    lines += ["", "## Modules", ""]
    for module in modules:
        classes = _collect_classes(module)
        if not classes:
            continue
        lines.append(f"### {module.stem}")
        lines.append("")
        for cls in classes:
            lines.append(f"#### `{cls['name']}`")
            lines.append("")
            if cls["summary"]:
                lines.append(cls["summary"])
                lines.append("")
            if cls["params"]:
                lines.append("**Hyper-parameters**")
                lines.append("")
                for param in cls["params"]:
                    lines.append(f"- {param}")
                lines.append("")
            if cls["references"]:
                lines.append(f"**Reference:** {cls['references']}")
                lines.append("")

    with gen_open(f"categories/{category}.md", "w") as fd:
        fd.write("\n".join(lines) + "\n")
    return len(catalog), len(modules)


def _generate_api() -> None:
    sections: list[str] = [
        "# Core API",
        "",
        "Signatures and docstrings for the modules most users touch. Generated "
        "directly from the sources at build time.",
        "",
    ]
    for relative in CORE_MODULES:
        path = REPO_ROOT / relative
        if not path.exists():
            continue
        tree = _parse(path)
        module_doc = _first_paragraph(_clean_doc(ast.get_docstring(tree)))
        title = relative.removesuffix(".pyx").removesuffix(".py").removesuffix("/__init__")
        sections += [f"## {title}", ""]
        if module_doc:
            sections += [module_doc, ""]
        sections += [f"Source: [`{relative}`]({_source_link(path)})", ""]

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = _clean_doc(ast.get_docstring(node, clean=False))
                sections += [f"### `{node.name}`", ""]
                sections += ["```python", f"class {_format_class_signature(node)}", "```", ""]
                sections += _doc_markdown(doc)
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if member.name.startswith("_") and member.name not in ("__init__",):
                            continue
                        mdoc = _clean_doc(ast.get_docstring(member, clean=False))
                        sections += [f"#### `{_format_signature(member.name, member)}`", ""]
                        sections += _doc_markdown(mdoc)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = _clean_doc(ast.get_docstring(node, clean=False))
                sections += [f"### `{_format_signature(node.name, node)}`", ""]
                sections += _doc_markdown(doc)

    with gen_open("api/core.md", "w") as fd:
        fd.write("\n".join(sections) + "\n")


def main() -> None:
    for category in CATEGORY_ORDER:
        if (COLLECTION_ROOT / category).is_dir():
            _generate_category(category)
    _generate_api()


main()
