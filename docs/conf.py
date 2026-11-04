"""Sphinx configuration for clypto documentation."""

import sys
from pathlib import Path

# Paths
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project information
project = "clypto"
copyright = "Maintained by LTSIM · MIT License"
author = "Thieu, LTSIM"

try:
    import clypto
    release = clypto.__version__
except Exception:
    release = "2026b1"

# General configuration
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

suppress_warnings = [
    "myst.header",
    "misc.highlighting_failure",
]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"

exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "scripts/**",
    "custom-optimizers/_snippets/**",
]

# HTML output options
html_theme = "nature"
html_title = "clypto documentation"
html_static_path = ["assets"]


def generate_docs(app):
    """Generate dynamic API and algorithm category documentation pages before build."""
    from docs.scripts.gen_pages import main
    main()


def setup(app):
    app.connect("builder-inited", generate_docs)
