# Makefile for Sphinx documentation local development
#

SPHINXOPTS    ?= -W
SPHINXBUILD   ?= uv run --no-sync sphinx-build
SPHINXAUTO    ?= uv run --no-sync sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = build
PORT          ?= 8000

.PHONY: help sync html dirhtml linkcheck clean serve livehtml open

help:
	@echo "Available commands for local documentation:"
	@echo "  make html        - Build HTML documentation"
	@echo "  make serve       - Live-reload dev server with sphinx-autobuild (port $(PORT))"
	@echo "  make linkcheck   - Check external links integrity"
	@echo "  make clean       - Remove built documentation artifacts"
	@echo "  make open        - Open built documentation in default browser"
	@echo "  make sync        - Sync docs dependencies via uv"

sync:
	uv sync --extra docs --no-install-project

html: sync
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

serve livehtml: sync
	@$(SPHINXAUTO) "$(SOURCEDIR)" "$(BUILDDIR)/html" --port $(PORT)

linkcheck: sync
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

open:
	@command -v xdg-open >/dev/null && xdg-open "$(BUILDDIR)/html/index.html" || \
	 command -v open >/dev/null && open "$(BUILDDIR)/html/index.html" || \
	 echo "Open file://$(CURDIR)/$(BUILDDIR)/html/index.html in your browser"

clean:
	rm -rf "$(BUILDDIR)"
	rm -rf api/
	find categories/ -name "*.md" ! -name "index.md" -exec rm -f {} +

%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
