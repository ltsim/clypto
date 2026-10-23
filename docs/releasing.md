# Releasing

clypto versions by cycle, not by feature: `v<year><edition><version>`. The `b1`
in `v2026b1` is the first patch of the `b` edition of 2026. Every version gets its
own branch, its own tag, and its own PyPI release.

## The cycle

1. **Cut the branch** from `master`, named after the version it will hold:

    ```bash
    git checkout master && git pull
    git checkout -b release/v2026b2
    ```

2. **Bump the version** in `clypto/__init__.py`. This is the first commit on the
   branch; `pre-commit` rejects anything else until it happens.

    ```python
    __version__ = "2026b2"
    ```

3. **Do the work** on the release branch. Each commit is gated by `ruff` and
   `mypy` over the files it stages, so the branch stays clean as it goes.

4. **Update the release metadata** in the same branch — see
   [Keeping the metadata in step](#keeping-the-metadata-in-step).

5. **Push and open a pull request into `master`**. `pre-push` runs the version
   check and then the full `uv run pytest tests/` suite in your `dev` environment;
   CI runs the same suite across Python 3.11–3.14 on Linux, Windows and macOS.

6. **Merge into `master`**. Two things fire on the merge: `test.yml` and
   `type.yml` have already validated the pull request, and `docs.yml` rebuilds and
   redeploys this site.

7. **Create the GitHub Release**, tagged `v2026b2`, from `master`. That tag is the
   release point: `publish.yml` triggers on the published release, builds wheels
   for Linux, Windows and macOS with cibuildwheel plus an sdist, and publishes
   them to PyPI.

    ```bash
    git tag v2026b2
    git push origin v2026b2
    # then: gh release create v2026b2 --generate-notes
    ```

8. **Start the next cycle** from the same commit: `release/v2026b3` for the next
   patch of the edition, or a new edition letter for the next batch of work.

## Version formats

| | Value |
| --- | --- |
| `clypto/__init__.py` and PyPI | `2026b2` |
| branch | `release/v2026b2` |
| tag and GitHub Release | `v2026b2` |

The `v` prefix belongs to the branch and the tag. The Python version is written
without it, so `pip install clypto==2026b2` matches what `clypto.__version__`
reports.

## Keeping the metadata in step

A release updates three version-carrying files. The hooks enforce the first; the
other two are yours to remember (or add to the hook if they keep slipping):

| File | What changes |
| --- | --- |
| `clypto/__init__.py` | `__version__` — checked by `pre-commit` and `pre-push` |
| `CITATION.cff` | `version`, and `date-released` if you keep that field |
| `CHANGELOG.md` | a new `# Version <version>` section on top; keep the per-edition sections below it |

`docs/reference/changelog.md` embeds `CHANGELOG.md` at build time, so the changelog
needs no separate docs edit.

## What the hooks enforce

`.husky/check-version.sh` is the machine-readable form of the cycle:

+ On a `release/v<version>` branch, the staged `__version__` must equal that
  branch's version. Drift is caught at commit time, and a bump commit is allowed
  through because the check reads the staged copy — the commit that fixes the
  version is the commit that unblocks itself.
+ A commit that changes `__version__` must be on the `release/v<version>` branch
  naming the version it sets. Bumping on `master` or a `development/*` branch is
  rejected.
+ Both the branch suffix and the version must look like the scheme: four digits,
  one lowercase edition letter, then digits.

Check it by hand against `HEAD` with `make check-version`, and read
`docs/installation.md` for what `pre-commit` and `pre-push` run beyond this.

## Bypassing

Nothing here is a trust mechanism — every gate is a convenience, and every gate
can be skipped:

```bash
HUSKY=0 git commit ...        # skip all hooks
git commit --no-verify        # skip pre-commit only
git push --no-verify          # skip pre-push only
```

CI runs the suite and `mypy` on every pull request into `master` regardless, so a
skipped hook delays the failure instead of avoiding it.
