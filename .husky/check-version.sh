#!/bin/sh
# Keep the release branch and the package version in lockstep.
#
# The cycle says every version lives on release/v<year><edition><version>, so:
#
#   1. On a release branch, clypto/__init__.py must carry that branch's version.
#   2. A commit that changes the version must do so on its own release branch.
#
# The staged copy is what gets checked (git show :path), not the working tree, so
# the commit that bumps the version is the one that unblocks itself -- cutting
# release/v2026b2 and bumping in the same commit is fine.
#
# Usage: sh .husky/check-version.sh [staged|head]
#   staged  pre-commit: the version as it would be committed (default)
#   head    pre-push:   the version as committed at HEAD

mode="${1:-staged}"
file="clypto/__init__.py"
# __version__ carries no "v" prefix ("2026b1"); the branch and the tag do.
pattern='s/^__version__ = "\([^"]*\)"/\1/p'

die() {
    printf '%s\n' "version-check: $1" >&2
    printf '%s\n' "  $2" >&2
    exit 1
}

# <year><edition letter><patch digits>, e.g. 2026b1 or 2026b12.
is_version() {
    case "$1" in
        [0-9][0-9][0-9][0-9][a-z]*)
            case "${1#?????}" in
                '' | *[!0-9]*) return 1 ;;
                *) return 0 ;;
            esac
            ;;
    esac
    return 1
}

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
[ -n "$branch" ] || exit 0 # no commits yet

if [ "$mode" = "head" ]; then
    version=$(git show "HEAD:$file" 2>/dev/null | sed -n "$pattern")
else
    version=$(git show ":$file" 2>/dev/null | sed -n "$pattern")
fi
# File not in the index (initial commit) or no __version__ line: nothing to say.
[ -n "$version" ] || exit 0

committed=$(git show "HEAD:$file" 2>/dev/null | sed -n "$pattern")

case "$branch" in
    release/v*) want=${branch#release/v} ;;
    *) want='' ;;
esac

if [ -n "$want" ]; then
    if ! is_version "$want"; then
        die "'$branch' is not a release branch name." \
            "expected release/v<year><edition><version>, e.g. release/v2026b1"
    fi
    if [ "$version" != "$want" ]; then
        if [ "$committed" = "$want" ]; then
            # The branch was correct and this commit moves away from it.
            die "branch $branch is pinned to __version__ = \"$want\", found \"$version\"." \
                "a different version belongs on its own release/v<version> branch"
        fi
        die "branch $branch expects __version__ = \"$want\", found \"$version\"." \
            "bump clypto/__init__.py to \"$want\" first -- every version lives on its own release branch"
    fi
fi

# A commit that changes the version sets the release it belongs to.
if [ -n "$committed" ] && [ "$committed" != "$version" ]; then
    if ! is_version "$version"; then
        die "\"$version\" is not a version." \
            "expected <year><edition><version>, e.g. 2026b1"
    fi
    if [ "$branch" != "release/v$version" ]; then
        die "bumping to \"$version\" (from \"$committed\") belongs on release/v$version, not $branch." \
            "cut the release branch and bump there; merge it into master to release"
    fi
fi

exit 0
