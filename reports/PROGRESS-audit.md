# Progress control — clypto audit (`refactoring/audit-simplifications`)

**Controller:** `cy-audit-advisor-code` (`opencode2_23077bf4…`). Assumed on 2026-11-17 15:2x.
**Scope:** progress, history and code-change control for every workstream landing on this branch.
**Rule in force:** nothing is committed or pushed without the controller's GO; every change is gated by
`pytest tests/` + `ruff check clypto/ tests/` + `mypy clypto/` on a fresh build (0 stale `.so`).

## Current state

| Item | Value |
|---|---|
| PR | **https://github.com/ltsim/clypto/pull/28** — `refactoring/audit-simplifications` → `release/v2026b2` (23 commits, 64 files, +2264/−482) |
| HEAD | `b49a26be` — *chore(reports): add the audit plans, reports and progress control* |
| origin | `b49a26be` — pushed (pre-push hook: 449 passed) |
| Tracked tree | clean; only `reports/probe-final.py.bak` left untracked on purpose |
| Build | 269 `.pyx`, 0 stale or missing `.so` |
| Checks at `b49a26be` | pytest **449 passed** · ruff **clean** · mypy **clean** (189 files) |
| Golden | `c678e115` → `8c02c16b…` (byte-identical to `c235f3bd`; docstrings only) |

## History on this branch (19 commits over `release/v2026b2`)

| Block | Commits | State |
|---|---|---|
| Simplification / perf, phases 1–6 | `f4732b50`, `bc919a95`, `e146ee83`, `049d0d35`, `3e69ba7a`, `5ea34989`, `d0adefb0`, `2bce0578`, `0aa483a7`, `94d1aa6a`, `f746a501` | accepted, pushed earlier |
| Bounds overflow fix | `4f2c0832` | accepted, pushed |
| RNG/NFE: scipy `rvs` from the optimizer stream | `7654d659` | accepted; **invalidates old seeded baselines** of JADE/OriginalSHADE/L_SHADE/LSHADEcnEpSin |
| Memory audit: `<Agent>` cast guards | `67c51eca` | accepted, pushed; 9 subprocess regression tests |
| `g_best` retention: option (c), documented | `afb6c2a4` → **corrected in `c678e115`** | Phase 0 proved the native return is a snapshot copy; the real (bounded) retention is the decorator's `_evaluator`. Docstrings corrected, pushed |
| RNG/NFE fixes (run `run_2d493908d714`) | `f7cc3595`, `df53341a`, `11cf68af`, `c235f3bd` | verified by controller, **pending push** |

## The four new commits (independent verification by the controller)

| Sha | Change | Tests | Gate |
|---|---|---|---|
| `f7cc3595` | `OriginalPSS` LHS sampler takes `rng=self.generator`; the `n_objs` probe saves/restores the bounds `bit_generator` state so it consumes no draw | `tests/test_rng_reproducibility.py` (2, red before) | ✅ |
| `df53341a` | SHADE family runs with `pop_size < 10`: `OriginalSHADE` clamps `2/NP` to the `0.2` cap; `OriginalLSHADEcnEpSin` clamps `NP_min` to `NP_init`; `L_SHADE` untouched | `tests/test_shade_small_population.py` (6) | ✅ |
| `11cf68af` | `Problem.coerce` keeps a seeded Problem's seed when `solve(seed=None)`; engines create their generators in `check_problem` from the resolved seed; the `n_objs` probe counts in `n_evals` | `tests/test_problem_seed_kept.py`, `tests/test_problem_n_evals.py`, updates to 2 older tests | ✅ |
| `c235f3bd` | `Population.argsort` and `argsort_agents` use `kind="stable"` so ties keep population order (matches mealpy's `sorted`) | `test_argsort_ties_keep_the_population_order` (red before) | ✅ |

Evidence trail: matrix fitness/solutions byte-identical from `f7cc3595` to `c235f3bd`
(`n_evals` 1550 → 1551 by design), 13/13 contract identical, nine module shas cross-checked,
`CHANGELOG.md` `[Unreleased]` carries the **BREAKING-SEEDED** note and the `max_fe` offset.

Caveat recorded: the repo's pre-commit hook lints staged `*.py` only, so `.pyx` correctness rests
on the rebuilt extension + the suite; the controller re-ran suite, ruff and mypy over the whole tree.

## Open risks (accepted unless the user says otherwise)

| # | Risk | Status |
|---|---|---|
| R1 | `Problem.coerce` with `seed=None` keeps (and rewrites) the Problem's seed; an unseeded Problem stays non-reproducible | intended, documented |
| R2 | `n_evals` +1 on a fresh Problem; exact `max_fe` budgets stop one epoch earlier; reusing one Problem gives 211 → 210 | intended, documented |
| R3 | Stable `argsort` changes tie-breaking: unit-tested, **not** covered by the trajectory golden (the matrix has no ties) | **closed as intended**: controller reproduced the flip on the test vector — n=20, fitness `[2,1,0,0,2,1,1,1,2,1,0,2,0,0,2,1,0,0,1,2]` → default `argsort` picks agent 3, stable picks agent 2 (the agent `sort()[0]`/`g_best` would take). `Population.best` is unaffected (strict comparison keeps the first) |
| R4 | SHADE `NP < 10` has no numeric reference, only "no exception + in bounds" | accepted |
| R5 | Memory (RSS/Valgrind) not re-measured after the RNG/ranking patches | **CLOSED — PASS**: RSS +0.000…+0.005 MB/100 iters (6/6 plateaus), Valgrind 40,705 B/168 blocks in 4/4 with **0 frames of our build**, refcounts flat. Logs `post_rng_*` |
| R6 | `solve()` retention: **Phase 0 corrected the record** — the native `solve()` returns a `snapshot_agent` copy that pins nothing; the decorator's returned agent carries `_evaluator` and pins the optimizer + population (+3 refs) until dropped, bounded | decision (c) stands, but the `afb6c2a4` text is factually wrong and is being replaced (docstring-only commit) |
| R7 | `7654d659` changes seeded trajectories of 4 algorithms | **decided: keep** (determinism fix, `BREAKING-SEEDED` documented) |

## Ownership / protocol

| Who | Owns |
|---|---|
| `cy-audit-advisor-code` (controller) | final verification, commits, push, this document |
| `cy-fix-report` (run `run_2d493908d714`) | may settle its own task board with the controller's GO; no commits/push |
| `cy-mem-verifier` | read-only verification and golden captures; no git writes |
| implementation subagents | scoped worktrees/`/tmp`, report diffs; no git writes |

## Pending

1. ~~Push `f7cc3595..c235f3bd`~~ — **done**, and continued: `origin` at `b49a26be` (23 commits,
   pre-push hook 449 passed each time).
2. ~~Settle the run's board~~ — done by `cy-fix-report` on the controller's GO.
3. ~~R5 memory re-measure~~ — **PASS**: RSS 6/6 plateaus (+0.000…+0.005 MB/100 iters); Valgrind
   4/4 at 40,705 B / 168 blocks with **0 frames of our build** (all scipy/numpy import
   registration); refcounts flat. Caveat recorded: the current probe imports the collection
   packages, so `ga`/`de`/`churn` carry the scipy bucket too — same bytes, not a regression.
4. ~~R7~~ — **decisión del usuario: dejar `7654d659`**.
5. ~~Re-pin the golden on `c678e115`~~ — done: same sha256 `8c02c16b…`, contract 13/13 identical.
6. ~~Doc sync + pending commits~~ — `f4831363` (CHANGELOG/tutorial/issues), `cf1af222`
   (memory-leak probes, ruff+mypy clean), `b49a26be` (reports). `reports/probe-final.py.bak`
   deliberately left untracked (backup of the probe).
7. **Open:** PR #28 → `release/v2026b2`. Note: `test.yml`/`type.yml` trigger only on PRs to
   `master`, so this PR reports no checks; local gate (449 / ruff / mypy) is the evidence.

## Evidence index

- `/tmp/opencode/mem_verify/` — goldens (`baseline_c235f3bd.json` + `.provenance.txt`), protocol,
  RNG functional checks, Valgrind/ASan logs, fast-path scripts.
- `/tmp/opencode/rng_review/` — audit report, per-milestone diffs (`argsort/`, `problemfix/`, `shade/`),
  red/green runs, `suite-final.log`.
- `/tmp/opencode/reports/` — memory-audit logs (RSS, refcounts).
- `reports/memory-audit-baseline.md` — memory verdict (no leaks; cast guards fixed).
- `reports/PLAN-fix-memory.md` — Phase 0/1/2 plan for the `g_best` retention question.
