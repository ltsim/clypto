# Plan — fix memory (g_best / generation retention)

**Status:** Phase 0 open. **Owner:** `cy-audit-advisor-code` supervises; GLM-5.3-flash session executes.
**Branch:** `refactoring/audit-simplifications` (HEAD `afb6c2a4`).
**Source:** `reports/memory-audit-baseline.md` (finding 2) — retention reported, not an RSS leak.

## Objective

Decide and execute the smallest correct resolution for the per-solve retention the memory audit
reported: `solve()` returns `self.g_best`, and in the PSO family `PSOAgents.owner` is a strong
reference to the optimizer, so a caller that keeps the returned agent may pin a whole
population generation until the next `solve`/`gc.collect()`.

## Phase 0 — verify the claim (no code changes) — **DONE 2026-11-17**

Result: `/tmp/opencode/fixmem/PHASE0-REPORT.md` (scripts `phase0_*.py`). Verdict: the finding is
**wrong for the native engine and right for the decorator API**.

- Native (`LegacyOptimizer`): `g_best` is written only as `snapshot_agent(ranked[0])` in
  `after_initialization`/`after_evolve`, so `solve()` returns an independent copy — measured
  `best is population[i]` **False**, no shared solution memory, `Δrc(problem) = 0` for
  `OriginalPSO`, `AIW_PSO`, `P_PSO`, `OriginalGA`, `OriginalGTO` (12 seeds), and the
  `{optimizer → population → factory.owner → optimizer}` cycle is collected while a live member
  is held (Populations=0, PSOAgents=0, optimizer=0).
- Decorator (`DecoratedOptimizer`): the returned snapshot carries `_evaluator`, a bound method
  of the optimizer, so keeping it pins the optimizer + population: `+3` refs on `problem`, one
  Population, one optimizer per kept agent, linear and fully bounded (collapses when dropped).

**Gate outcome:** option (b) (weakref `PSOAgents.owner`) is dead — there is nothing to fix.
The accurate fix is option (c) with corrected text, plus a decorator note; a weakref on
`_evaluator` would change re-evaluation semantics (RuntimeAgent's solution setter needs it).

## Phase 1 — choice (settled)

| Option | Change | Status |
|---|---|---|
| (c) document | corrected `solve()` docstrings on both engines + AGENTS note | **chosen** (pending commit) |
| (b) weakref owner | `PSOAgents.owner` as weakref | rejected: no retention exists in the native path |
| (b′) weakref `_evaluator` | decorator agent loses its re-evaluation path | rejected: changes semantics for a bounded, caller-controlled retention |
| (a) snapshot g_best | `solve()` returns a copy | already the native behaviour |

## Phase 2 — implement (b), if chosen — **CANCELLED**

Phase 0 shows there is no strong `owner` retention in the native path, so Phase 2 is cancelled
as designed. What ships instead is the (c) correction:

1. `clypto/optimizer/native/optimizer.pyx` — `solve()` docstring: returns a `snapshot_agent`
   copy that pins nothing.
2. `clypto/optimizer/precompile/base.pyx` — `solve()` docstring: the decorator's returned agent
   carries `_evaluator`, so keeping it pins the optimizer + population (+3 refs on `problem`)
   until dropped; bounded.
3. AGENTS.md bullet corrected (local, gitignored).
4. Gate: rebuild, `pytest tests/` (449), ruff, mypy; no behaviour change, so the golden
   fitness/solutions must stay identical and the golden file only needs re-pinning because the
   module shas change.

## Acceptance

- [x] Phase 0 verdict documented with the exact chain.
- [x] Option (b) ruled out on measurement, not opinion.
- [ ] Corrected text committed (Phase 2 above).
- [x] Finding 2 closed in `reports/memory-audit-baseline.md`.

## Risks / rollback

- Documentation only: no runtime risk. Rollback is a text revert.
- The wrong `afb6c2a4` claim is the cautionary tale: a plausible ownership story is not a
  measurement. Any future retention claim needs `gc.get_referrers` or a counter, not reasoning.
