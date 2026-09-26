"""Batch helpers shared by the vectorized collection (``from clypto.optimizer.native import ops``)."""
import numpy as np

from clypto.optimizer.native.agent cimport LegacyNativeAgent
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


def accept(LegacyNativeOptimizer opt, NativePopulation cand, Py_ssize_t start=0, Py_ssize_t stop=-1,
           NativePopulation dst=None, bint old_first=False):
    """Row-wise survivor selection of ``cand`` into ``dst`` (default ``opt.pop``) over rows ``[start, stop)``.

    Sequential mode (``mode=None``) reproduces ``get_better_agent(new, old)``: min ``new < old``,
    max ``not (new < old)``. ``swarm``/``parallel`` modes reproduce ``greedy_selection_population``:
    min ``new < old``, max ``new > old``. Whole rows (fitness, objectives, position, fields) are copied.
    ``old_first`` is ``get_better_agent(old, new)``: min ``not (old < new)``, max ``old < new``.
    """
    cdef NativePopulation pop = opt.pop if dst is None else dst
    if stop < 0:
        stop = pop.n
    new = cand.buf[start:stop, cand.cF]
    old = pop.buf[start:stop, pop.cF]
    minimize = opt.problem.minmax == "min"
    if opt.mode in opt.AVAILABLE_MODES:
        better = new < old if minimize else new > old
    elif old_first:
        better = ~(old < new) if minimize else old < new
    else:
        better = new < old if minimize else ~(new < old)
    rows = start + np.flatnonzero(better)
    pop.buf[rows] = cand.buf[rows]


cdef void _write(NativePopulation pop, Py_ssize_t i, object x, NativeTarget target):
    cdef double[:, ::1] v = pop.view
    cdef double[::1] xs = np.ascontiguousarray(x, dtype=np.float64)
    cdef double[::1] os
    cdef Py_ssize_t j
    for j in range(pop.d):
        v[i, pop.cX + j] = xs[j]
    os = np.ascontiguousarray(np.asarray(target.objectives, dtype=np.float64).reshape(-1))
    for j in range(pop.m):
        v[i, pop.cO + j] = os[j]
    v[i, pop.cF] = target.fitness


def set_row(NativePopulation pop, Py_ssize_t i, object x, NativeTarget target):
    """``pop[i] = agent(x, target)``: position, objectives and fitness of row ``i``."""
    _write(pop, i, x, target)


def commit(LegacyNativeOptimizer opt, NativePopulation pop, NativePopulation cand, Py_ssize_t idx, object x, bint swarm,
           bint old_first=False):
    """One agent of a loop that reads the rows updated before it.

    Sequential mode (``swarm`` false): evaluate ``x`` and replace row ``idx`` when it is better
    (``get_better_agent(new, old)``). ``swarm``/``parallel`` modes: only store ``x`` in ``cand``;
    call :func:`finish` after the loop to evaluate the block and apply the greedy selection.
    """
    cdef NativeTarget target
    cdef double old, new
    cdef bint minimize, better
    if swarm:
        cand.buf[idx, cand.cX:cand.cX + cand.d] = x
        return
    target = opt.get_target(x)
    old = pop.view[idx, pop.cF]
    new = target.fitness
    minimize = opt.problem.minmax == "min"
    if old_first:  # get_better_agent(old, new)
        better = not (old < new) if minimize else old < new
    else:
        better = new < old if minimize else not (new < old)
    if better:
        _write(pop, idx, x, target)


def finish(LegacyNativeOptimizer opt, NativePopulation cand, Py_ssize_t start, Py_ssize_t stop):
    """Batched evaluation of ``cand[start:stop]`` and greedy selection (``swarm``/``parallel`` modes)."""
    opt.evaluate(cand, start, stop)
    accept(opt, cand, start, stop)


def greedy(LegacyNativeOptimizer opt, NativePopulation cand, Py_ssize_t start=0, Py_ssize_t stop=-1,
           NativePopulation dst=None):
    """``greedy_selection_population(old, new)`` in every mode: min ``new < old``, max ``new > old``."""
    cdef NativePopulation pop = opt.pop if dst is None else dst
    if stop < 0:
        stop = pop.n
    new = cand.buf[start:stop, cand.cF]
    old = pop.buf[start:stop, pop.cF]
    better = new < old if opt.problem.minmax == "min" else new > old
    rows = start + np.flatnonzero(better)
    pop.buf[rows] = cand.buf[rows]


# -- helpers for algorithms that keep classic agent lists (shared/aliased agents, growing archives, ...) --

def agents_of(NativePopulation pop):
    """The rows of ``pop`` as standalone agents."""
    return [pop.agent(i) for i in range(pop.n)]


def population_of(NativePopulation template, agents):
    """A population with the layout of ``template`` holding the position/target of each agent."""
    cdef NativePopulation pop = template.take(np.zeros(len(agents), dtype=int))
    for i, agent in enumerate(agents):
        set_row(pop, i, agent.solution, agent.target)
    return pop


def better_agent(LegacyNativeOptimizer opt, x, y):
    """``get_better_agent(x, y)``: a copy of the winner (ties go to ``y`` for min, ``x`` for max)."""
    if opt.problem.minmax == "min":
        return x.copy() if x.target.fitness < y.target.fitness else y.copy()
    return y.copy() if x.target.fitness < y.target.fitness else x.copy()


def sorted_agents(LegacyNativeOptimizer opt, agents):
    """``get_sorted_population``: best first, the classic argsort order."""
    order = np.argsort([agent.target.fitness for agent in agents]).tolist()
    if opt.problem.minmax == "max":
        order = order[::-1]
    return [agents[i] for i in order]


def update_targets(LegacyNativeOptimizer opt, agents):
    """``update_target_for_population``: swarm/parallel modes evaluate every agent, sequential mode does nothing."""
    if opt.mode in opt.AVAILABLE_MODES:
        for agent in agents:
            agent.target = opt.get_target(agent.solution, counted=False)
        opt._nfe_counter += len(agents)
    return agents


def greedy_agents(LegacyNativeOptimizer opt, old, new):
    """``greedy_selection_population(old, new)`` for agent lists."""
    if len(old) != len(new):
        raise ValueError("Greedy selection of two population with different length.")
    if opt.problem.minmax == "min":
        return [new[i] if new[i].target.fitness < old[i].target.fitness else old[i] for i in range(len(old))]
    return [new[i] if new[i].target.fitness > old[i].target.fitness else old[i] for i in range(len(old))]


def new_agent(LegacyNativeOptimizer opt, x, evaluate=True):
    """``generate_agent(x)`` (evaluated) or ``generate_empty_agent(x)`` (``evaluate=False``)."""
    return LegacyNativeAgent(x, opt.get_target(x) if evaluate else None)


def build_population(LegacyNativeOptimizer opt, agents):
    """A new population (no template needed) holding the position/target of each agent."""
    # (problem.n_objs is not read: its first access draws a random solution from the seeded stream)
    cdef Py_ssize_t m = np.asarray(agents[0].target.objectives).size if len(agents) else 1
    cdef NativePopulation pop = NativePopulation(
        len(agents), opt.problem.n_dims, m, opt.layout(opt.problem.n_dims, m), opt.problem._obj_weights,
    )
    for i, agent in enumerate(agents):
        set_row(pop, i, agent.solution, agent.target)
    return pop


# -- whole-population steps for the vectorized collection (synchronous updates, strict comparisons) --

def better(LegacyNativeOptimizer opt, a, b):
    """Element-wise "a is better than b" (strict): ``a < b`` for min, ``a > b`` for max."""
    return a < b if opt.problem.minmax == "min" else a > b


def step(LegacyNativeOptimizer opt, pos, NativePopulation dst=None, Py_ssize_t stop=-1, Py_ssize_t start=0):
    """One synchronous phase: bound ``pos`` (rows ``[start, stop)``, default all), evaluate the block once, keep the better rows of ``dst`` (default ``opt.pop``)."""
    cdef NativePopulation pop = opt.pop if dst is None else dst
    cdef NativePopulation cand = pop.empty_like()
    if stop < 0:
        stop = pop.n
    cand.buf[:] = pop.buf  # the candidates carry the agents' extra fields
    cand.X[start:stop] = opt.correct_solution(pos)
    opt.evaluate(cand, start, stop)
    greedy(opt, cand, start, stop, pop)


def replace(LegacyNativeOptimizer opt, pos):
    """Every agent moves to its (bounded, evaluated) candidate."""
    cdef NativePopulation pop = opt.pop
    cdef NativePopulation cand = pop.empty_like()
    cand.X[:] = opt.correct_solution(pos)
    opt.evaluate(cand, 0, pop.n)
    opt.pop = cand


def others(LegacyNativeOptimizer opt, Py_ssize_t n, Py_ssize_t k=1):
    """``k`` random agent indices per agent, never the agent itself (n, k); with replacement among the others."""
    return (np.arange(n)[:, None] + opt.generator.integers(1, n, size=(n, k))) % n


def better_pick(LegacyNativeOptimizer opt, NativePopulation pop):
    """For each agent, a random agent that is strictly better than it: ``(index, has_one)``, both shaped ``(n,)``."""
    F = np.ascontiguousarray(pop.F)
    B = F[None, :] < F[:, None] if opt.problem.minmax == "min" else F[None, :] > F[:, None]
    keys = opt.generator.random(B.shape)
    keys[~B] = -1.0
    return keys.argmax(axis=1), B.any(axis=1)


def roulette(LegacyNativeOptimizer opt, fitness, Py_ssize_t size):
    """``size`` indices drawn by roulette wheel on ``fitness`` (min or max problems, negative values allowed), as ``get_index_roulette_wheel_selection``."""
    f = np.asarray(fitness, dtype=float).ravel()
    n = len(f)
    if np.ptp(f) == 0:
        return opt.generator.integers(0, n, size=size)
    if np.any(f < 0):
        f = f - np.min(f)
    if opt.problem.minmax == "min":
        f = np.max(f) - f
    return opt.generator.choice(n, size=size, p=f / np.sum(f))


def two_others(LegacyNativeOptimizer opt, Py_ssize_t n, Py_ssize_t k):
    """Two random agent indices per (agent, column): ``(a, b)`` shaped ``(n, k)``, both different from the agent and from each other."""
    rng = opt.generator
    me = np.arange(n)[:, None]
    a = rng.integers(0, n - 1, size=(n, k))
    a = a + (a >= me)
    b = rng.integers(0, n - 2, size=(n, k))
    lo, hi = np.minimum(me, a), np.maximum(me, a)
    b = b + (b >= lo)
    b = b + (b >= hi)
    return a, b


def scatter(LegacyNativeOptimizer opt, NativePopulation cand, targets, NativePopulation dst=None):
    """Row ``i`` of ``cand`` competes with row ``targets[i]`` of ``dst`` (default ``opt.pop``); when several candidates aim at the same row the best one wins."""
    cdef NativePopulation pop = opt.pop if dst is None else dst
    targets = np.asarray(targets)
    cf = np.asarray(cand.F)
    key = cf if opt.problem.minmax == "min" else -cf
    order = np.argsort(key, kind="stable")
    _, first = np.unique(targets[order], return_index=True)
    win = order[first]
    tgt = targets[win]
    ok = better(opt, cf[win], np.asarray(pop.F)[tgt])
    pop.buf[tgt[ok]] = cand.buf[win[ok]]


def k_others(LegacyNativeOptimizer opt, Py_ssize_t n, Py_ssize_t k):
    """``k`` distinct random agent indices per agent, none of them the agent itself: ``(n, k)`` (unordered)."""
    keys = opt.generator.random((n, n))
    keys[np.arange(n), np.arange(n)] = 2.0
    return np.argpartition(keys, k - 1, axis=1)[:, :k]


def neighbors(Py_ssize_t n):
    """Ring-with-ends neighbours of every agent of a sorted population: ``(prev, next)`` (GSK-style, ends look two agents in)."""
    idx = np.arange(n)
    prev, nxt = idx - 1, idx + 1
    prev[0], nxt[0] = 2, 1
    prev[n - 1], nxt[n - 1] = n - 3, n - 2
    return prev, nxt


def exclude(r, excluded):
    """Map values ``r`` drawn from ``[0, n - k)`` to ``[0, n)`` skipping the ``k`` (per-row) ``excluded`` values (n, k)."""
    r = np.array(r)
    for col in np.sort(excluded, axis=1).T:
        r = r + (r >= col)
    return r


def pick_range(LegacyNativeOptimizer opt, Py_ssize_t lo, Py_ssize_t hi, idx, size=None):
    """A random index in ``[lo, hi)`` per agent, never the agent itself (``idx``, per-row); shape ``size`` (default ``idx``'s)."""
    idx = np.asarray(idx)
    inside = (idx >= lo) & (idx < hi)
    r = lo + opt.generator.integers(0, (hi - lo) - inside.astype(int), size=idx.shape if size is None else size)
    return r + (inside & (r >= idx))


def best_row(LegacyNativeOptimizer opt, NativePopulation pop):
    """Index of the best agent of ``pop`` right now (the engine only refreshes ``g_best`` at the end of the epoch)."""
    F = np.asarray(pop.F)
    return int(F.argmin() if opt.problem.minmax == "min" else F.argmax())
