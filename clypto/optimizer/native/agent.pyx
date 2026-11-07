#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""``Agent``: a single, fixed-layout data structure for agent state.

Every agent holds its ``solution`` and, once evaluated, its ``objectives``,
their ``weights`` and the weighted ``fitness``, plus the per-agent state the
algorithms need (``velocity``, ``pbest_solution``, ``age``, ..., one slot per
algorithm family; the slots the current algorithm does not use stay ``None``/0).
``Agent`` is the one agent type of the engine and the single clone/copy entry
point: ``cy.duplicate_agent`` builds a copy of the **same dynamic type** —
solution and evaluation shared (references), state copied field by field — and
``agent.snapshot()`` also copies the solution vector. Agent subclasses carry
extra state by implementing ``_new_instance`` + ``_copy_state_to`` (the two
hooks ``duplicate_agent`` dispatches on; see ``tools/scaffold_agent.py``).
``cy.is_better`` compares fitness and ``cy.sync_if_duplicate`` copies an
evaluation between agents with the same solution.

The evaluation is read-only: it changes only through ``evaluate(problem)`` or
``update_solution(other)``. Copies share the ``solution`` array: replace it,
never mutate it in place.
"""
import cython

import numpy as np
cimport numpy as cnp

from clypto.optimizer.native.problem cimport Problem

cnp.import_array()


cpdef Agent duplicate_agent(Agent agent):
    """New agent of the same dynamic type: solution and evaluation shared (references), state copied field by field.

    The engine's canonical copy (Python callers use ``agent.copy()``). Hooks are
    virtual: a subclass carries its extra state by overriding ``_new_instance``
    and ``_copy_state_to`` (the base fields go first, always); it must never
    override ``copy`` itself.
    """
    cdef Agent new = agent._new_instance()
    if type(new) is not type(agent):
        raise TypeError(
            f"{type(agent).__name__}._new_instance() returned {type(new).__name__}; "
            "it must return a blank instance of the same type."
        )
    agent._copy_state_to(new)
    if agent.solution is not None and new.solution is not agent.solution:
        raise RuntimeError(
            f"{type(agent).__name__}._copy_state_to() must call Agent._copy_state_to() first "
            "(the state is copied and the solution shared, not replaced)."
        )
    return new


cpdef bint sync_if_duplicate(Agent that, Agent other):
    """Give ``that`` the evaluation of ``other`` when their solutions match (atol 1e-6)."""
    try:
        same = np.allclose(that.solution, other.solution, rtol=0.0, atol=1e-6)
    except (TypeError, ValueError):
        same = that.solution is other.solution or np.array_equal(
            np.asarray(that.solution), np.asarray(other.solution)
        )
    if same:
        that.copy_evaluation(other)
        return True
    return False


cpdef bint better_fitness(double x, double y, str sense):
    """True when fitness ``x`` is better than ``y``: ``x < y`` for min, ``not (x < y)`` for max."""
    if sense == "min":
        return x < y
    return not (x < y)


cpdef bint is_better(Agent x, Agent y, str sense):
    """True when agent ``x`` is better than ``y`` (``better_fitness`` of their fitness)."""
    return better_fitness(x.fitness, y.fitness, sense)


cpdef Agent get_better_agent(Agent x, Agent y, str sense="min", bint reverse=False):
    """Copy of the better agent; a tie keeps ``y`` when minimizing, ``x`` when maximizing."""
    cdef bint maximize = (sense != "min") != reverse
    if x.fitness < y.fitness:
        return duplicate_agent(y if maximize else x)
    return duplicate_agent(x if maximize else y)


cpdef list argsort_agents(object agents, str sense):
    """Positions best first (``np.argsort`` of the fitness, reversed when maximizing)."""
    fitness = np.array([agent.fitness for agent in agents], dtype=float)
    order = np.argsort(fitness).tolist()
    return order[::-1] if sense == "max" else order


cpdef list greedy_agents(object old, object new, str sense, str mode="swarm"):
    """Per position, the ``new`` agent when better, else the ``old`` one (lists of equal length).

    The batch modes keep ``new`` only when strictly better. ``mode="sequential"`` applies the
    one-by-one rule of ``get_better_agent`` (``better_fitness``: when maximizing, a tie keeps ``new``),
    so an algorithm that evaluates its whole batch at once keeps its sequential results exactly.
    """
    if len(old) != len(new):
        raise ValueError("Greedy selection of two population with different length.")
    if mode == "sequential":
        return [n if better_fitness(n.fitness, o.fitness, sense) else o for o, n in zip(old, new)]
    if sense == "max":
        return [n if n.fitness > o.fitness else o for o, n in zip(old, new)]
    return [n if n.fitness < o.fitness else o for o, n in zip(old, new)]


cpdef list sort_agents(object agents, str sense):
    """The agents best first (see ``argsort_agents``)."""
    return [agents[i] for i in argsort_agents(agents, sense)]


cdef class Agent:
    def __cinit__(self):
        self.fitness = np.nan

    def __init__(self, solution=None, objectives=None, weights=None):
        self.solution = solution
        if objectives is not None:
            self.set_evaluation(objectives, weights)

    cdef void set_evaluation(self, object objectives, object weights):
        """Store ``objectives`` (number or sequence, always ``float64``) and ``weights``;
        fitness is their dot product. Mismatched weight lengths now raise."""
        cdef cnp.ndarray obs = np.asarray(objectives, dtype=np.float64).ravel()
        cdef Py_ssize_t n = obs.shape[0], m
        self.objectives = obs
        if weights is None:
            self.weights = n
            if n == 1:
                self.fitness = obs[0]
            else:
                self.fitness = np.dot(n * (1.0,), obs)
            return
        cdef cnp.ndarray w = np.asarray(weights, dtype=np.float64).ravel()
        m = w.shape[0]
        if m != n:
            raise ValueError(
                f"Objective weights of length {m} do not match the number of objectives {n}."
            )
        self.weights = w
        if n == 1:
            self.fitness = w[0] * obs[0]
        else:
            self.fitness = np.dot(w, obs)

    cdef void copy_evaluation(self, Agent other):
        self.objectives = other.objectives
        self.weights = other.weights
        self.fitness = other.fitness

    cpdef Agent evaluate(self, Problem problem):
        """Evaluate ``solution`` on ``problem`` (counted in ``problem.n_evals``)."""
        self.set_evaluation(problem.evaluate_solution(self.solution), problem._obj_weights)
        return self

    cpdef void update_solution(self, Agent other, object solution=None):
        """Take the evaluation of ``other`` and move to ``solution`` (default: a copy of ``other.solution``).

        ``solution`` is stored as given, so ``agent.update_solution(candidate, pos_new)``
        keeps ``pos_new`` itself; ``agent.update_solution(other, agent.solution)``
        only takes the evaluation.
        """
        self.solution = other.solution.copy() if solution is None else solution
        self.copy_evaluation(other)

    # -- the copy protocol --------------------------------------------------------
    cdef Agent _new_instance(self):
        """A blank instance of this type; subclasses must override it."""
        if type(self) is not Agent:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _new_instance() returning "
                "a blank instance of its own type (Agent.__new__ is not safe there)."
            )
        return Agent.__new__(Agent)

    cdef void _copy_state_to(self, Agent other):
        """Copy every field onto ``other``: evaluation state and the 41 state slots.

        Always shared by reference (arrays and objects). A subclass must call
        ``Agent._copy_state_to(self, other)`` first, then copy its own fields.
        """
        other.solution = self.solution
        other.objectives = self.objectives
        other.weights = self.weights
        other.fitness = self.fitness
        other.acc = self.acc
        other.age = self.age
        other.bitstring = self.bitstring
        other.cost = self.cost
        other.crossover = self.crossover
        other.damage = self.damage
        other.den = self.den
        other.flag = self.flag
        other.g = self.g
        other.hunger = self.hunger
        other.intensity = self.intensity
        other.interaction = self.interaction
        other.local_best = self.local_best
        other.local_solution = self.local_solution
        other.local_vector = self.local_vector
        other.loudness = self.loudness
        other.m = self.m
        other.mask = self.mask
        other.mass = self.mass
        other.model_weights = self.model_weights
        other.mu = self.mu
        other.mutation = self.mutation
        other.nutrients = self.nutrients
        other.pbest_solution = self.pbest_solution
        other.pbest_fitness = self.pbest_fitness
        other.pop_size = self.pop_size
        other.pulse_frequency = self.pulse_frequency
        other.pulse_rate = self.pulse_rate
        other.q_table = self.q_table
        other.sigma = self.sigma
        other.solution_new = self.solution_new
        other.step = self.step
        other.strategy = self.strategy
        other.target_move = self.target_move
        other.target_new = self.target_new
        other.target_solution = self.target_solution
        other.v = self.v
        other.velocity = self.velocity
        other.vol = self.vol
        other.weight = self.weight
        other.win = self.win

    cpdef Agent copy(self):
        """The copy protocol's Python face: ``duplicate_agent(self)`` (never override it)."""
        return duplicate_agent(self)

    cpdef Agent snapshot(self):
        """A ``copy`` whose solution vector is independent of this agent's."""
        cdef Agent new = duplicate_agent(self)
        if self.solution is not None:
            new.solution = self.solution.copy()
        return new

    def __repr__(self):
        return f"{type(self).__name__}(fitness={self.fitness}, solution={self.solution})"

    def __float__(self):
        if self.objectives is None:
            raise ValueError("Agent cannot generate a value from fitness")
        return self.fitness
