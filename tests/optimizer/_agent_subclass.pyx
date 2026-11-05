# Compiled fixture of tests/optimizer: the agent-subclass copy contract.
# --------------------------------------------------%
"""Agent subclasses implementing the copy protocol of ``cy.duplicate_agent``.

+ ``TaggedAgent`` — the reference: extra ``tag``/``salary``/``history`` fields,
  ``_new_instance`` + ``_copy_state_to`` with the base call first.
+ ``NaughtyAgent`` — overrides ``_copy_state_to`` WITHOUT the base call
  (``duplicate_agent`` must fail loudly instead of losing the solution).
"""
import numpy as np
cimport numpy as cnp

from clypto.optimizer.native.agent cimport Agent

cnp.import_array()


cdef class TaggedAgent(Agent):
    """The protocol's reference subclass; extras are shared by reference."""

    def __cinit__(self):
        # Base __cinit__ does NOT run here (Cython calls the most derived one):
        # copies overwrite every field anyway, but keep fresh instances honest.
        self.fitness = np.nan
        self.salary = 0.0

    cdef Agent _new_instance(self):
        return TaggedAgent.__new__(TaggedAgent)

    cdef void _copy_state_to(self, Agent other):
        Agent._copy_state_to(self, other)
        cdef TaggedAgent target = <TaggedAgent>other
        target.tag = self.tag
        target.salary = self.salary
        target.history = self.history


cdef class NaughtyAgent(TaggedAgent):
    """A subclass that forgot the base call: the guard must reject it."""

    cdef Agent _new_instance(self):
        return NaughtyAgent.__new__(NaughtyAgent)

    cdef void _copy_state_to(self, Agent other):
        (<NaughtyAgent>other).tag = self.tag  # base fields silently skipped
