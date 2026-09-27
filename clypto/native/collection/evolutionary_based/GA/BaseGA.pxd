cimport clypto.core as cy


cdef class BaseGA(cy.Optimizer):
    cdef tuple strategy_0(self, object pop_selected)
    cdef tuple strategy_1(self, object pop_dad, object pop_mom)
    cdef object offspring(self, object dad, object mom)
