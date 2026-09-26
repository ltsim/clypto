from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalCRO(LegacyNativeOptimizer):
    cdef public object po
    cdef public object Fb
    cdef public object Fa
    cdef public object Fd
    cdef public object Pd
    cdef public object GCR
    cdef public object gamma_min
    cdef public object gamma_max
    cdef public object n_trials
    cdef public object reef
    cdef public object occupied_position
    cdef public object G1
    cdef public object alpha
    cdef public object gama
    cdef public object num_occupied
    cdef public object dyn_Pd
    cdef public object occupied_list
    cdef public object occupied_idx_list
    cdef public object objs

    cdef void initialization(self)
    cdef void evolve(self, int epoch_c)
