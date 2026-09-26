from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevGCO(VectorizeOptimizer):
    cdef public double cr
    cdef public double wf
    cdef public object dyn_list_cell_counter
    cdef public object dyn_list_life_signal

