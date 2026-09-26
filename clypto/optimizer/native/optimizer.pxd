from clypto.optimizer.native.agent cimport LegacyNativeAgent
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.problem cimport NativeProblem
from clypto.optimizer.native.target cimport NativeTarget


cdef class LegacyNativeOptimizer:
    cdef public int epoch
    cdef public int pop_size
    cdef public LegacyNativeAgent g_best
    cdef public LegacyNativeAgent g_worst
    cdef public NativePopulation pop
    cdef public NativeProblem problem
    cdef public object generator
    cdef public object rng
    cdef public object tracker
    cdef public object mode
    cdef public object n_workers
    cdef public bint sort_flag
    cdef public bint is_parallelizable
    cdef public double EPSILON
    cdef public tuple AVAILABLE_MODES
    cdef public tuple SUPPORTED_ARRAYS
    cdef object _termination
    cdef str _name
    cdef tuple _params_name_ordered
    cdef object _last_gbest_fit
    cdef long long _nfe_counter
    cdef int _repeated_times
    cdef Py_ssize_t _g_best_row     # row aliased by g_best (-1: g_best is a copy)
    cdef object _starting

    # Lifecycle hooks: override with the same cdef signature.
    cdef void initialize_variables(self)
    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void init_fields(self, NativePopulation pop)
    cdef void before_initialization(self, object starting_solutions=*)
    cdef void initialization(self)
    cdef void after_initialization(self)
    cdef void before_main_loop(self)
    cdef void evolve(self, int epoch)
    cdef object amend_solution(self, object solution)

    cpdef object correct_solution(self, object solution)
    cdef NativePopulation generate_population(self, Py_ssize_t k)
    cdef NativePopulation new_population(self, object X)
    cdef object _objectives(self, object X)
    cdef object _fitness(self, object R)
    cdef void evaluate(self, NativePopulation pop, Py_ssize_t start, Py_ssize_t stop)
    cdef object sorted_order(self, NativePopulation pop)
    cdef object g_best_x(self)
    cdef LegacyNativeAgent current_g_best(self)
    cpdef NativeTarget get_target(self, object solution, bint counted=*)
    cdef void _update_repeated_times(self)
    cpdef LegacyNativeAgent solve(
        self, object problem, object termination=*, object starting_solutions=*,
        object seed=*, bint debug=*, bint track_population=*, object history_path=*,
        object before_iteration=*, object after_iteration=*,
    )
