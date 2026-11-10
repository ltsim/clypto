# Helpers cimported by the native collection through ``cimport clypto.core as cy``.

cpdef object validator(object dtype, object value, object bound=*, str name=*)
cpdef object check_is_int_and_float(str name, object value, object bound_int=*, object bound_float=*)
cpdef object levy_flight(object generator, double beta=*, double multiplier=*, object size=*, int case=*)
cpdef int roulette_wheel(object generator, str sense, object fitness)
cpdef list kway_tournament(object generator, str sense, object agents, object k_way=*, int output=*, bint reverse=*)
cpdef list split_groups(object agents, Py_ssize_t n_groups, Py_ssize_t m_agents)
