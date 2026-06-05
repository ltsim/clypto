from opfunu.name_based.a_func import Ackley01

import clypto as cy
from clypto.collection.swarm_based import GWO

# 1. Define the benchmark function from Opfunu
# We'll use F3 from the CEC 2017 competition with 30 dimensions
ackley_f = Ackley01(ndim=30)

# 2. Define the problem dictionary for Clypto
# 'obj_func' is the evaluation method from Opfunu
# 'bounds' uses the lb (lower bound) and ub (upper bound) from the Opfunu object
problem = cy.Problem(
    obj_func=ackley_f.evaluate,
    bounds=cy.FloatVar(lb=ackley_f.lb, ub=ackley_f.ub),
    minmax="min",
)
# 3. Initialize and run the GWO algorithm
# epoch: number of iterations, pop_size: number of particles
model = GWO.OriginalGWO(epoch=150, pop_size=75)
g_best = model.solve(problem, debug=False)

# 4. Access the results
print(f"Best solution: {g_best.solution}")
print(f"Best fitness: {g_best.target.fitness}")
