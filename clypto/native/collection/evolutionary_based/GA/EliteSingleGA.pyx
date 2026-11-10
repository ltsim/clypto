#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.evolutionary_based.GA.SingleGA cimport SingleGA
cimport clypto.core as cy


cdef class EliteSingleGA(SingleGA):
    """
    The developed elite single-point mutation of: Genetic Algorithm (GA)

    Links:
        1. https://www.baeldung.com/cs/elitism-in-evolutionary-algorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation (str): Optional, can be ["flip", "swap", "scramble", "inversion"] for one-point
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + elite_best (float/int): Optional, can be float (percentage of the best in elite group), or int (the number of best elite), default = 0.1
        + elite_worst (float/int): Opttional, can be float (percentage of the worst in elite group), or int (the number of worst elite), default = 0.3
        + strategy (int): Optional, can be 0 or 1. If = 0, the selection is select parents from (elite_worst + non_elite_group).
            Else, the selection will select dad from elite_worst and mom from non_elite_group.
        + pop_size = elite_group (elite_best + elite_worst) + non_elite_group

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import GA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection = "roulette", crossover = "uniform",
    >>>                         mutation = "swap", elite_best = 0.1, elite_worst = 0.3, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    >>>
    >>> model2 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="swap")
    >>>
    >>> model5 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(
            self,
            epoch=10000,
            pop_size=100,
            pc=0.95,
            pm=0.8,
            selection="roulette",
            crossover="uniform",
            mutation="swap",
            k_way=0.2,
            elite_best=0.1,
            elite_worst=0.3,
            strategy=0,
            **kwargs
    ):
        super().__init__(
            epoch, pop_size, pc, pm, selection, crossover, mutation, k_way, **kwargs
        )
        self.elite_best = cy.check_is_int_and_float("elite_best", elite_best, [1, int(self.population.size() / 2) - 1], (0, 0.5))
        self.n_elite_best = (
            int(self.elite_best * self.population.size())
            if self.elite_best < 1
            else self.elite_best
        )
        if self.n_elite_best < 1:
            self.n_elite_best = 1

        self.elite_worst = cy.check_is_int_and_float("elite_worst", elite_worst, [1, int(self.population.size() / 2) - 1], (0, 0.5))
        self.n_elite_worst = (
            int(self.elite_worst * self.population.size())
            if self.elite_worst < 1
            else self.elite_worst
        )
        if self.n_elite_worst < 1:
            self.n_elite_worst = 1

        self.strategy = cy.validator(int, strategy, [0, 1], "strategy")
        self.parameters = [ "epoch", "pop_size", "pc", "pm", "selection", "crossover", "mutation", "k_way", "elite_best", "elite_worst", "strategy", ]
        self.sort_flag = True

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm: the ``n_elite_best`` agents survive, the others are
        replaced by children whose parents come from ``strategy_0`` or ``strategy_1``.

        Args:
            epoch (int): The current iteration
        """
        cdef int pop_size = self.population.size()
        cdef int n_best = self.n_elite_best
        cdef int n_worst = self.n_elite_worst
        cdef cy.Population n_population = self.population[:n_best]
        pop_old = self.population[n_best:]
        pop_dad, pop_mom = self.population[n_best:n_best + n_worst], self.population[n_best + n_worst:]
        for _ in range(n_best, pop_size):
            dad, mom = self.strategy_0(pop_old) if self.strategy == 0 else self.strategy_1(pop_dad, pop_mom)
            n_population.append(self.population.create_agent(cy.correct_solution(self.problem, self.offspring(dad, mom))))
        self.population = self.population.evaluate(n_population, self.mode)
