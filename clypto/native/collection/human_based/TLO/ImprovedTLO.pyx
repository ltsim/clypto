#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from functools import reduce
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.human_based.TLO.DevTLO cimport DevTLO


cdef class ImprovedTLO(DevTLO):
    """
    The original version of: Improved Teaching-Learning-based Optimization (ImprovedTLO)

    Links:
       1. https://doi.org/10.1016/j.scient.2012.12.005

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_teachers (int): [3, 10], number of teachers in class, default=5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import TLO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = TLO.ImprovedTLO(epoch=1000, pop_size=50, n_teachers = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V. and Patel, V., 2013. An improved teaching-learning-based optimization algorithm
    for solving unconstrained optimization problems. Scientia Iranica, 20(3), pp.710-720.
    """

    cdef public int n_teachers

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            n_teachers: int = 3,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_teachers (int): number of teachers in class
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.n_teachers = cy.validator(int, n_teachers, [2, int(np.sqrt(self.population.size()) - 1)], "n_teachers")
        self.parameters = ["epoch", "pop_size", "n_teachers"]
        self.n_students = self.population.size() - self.n_teachers
        self.n_students_in_team = int(self.n_students / self.n_teachers)
        self.sort_flag = False

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        sorted_pop = self.population.sort()
        self.g_best = cy.duplicate_agent(sorted_pop[0])
        self.teachers = sorted_pop[: self.n_teachers].copy()
        sorted_pop = sorted_pop[self.n_teachers:]
        idx_list = self.generator.permutation(range(0, self.n_students))
        self.teams = []
        for id_teacher in range(0, self.n_teachers):
            group = []
            for idx in range(0, self.n_students_in_team):
                start_index = id_teacher * self.n_students_in_team + idx
                group.append(sorted_pop[idx_list[start_index]])
            self.teams.append(group)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        for id_teach, teacher in enumerate(self.teachers):
            team = self.teams[id_teach]
            list_pos = np.array(
                [student.solution for student in self.teams[id_teach]]
            )  # Step 7
            mean_team = np.mean(list_pos, axis=0)
            pop_new = []
            for id_stud, student in enumerate(team):
                if teacher.fitness == 0:
                    TF = 1
                else:
                    TF = student.fitness / teacher.fitness
                diff_mean = self.generator.random() * (
                        teacher.solution - TF * mean_team
                )  # Step 8
                id2 = self.generator.choice(
                    list(set(range(0, self.n_teachers)) - {id_teach})
                )
                if cy.is_better(teacher, team[id2], self.problem.sense):
                    pos_new = (
                                      student.solution + diff_mean
                              ) + self.generator.random() * (
                                      team[id2].solution - student.solution
                              )
                else:
                    pos_new = (
                                      student.solution + diff_mean
                              ) + self.generator.random() * (
                                      student.solution - team[id2].solution
                              )
                pos_new = cy.correct_solution(self.problem, pos_new)
                agent = self.population.create_agent(pos_new)
                pop_new.append(agent)
                if self.mode == "sequential":
                    agent.evaluate(self.problem)
                    pop_new[-1] = cy.get_better_agent(agent, student, self.problem.sense)
            if self.mode != "sequential":
                pop_new = self.population.evaluate(pop_new, self.mode)
                pop_new = cy.greedy_agents(team, pop_new, self.problem.sense)
            self.teams[id_teach] = pop_new

        for id_teach, teacher in enumerate(self.teachers):
            ef = round(1 + self.generator.random())
            team = self.teams[id_teach]
            pop_new = []
            for id_stud, student in enumerate(team):
                id2 = self.generator.choice(
                    list(set(range(0, self.n_students_in_team)) - {id_stud})
                )
                if cy.is_better(student, team[id2], self.problem.sense):
                    pos_new = (
                            student.solution
                            + self.generator.random()
                            * (student.solution - team[id2].solution)
                            + self.generator.random()
                            * (teacher.solution - ef * team[id2].solution)
                    )
                else:
                    pos_new = (
                            student.solution
                            + self.generator.random()
                            * (team[id2].solution - student.solution)
                            + self.generator.random()
                            * (teacher.solution - ef * student.solution)
                    )
                pos_new = cy.correct_solution(self.problem, pos_new)
                agent = self.population.create_agent(pos_new)
                pop_new.append(agent)
                if self.mode == "sequential":
                    agent.evaluate(self.problem)
                    pop_new[-1] = cy.get_better_agent(agent, student, self.problem.sense)
            if self.mode != "sequential":
                pop_new = self.population.evaluate(pop_new, self.mode)
                pop_new = cy.greedy_agents(team, pop_new, self.problem.sense)
            self.teams[id_teach] = pop_new
        for id_teach, teacher in enumerate(self.teachers):
            team = self.teams[id_teach] + [teacher]
            team = cy.sort_agents(team, self.problem.sense)
            self.teachers[id_teach] = cy.duplicate_agent(team[0])
            self.teams[id_teach] = team[1:]
        self.population = self.population.spawn(self.teachers + reduce(lambda x, y: x + y, self.teams))
