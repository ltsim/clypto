#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np



from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalSRSR(AgentListOptimizer):
    """
    The original version of: Swarm Robotics Search And Rescue (SRSR)

    Links:
        1. https://doi.org/10.1016/j.asoc.2017.02.028

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SRSR    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = SRSR.OriginalSRSR(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bakhshipour, M., Ghadi, M.J. and Namdari, F., 2017. Swarm robotics search & rescue: A novel
    artificial intelligence-inspired optimization approach. Applied Soft Computing, 57, pp.708-726.
    """

    cdef public object mu_factor
    cdef public object sigma_temp
    cdef public object SIF
    cdef public object movement_factor

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def generate_empty_agent(self, solution: np.ndarray | None = None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)

        mu = 0
        sigma = 0
        x_new = solution.copy()
        target_move = 0

        return FieldAgent(
            solution=solution,
            mu=mu,
            sigma=sigma,
            solution_new=x_new,
            target_move=target_move,
        )

    def generate_agent(self, solution: np.ndarray | None = None):
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.target_new = agent.target.copy()
        return agent

    cdef void initialize_variables(self):
        # Control Parameters Of Algorithm
        # ==============================================================================================
        #  [c1] movement_factor : Determines Movement Pace Of Robots During Exploration Policy
        #  [c2] sigma_factor    : Determines Level Of Divergence Of Slave Robots From Master Robot
        #  [c3] sigma_limit     : Limits Value Of Sigma Factor
        #  [c4] mu_factor       : Controls Mean Value For Master And Slave Robots
        #       Control Parameters C1, C2 And C3 Are Automatically Tuned While C4 Should Be Set By User
        # ==============================================================================================
        self.mu_factor = (
            2 / 3
        )  # [0.1-0.9] Controls Dominance Of Master Robot, Preferably 2/3
        self.sigma_temp = np.zeros(self.pop_size)  # Initializing Temporary Stacks
        self.SIF = 2
        self.movement_factor = self.problem.ub - self.problem.lb

    def evolve_agents(self, epoch):
        # ========================================================================================= %%
        #            PHASE 1 (ACCUMULATION): CALCULATING Mu AND SIGMA values FOR SOLUTIONS            %
        # ===========================================================================================%%
        # ------ CALCULATING MU AND SIGMA FOR MASTER ROBOT ----------
        self.objs[0].sigma = self.generator.uniform()
        if epoch % 2 == 1:
            self.objs[0].mu = (1 - self.objs[0].sigma) * self.objs[0].solution
        else:
            self.objs[0].mu = (1 + (1 - self.mu_factor) * self.objs[0].sigma) * self.objs[
                0
            ].solution

        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.objs[idx].copy()
            # ---------- CALCULATING MU AND SIGMA FOR SLAVE ROBOTS ---------
            self.objs[idx].mu = (
                self.mu_factor * self.objs[0].solution
                + (1 - self.mu_factor) * self.objs[idx].solution
            )
            if epoch == 0:
                self.SIF = 6
            self.sigma_temp[idx] = self.SIF * self.generator.uniform()
            self.objs[idx].sigma = self.sigma_temp[idx] * np.abs(
                self.objs[0].solution - self.objs[idx].solution
            ) + self.generator.uniform() ** 2 * (
                (self.objs[0].solution - self.objs[idx].solution) < 0.05
            )
            # ----- Generating New Positions Using New Obtained Mu And Sigma Values --------------
            pos_new = self.generator.normal(
                self.objs[idx].mu, self.objs[idx].sigma, self.problem.n_dims
            )
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            # --------- Calculate Degree Of Cost Movement Of Robots During Movement --------------
            self.objs[idx].target_move = (
                self.objs[idx].target.fitness - self.objs[idx].target_new.fitness
            )
            self.objs[idx].solution_new = pop_new[idx].solution.copy()
            self.objs[idx].target_new = pop_new[idx].target.copy()
            # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------
            # Replace Solution If It Reached To A More Quality Position
            if self.compare_target(
                pop_new[idx].target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx].solution = pop_new[idx].solution.copy()
                self.objs[idx].target = pop_new[idx].target.copy()

        # --------- Determining Sigma Improvement Factor (Sif) Based On Vvss Movement -------------------
        ## Get best improved fitness
        fit_id = np.argmax([agent.target_move for agent in self.objs])
        sigma_factor = 1 + self.generator.uniform() * np.max(
            self.problem.ub - self.problem.lb
        )
        self.SIF = sigma_factor * self.sigma_temp[fit_id]
        # Controlling Parameter Of Algorithm
        if self.SIF > np.max(self.problem.ub):
            self.SIF = np.max(self.problem.ub) * self.generator.uniform()

        # ========================================================================================= %%
        #            Phase 2 (Exploration): Moving Slave Robots Toward Master Robot                   %
        # ===========================================================================================%%
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.objs[idx].copy()
            gb = self.generator.uniform(-1, 1, self.problem.n_dims)
            gb[gb >= 0] = 1
            gb[gb < 0] = -1
            pos_new = (
                self.objs[idx].solution * self.generator.uniform()
                + gb * (self.objs[0].solution - self.objs[idx].solution)
                + self.movement_factor
                * self.generator.uniform(self.problem.lb, self.problem.ub)
            )
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            # --------- Calculate Degree Of Cost Movement Of Robots During Movement --------------
            self.objs[idx].target_move = (
                self.objs[idx].target.fitness - self.objs[idx].target_new.fitness
            )
            self.objs[idx].solution_new = pop_new[idx].solution.copy()
            self.objs[idx].target_new = pop_new[idx].target.copy()
            # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------
            # Replace Solution If It Reached To A More Quality Position
            if self.compare_target(
                pop_new[idx].target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx].solution = pop_new[idx].solution.copy()
                self.objs[idx].target = pop_new[idx].target.copy()

        # ========================================================================================= %%
        #        PHASE 3 (LOCAL SEARCH): CREATING SOME WORKER ROBOTS ASSIGNED TO SEARCH               %
        #                      LOCATIONS AROUND POSITION OF MASTER ROBOT                              %
        # ===========================================================================================%%

        if epoch > 0:
            # --- EXTRACTING "INTEGER PART" AND "FRACTIONAL PART"  OF THE ELEMENTS OF MASTER RPBOT POSITION------
            master_robot = {
                "original": np.reshape(self.objs[0].solution, (self.problem.n_dims, 1)),
                "sign": np.reshape(
                    np.sign(self.objs[0].solution), (self.problem.n_dims, 1)
                ),
                "abs": np.reshape(abs(self.objs[0].solution), (self.problem.n_dims, 1)),
                "int": np.reshape(
                    np.floor(abs(self.objs[0].solution)), (self.problem.n_dims, 1)
                ),
                # INTEGER PART
                "frac": np.reshape(
                    abs(self.objs[0].solution) - np.floor(abs(self.objs[0].solution)),
                    (self.problem.n_dims, 1),
                ),
            }  # FRACTIONAL PART

            # ------- Applying Nth-root And Nth-exponent Operators To Create Position Of New Worker Robots -------
            worker_robot1 = (
                master_robot["int"]
                + np.power(
                    master_robot["frac"], 1 / (1 + self.generator.integers(1, 4))
                )
            ) * master_robot["sign"]
            id_changed1 = np.argwhere(
                np.round(self.generator.uniform(self.problem.lb, self.problem.ub))
            )
            id_changed1 = np.reshape(id_changed1, (len(id_changed1)))
            worker_robot1 = np.reshape(worker_robot1, (self.problem.n_dims, 1))
            worker_robot1[id_changed1] = master_robot["original"][id_changed1]

            worker_robot2 = (
                master_robot["int"]
                + np.power(master_robot["frac"], (1 + self.generator.integers(1, 4)))
            ) * master_robot["sign"]
            id_changed2 = np.argwhere(
                np.round(self.generator.uniform(self.problem.lb, self.problem.ub))
            )
            id_changed2 = np.reshape(id_changed2, (len(id_changed2)))
            worker_robot2 = np.reshape(worker_robot2, (self.problem.n_dims, 1))
            worker_robot2[id_changed2] = master_robot["original"][id_changed2]

            # -------- Applying A Combined Ga-like Operator To Create Position Of New Worker Robot -------------
            random_per_mutation = self.generator.permutation(self.problem.n_dims)
            sec1 = random_per_mutation[0 : int(self.problem.n_dims / 2)]
            sec2 = random_per_mutation[int(self.problem.n_dims / 2) :]
            worker_robot3 = np.zeros((self.problem.n_dims, 1))
            worker_robot3[sec1] = (
                master_robot["int"][sec1]
                + np.power(
                    master_robot["frac"][sec1], 1 / (1 + self.generator.integers(1, 4))
                )
            ) * master_robot["sign"][sec1]
            worker_robot3[sec2] = (
                master_robot["int"][sec2]
                + master_robot["frac"][sec2] ** (1 + self.generator.integers(1, 4))
            ) * master_robot["sign"][sec2]
            id_changed3 = np.argwhere(
                np.round(self.generator.uniform(self.problem.lb, self.problem.ub))
            )
            id_changed3 = np.reshape(id_changed3, (len(id_changed3)))
            worker_robot3[id_changed3] = master_robot["original"][id_changed3]

            # ------- Applying Round Operators To Create Position Of New Worker Robot -------------------
            worker_robot4 = np.ceil(master_robot["abs"]) * master_robot["sign"]
            id_changed4 = np.argwhere(
                np.round(self.generator.uniform(self.problem.lb, self.problem.ub))
            )
            id_changed4 = np.reshape(id_changed4, (len(id_changed4)))
            worker_robot4[id_changed4] = master_robot["original"][id_changed4]

            worker_robot5 = np.floor(master_robot["abs"]) * master_robot["sign"]
            id_changed5 = np.argwhere(
                np.round(self.generator.uniform(self.problem.lb, self.problem.ub))
            )
            id_changed5 = np.reshape(id_changed5, (len(id_changed5)))
            worker_robot5[id_changed5] = master_robot["original"][id_changed5]

            # --------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ---------------
            workers = np.concatenate(
                (
                    worker_robot1.T,
                    worker_robot2.T,
                    worker_robot3.T,
                    worker_robot4.T,
                    worker_robot5.T,
                ),
                axis=0,
            )
            pop_workers = []
            for idx in range(0, 5):
                pos_new = self.correct_solution(workers[idx])
                agent = self.generate_empty_agent(pos_new)
                pop_workers.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_workers[-1].target = self.get_target(pos_new)
            pop_workers = self.update_target_for_population(pop_workers)

            for idx in range(0, 5):
                if self.compare_target(
                    pop_workers[idx].target, self.objs[1].target, self.problem.minmax
                ):
                    self.objs[-(idx + 1)].solution = pop_workers[idx].solution.copy()
                    self.objs[-(idx + 1)].target = pop_workers[idx].target.copy()
