#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Particle Swarm Optimization on the vectorized native engine.

Each agent is a row of ``self.pop`` holding ``[F | O | X | V | P | PO | PF]``:
velocity ``V``, personal best position ``P`` and its objectives/fitness
``PO``/``PF``. Results are bit-identical to the classic per-agent version:

* Synchronous variants (OriginalPSO, AIW_PSO, LDW_PSO, P_PSO, C_PSO) update all
  agents at once. The random numbers the classic loop drew agent by agent are
  drawn as one block in the same order, and the population is processed in two
  chunks around the best row, because the classic ``g_best`` aliased that agent
  (agents after it saw its update).
* Sequential variants (HPSO_TVAC, CL_PSO) read agents updated earlier in the
  same epoch or draw a data-dependent number of random values, so they keep the
  agent-by-agent loop over buffer rows.
"""
from clypto.collection.swarm_based.PSO.OriginalPSO import OriginalPSO
from clypto.collection.swarm_based.PSO.AIW_PSO import AIW_PSO
from clypto.collection.swarm_based.PSO.LDW_PSO import LDW_PSO
from clypto.collection.swarm_based.PSO.P_PSO import P_PSO
from clypto.collection.swarm_based.PSO.HPSO_TVAC import HPSO_TVAC
from clypto.collection.swarm_based.PSO.C_PSO import C_PSO
from clypto.collection.swarm_based.PSO.CL_PSO import CL_PSO

__all__ = ["OriginalPSO", "AIW_PSO", "LDW_PSO", "P_PSO", "HPSO_TVAC", "C_PSO", "CL_PSO"]
