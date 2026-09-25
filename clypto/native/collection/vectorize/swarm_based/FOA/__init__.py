#!/usr/bin/env python
# Created by "Thieu" at 14:01, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

from clypto.collection.swarm_based.FOA.OriginalFOA import OriginalFOA
from clypto.collection.swarm_based.FOA.DevFOA import DevFOA

from clypto.collection.swarm_based.FOA.WhaleFOA import WhaleFOA

__all__ = ["OriginalFOA", "DevFOA", "WhaleFOA"]
