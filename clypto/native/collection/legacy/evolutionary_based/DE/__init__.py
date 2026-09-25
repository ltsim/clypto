#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

from clypto.native.collection.legacy.evolutionary_based.DE.OriginalDE import OriginalDE

from clypto.native.collection.legacy.evolutionary_based.DE.JADE import JADE

from clypto.native.collection.legacy.evolutionary_based.DE.SADE import SADE

from clypto.native.collection.legacy.evolutionary_based.DE.SAP_DE import SAP_DE

__all__ = ["OriginalDE", "JADE", "SADE", "SAP_DE"]
