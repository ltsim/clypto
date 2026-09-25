#!/usr/bin/env python
# Created by "Furkan Gunbaz (gunbaz) ---------------%
#       Email: furkan.gunbaz@gmail.com              %
#       Github: https://github.com/gunbaz           %
# --------------------------------------------------%
# IMPLEMENTATION NOTES:
# This implementation follows the PTI (Polarization Type Indicator) mechanism
# from the original paper exactly as described in Algorithm 1 and Algorithm 2.
# Key features:
# 1. PTI Update (Algorithm 1): LPA, RPA, LPT, RPT, LAD, RAD calculations (Eq. 5, 6, 7)
# 2. Strategy 1 - Foraging: Langevin/Brownian equation (Eq. 12)
# 3. Strategy 2 - Attack/Strike: Circular motion equation (Eq. 14)
# 4. Strategy 3 - Defense/Burrow: Defense/Shelter split with k parameter (Eq. 15)
#
# CRITICAL: LPA is calculated from intra-iteration change (X_i(t) vs X'_i(t)),
# not inter-iteration change. PTI update happens AFTER strategy application.

from clypto.native.collection.legacy.swarm_based.MShOA.OriginalMShOA import OriginalMShOA

__all__ = ["OriginalMShOA"]
