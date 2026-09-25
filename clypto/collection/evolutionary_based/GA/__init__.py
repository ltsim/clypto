#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.collection.evolutionary_based.GA.BaseGA import BaseGA
from clypto.collection.evolutionary_based.GA.SingleGA import SingleGA

from clypto.collection.evolutionary_based.GA.EliteSingleGA import EliteSingleGA
from clypto.collection.evolutionary_based.GA.MultiGA import MultiGA

from clypto.collection.evolutionary_based.GA.EliteMultiGA import EliteMultiGA
from clypto.collection.evolutionary_based.GA.OriginalGA import OriginalGA

__all__ = ["BaseGA", "SingleGA", "EliteSingleGA", "MultiGA", "EliteMultiGA", "OriginalGA"]
