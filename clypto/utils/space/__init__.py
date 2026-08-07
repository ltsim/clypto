#!/usr/bin/env python
# Created by "Thieu" at 05:33, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
from clypto.utils.space.base import BaseVar
from clypto.utils.space.binary import BinaryVar, TransferBinaryVar
from clypto.utils.space.boolean import BoolVar, TransferBoolVar
from clypto.utils.space.categorical import CategoricalVar
from clypto.utils.space.floats import FloatVar
from clypto.utils.space.integers import IntegerVar
from clypto.utils.space.permutation import PermutationVar
from clypto.utils.space.sequence import SequenceVar
from clypto.utils.space.strings import StringVar

__all__ = ["BaseVar", "IntegerVar", "FloatVar", "StringVar", "BinaryVar", "TransferBinaryVar", "CategoricalVar", "SequenceVar",
           "PermutationVar", "BoolVar", "TransferBoolVar"]
