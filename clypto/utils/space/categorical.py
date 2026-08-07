import numpy as np
import typing

from clypto.utils.space.strings import StringVar


class CategoricalVar(StringVar):
    def __init__(self, valid_sets=(("",),), name="categorical"):
        super().__init__(valid_sets, name)

    def generate(self):
        return [
            self.generator.choice(np.array(vl_set, dtype=object))
            for vl_set in self.valid_sets
        ]
