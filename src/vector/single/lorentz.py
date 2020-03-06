from ..common.lorentz import LorentzXYZCommon

import json


class LorentzXYZFree(LorentzXYZCommon):
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def __repr__(self):
        return "Lxyz({0:.3g} {1:.3g} {2:.3g} {3:.3g})".format(
            self.x, self.y, self.z, self.t
        )

    def __getitem__(self, attr):
        # It has to behave the same way as the bound objects or users will get confused.
        if attr in ("x", "y", "z", "t"):
            return getattr(self, attr)
        else:
            raise ValueError(
                "key {0} does not exist (not in record)".format(json.dumps(attr))
            )
