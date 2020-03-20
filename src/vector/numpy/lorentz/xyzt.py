from vector.core import numpy as np
from vector.common.lorentz.xyzt import LorentzXYZTCommon


class LorentzXYZT(LorentzXYZTCommon):
    def __init__(self, x, y, z, t):
        """
        Notes
        =====

        For now, all arrays are broadcast here - in the future, arrays may remain unbroadcast until an action is taken.
        """

        self.x, self.y, self.z, self.t = np.broadcast_arrays(x, y, z, t)

    def __repr__(self):
        return "Lxyz({0}, {1}, {2}, {3})".format(self.x, self.y, self.z, self.t)

    def __getitem__(self, attr):
        # It has to behave the same way as the bound objects or users will get confused.
        if attr in ("x", "y", "z", "t"):
            return getattr(self, attr)
        else:
            raise ValueError("key {0} does not exist in x,y,z,t".format(attr))
