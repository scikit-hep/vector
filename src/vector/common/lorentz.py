import numpy as np

from ..core import lorentz


class LorentzXYZCommon(object):
    @property
    def pt(self):
        with np.errstate(invalid="ignore"):
            return lorentz.xyzt.pt(self)

    @property
    def eta(self):
        with np.errstate(invalid="ignore"):
            return lorentz.xyzt.eta(self)

    @property
    def phi(self):
        with np.errstate(invalid="ignore"):
            return lorentz.xyzt.phi(self)

    @property
    def mass(self):
        with np.errstate(invalid="ignore"):
            return lorentz.xyzt.mass(self)
