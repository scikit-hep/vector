# Copyright (c) 2019-2020, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import awkward1 as ak
import numpy as np

from .common.lorentz import LorentzXYZCommon


class LorentzXYZ(ak.Record, LorentzXYZCommon):
    def __repr__(self):
        return "Lxyz({0:.3g} {1:.3g} {2:.3g} {3:.3g})".format(
            self.x, self.y, self.z, self.t
        )


class LorentzXYZArray(ak.Array, LorentzXYZCommon):
    pass


# This function only works as a ufunc overload, but it creates an AwkwardArray
def lorentz_add_xyz_xyz(left, right):
    x = ak.layout.NumpyArray(np.asarray(left["x"]) + np.asarray(right["x"]))
    y = ak.layout.NumpyArray(np.asarray(left["y"]) + np.asarray(right["y"]))
    z = ak.layout.NumpyArray(np.asarray(left["z"]) + np.asarray(right["z"]))
    t = ak.layout.NumpyArray(np.asarray(left["t"]) + np.asarray(right["t"]))
    return ak.layout.RecordArray(
        {"x": x, "y": y, "z": z, "t": t}, parameters={"__record__": "LorentzXYZ"},
    )


# Define some behaviors for Lorentz vectors.
lorentzbehavior = dict(ak.behavior)

# Any records with __record__ = "LorentzXYZ" will be mapped to LorentzXYZ instances.
lorentzbehavior["LorentzXYZ"] = LorentzXYZ

# Any arrays containing such records (any number of levels deep) will be LorentsXYZArrays.
lorentzbehavior["*", "LorentzXYZ"] = LorentzXYZArray

# The NumPy ufunc for "add" will use our definition for __record__ = "LorentzXYZ".
lorentzbehavior[np.add, "LorentzXYZ", "LorentzXYZ"] = lorentz_add_xyz_xyz
