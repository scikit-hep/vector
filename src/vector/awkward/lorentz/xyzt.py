# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import division, absolute_import, print_function

import awkward1 as ak
import numpy as np

from vector.common.lorentz.xyzt import LorentzXYZTCommon


class LorentzXYZT(ak.Record, LorentzXYZTCommon):
    def __repr__(self):
        return "Lxyz({0:.3g} {1:.3g} {2:.3g} {3:.3g})".format(
            self.x, self.y, self.z, self.t
        )


class LorentzXYZTArray(ak.Array, LorentzXYZTCommon):
    pass


# This function only works as a ufunc overload, but it creates an AwkwardArray
def lorentz_add_xyz_xyz(left, right):
    x = ak.layout.NumpyArray(np.asarray(left["x"]) + np.asarray(right["x"]))
    y = ak.layout.NumpyArray(np.asarray(left["y"]) + np.asarray(right["y"]))
    z = ak.layout.NumpyArray(np.asarray(left["z"]) + np.asarray(right["z"]))
    t = ak.layout.NumpyArray(np.asarray(left["t"]) + np.asarray(right["t"]))
    return ak.layout.RecordArray(
        {"x": x, "y": y, "z": z, "t": t}, parameters={"__record__": "LorentzXYZT"},
    )


# Define some behaviors for Lorentz vectors.
behavior = dict(ak.behavior)

# Any records with __record__ = "LorentzXYZT" will be mapped to LorentzXYZT instances.
behavior["LorentzXYZT"] = LorentzXYZT

# Any arrays containing such records (any number of levels deep) will be LorentsXYZArrays.
behavior["*", "LorentzXYZT"] = LorentzXYZTArray

# The NumPy ufunc for "add" will use our definition for __record__ = "LorentzXYZT".
behavior[np.add, "LorentzXYZT", "LorentzXYZT"] = lorentz_add_xyz_xyz
