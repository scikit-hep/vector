# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


import numbers
from typing import Any, Dict

import awkward as ak
import numpy as np

import vector.core.lorentz.xyzt
from vector.mixins.lorentz.xyzt import LorentzXYZTMethodMixin


class LorentzXYZT(ak.Record, LorentzXYZTMethodMixin):
    def __repr__(self):
        return "Lxyz({:.3g} {:.3g} {:.3g} {:.3g})".format(
            self.x, self.y, self.z, self.t
        )


class LorentzXYZTArray(ak.Array, LorentzXYZTMethodMixin):
    pass


def _create_dict(args):
    keys = ("x", "y", "z", "t")
    vals = {k: v for k, v in zip(keys, args)}
    return ak.zip(vals, with_name="LorentzXYZT")


def _create_behavior(function):
    return lambda a, b: _create_dict(function(a, b))


def _create_behavior_r(function):
    return lambda a, b: _create_dict(function(b, a))


# Define some behaviors for Lorentz vectors.
behavior = dict()  # type: Dict[Any, Any]

# Any records with __record__ = "LorentzXYZT" will be mapped to LorentzXYZT instances.
behavior["LorentzXYZT"] = LorentzXYZT

# Any arrays containing such records (any number of levels deep) will be LorentsXYZArrays.
behavior["*", "LorentzXYZT"] = LorentzXYZTArray

behavior[np.add, "LorentzXYZT", "LorentzXYZT"] = _create_behavior(
    vector.core.lorentz.xyzt.add
)

behavior[np.add, "LorentzXYZT", numbers.Real] = _create_behavior(
    vector.core.lorentz.xyzt.add_scalar
)

behavior[np.add, numbers.Real, "LorentzXYZT"] = _create_behavior_r(
    vector.core.lorentz.xyzt.add_scalar
)

behavior[np.multiply, "LorentzXYZT", "LorentzXYZT"] = vector.core.lorentz.xyzt.dot

behavior[np.multiply, "LorentzXYZT", numbers.Real] = _create_behavior(
    vector.core.lorentz.xyzt.multiply_scalar
)

behavior[np.multiply, numbers.Real, "LorentzXYZT"] = _create_behavior_r(
    vector.core.lorentz.xyzt.multiply_scalar
)
