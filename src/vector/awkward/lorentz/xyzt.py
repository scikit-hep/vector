# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

from typing import Any, Dict

import awkward1 as ak

import numpy as np

import vector.core.lorentz.xyzt
from vector.common.lorentz.xyzt import LorentzXYZTCommon


class LorentzXYZT(ak.Record, LorentzXYZTCommon):
    def __repr__(self):
        return "Lxyz({0:.3g} {1:.3g} {2:.3g} {3:.3g})".format(
            self.x, self.y, self.z, self.t
        )


class LorentzXYZTArray(ak.Array, LorentzXYZTCommon):
    pass


def _create_dict(args):
    keys = ("x", "y", "z", "t")
    vals = {k: v for k, v in zip(keys, args)}
    return ak.zip(vals, with_name="LorentzXYZT")


def _create_behavior(function):
    return lambda a, b: _create_dict(function(a, b))


# Define some behaviors for Lorentz vectors.
behavior = dict()  # type: Dict[Any, Any]

# Any records with __record__ = "LorentzXYZT" will be mapped to LorentzXYZT instances.
behavior["LorentzXYZT"] = LorentzXYZT

# Any arrays containing such records (any number of levels deep) will be LorentsXYZArrays.
behavior["*", "LorentzXYZT"] = LorentzXYZTArray

# The NumPy ufunc for "add" will use our definition for __record__ = "LorentzXYZT".
behavior[np.add, "LorentzXYZT", "LorentzXYZT"] = _create_behavior(
    vector.core.lorentz.xyzt.add
)
