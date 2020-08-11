# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

import vector.common.lorentz.xyzt
from vector.core import numpy as np


class LorentzXYZT(vector.common.lorentz.xyzt.LorentzXYZTNormal):
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
