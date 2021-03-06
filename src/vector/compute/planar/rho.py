# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.planar import rho2
from vector.geometry import AzimuthalRhoPhi, AzimuthalXY, aztype


def xy(lib, x, y):
    return lib.sqrt(rho2.xy(lib, x, y))


def rhophi(lib, rho, phi):
    return rho


dispatch_map = {
    (AzimuthalXY,): xy,
    (AzimuthalRhoPhi,): rhophi,
}


def dispatch(v):
    with numpy.errstate(all="ignore"):
        return dispatch_map[
            aztype(v),
        ](v.lib, *v.azimuthal.elements)
