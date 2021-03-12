# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.planar import rho
from vector.geometry import AzimuthalRhoPhi, AzimuthalXY, aztype


def xy(lib, angle, x, y):
    norm = rho.xy(lib, angle, x, y)
    return (lib.nan_to_num(x / norm, nan=0), lib.nan_to_num(y / norm, nan=0))


def rhophi(lib, angle, rho, phi):
    return (1, phi)


dispatch_map = {
    (AzimuthalXY,): (xy, AzimuthalXY),
    (AzimuthalRhoPhi,): (rhophi, AzimuthalRhoPhi),
}


def dispatch(angle, v):
    function, *returns = dispatch_map[
        aztype(v),
    ]
    with numpy.errstate(all="ignore"):
        return v._wrap_result(function(v.lib, angle, *v.azimuthal.elements), returns)
