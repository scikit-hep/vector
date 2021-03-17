# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.planar import x, y
from vector.methods import AzimuthalRhoPhi, AzimuthalXY, _aztype


def xy_xy(lib, x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def xy_rhophi(lib, x1, y1, rho2, phi2):
    return x1 * x.rhophi(lib, rho2, phi2) + y1 * y.rhophi(lib, rho2, phi2)


def rhophi_xy(lib, rho1, phi1, x2, y2):
    return x.rhophi(lib, rho1, phi1) * x2 + y.rhophi(lib, rho1, phi1) * y2


def rhophi_rhophi(lib, rho1, phi1, rho2, phi2):
    return rho1 * rho2 * lib.cos(phi1 - phi2)


dispatch_map = {
    (AzimuthalXY, AzimuthalXY): (xy_xy, float),
    (AzimuthalXY, AzimuthalRhoPhi): (xy_rhophi, float),
    (AzimuthalRhoPhi, AzimuthalXY): (rhophi_xy, float),
    (AzimuthalRhoPhi, AzimuthalRhoPhi): (rhophi_rhophi, float),
}


def dispatch(v1, v2):
    if v1.lib is not v2.lib:
        raise TypeError(
            f"cannot use {v1} (requires {v1.lib}) and {v2} (requires {v1.lib}) together"
        )
    function, *returns = dispatch_map[
        _aztype(v1),
        _aztype(v2),
    ]
    with numpy.errstate(all="ignore"):
        return v1._wrap_result(
            function(v1.lib, *v1.azimuthal.elements, *v2.azimuthal.elements), returns
        )
