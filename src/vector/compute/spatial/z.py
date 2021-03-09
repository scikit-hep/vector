# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.planar import rho
from vector.geometry import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    aztype,
    ltype,
)


def xy_z(lib, x, y, z):
    return z


def xy_theta(lib, x, y, theta):
    return lib.nan_to_num(rho.xy(lib, x, y) / lib.tan(theta), 0.0)


def xy_eta(lib, x, y, eta):
    return rho.xy(lib, x, y) * lib.sinh(eta)


def rhophi_z(lib, rho, phi, z):
    return z


def rhophi_theta(lib, rho, phi, theta):
    return lib.nan_to_num(rho / lib.tan(theta), 0.0)


def rhophi_eta(lib, rho, phi, eta):
    return rho * lib.sinh(eta)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, float),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, float),
    (AzimuthalXY, LongitudinalEta): (xy_eta, float),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, float),
    (AzimuthalRhoPhi, LongitudinalTheta): (rhophi_theta, float),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, float),
}


def dispatch(v):
    function, *returns = dispatch_map[
        aztype(v),
        ltype(v),
    ]
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            function(v.lib, *v.azimuthal.elements, *v.longitudinal.elements), returns
        )
