# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

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
    return x ** 2 + y ** 2 + z ** 2


def xy_theta(lib, x, y, theta):
    return (x ** 2 + y ** 2) * (1.0 + 1.0 / lib.tan(theta) ** 2)


def xy_eta(lib, x, y, eta):
    return (x ** 2 + y ** 2) * (
        1.0 + lib.nan_to_num((1.0 - lib.exp(-2.0 * eta)) / (2.0 * lib.exp(-eta)), lib.inf) ** 2
    )


def rhophi_z(lib, rho, phi, z):
    return rho ** 2 + z ** 2


def rhophi_theta(lib, rho, phi, theta):
    return rho ** 2 * (1.0 + 1.0 / lib.tan(theta) ** 2)


def rhophi_eta(lib, rho, phi, eta):
    return rho ** 2 * (1.0 + lib.nan_to_num((1.0 - lib.exp(-2.0 * eta)) / (2.0 * lib.exp(-eta)), lib.inf) ** 2)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): xy_z,
    (AzimuthalXY, LongitudinalTheta): xy_theta,
    (AzimuthalXY, LongitudinalEta): xy_eta,
    (AzimuthalRhoPhi, LongitudinalZ): rhophi_z,
    (AzimuthalRhoPhi, LongitudinalTheta): rhophi_theta,
    (AzimuthalRhoPhi, LongitudinalEta): rhophi_eta,
}


def dispatch(v):
    with numpy.errstate(all="ignore"):
        return dispatch_map[
                aztype(v),
                ltype(v),
            ](v.lib, *v.azimuthal.elements, *v.longitudinal.elements)
