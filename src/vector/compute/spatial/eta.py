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
    return lib.nan_to_num(lib.arctanh(z / lib.sqrt(x ** 2 + y ** 2 + z ** 2)), 0.0)


def xy_theta(lib, x, y, theta):
    return lib.nan_to_num(-lib.log(lib.tan(0.5 * theta)), 0.0)


def xy_eta(lib, x, y, eta):
    return eta


def rhophi_z(lib, rho, phi, z):
    return lib.nan_to_num(lib.arctanh(z / lib.sqrt(rho ** 2 + z ** 2)), 0.0)


def rhophi_theta(lib, rho, phi, theta):
    return -lib.log(lib.tan(0.5 * theta))


def rhophi_eta(lib, rho, phi, eta):
    return eta


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
