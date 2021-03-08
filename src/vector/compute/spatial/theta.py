# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.spatial import costheta
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
    return lib.arccos(costheta.xy_z(lib, x, y, z))


def xy_theta(lib, x, y, theta):
    return theta


def xy_eta(lib, x, y, eta):
    return 2.0 * lib.arctan(lib.exp(-eta))


def rhophi_z(lib, rho, phi, z):
    return lib.arccos(costheta.rhophi_z(lib, rho, phi, z))


def rhophi_theta(lib, rho, phi, theta):
    return theta


def rhophi_eta(lib, rho, phi, eta):
    return 2.0 * lib.arctan(lib.exp(-eta))


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
