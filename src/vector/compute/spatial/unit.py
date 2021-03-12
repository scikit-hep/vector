# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.spatial import mag
from vector.geometry import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    aztype,
    ltype,
)


def xy_z(lib, angle, x, y, z):
    norm = mag.xy_z(lib, angle, x, y, z)
    return (
        lib.nan_to_num(x / norm, nan=0),
        lib.nan_to_num(y / norm, nan=0),
        lib.nan_to_num(z / norm, nan=0),
    )


def xy_theta(lib, angle, x, y, theta):
    norm = mag.xy_theta(lib, angle, x, y, theta)
    return (lib.nan_to_num(x / norm, nan=0), lib.nan_to_num(y / norm, nan=0), theta)


def xy_eta(lib, angle, x, y, eta):
    norm = mag.xy_eta(lib, angle, x, y, eta)
    return (lib.nan_to_num(x / norm, nan=0), lib.nan_to_num(y / norm, nan=0), eta)


def rhophi_z(lib, angle, rho, phi, z):
    norm = rhophi_z(lib, angle, rho, phi, z)
    return (lib.nan_to_num(rho / norm, nan=0), phi, lib.nan_to_num(z / norm, nan=0))


def rhophi_theta(lib, angle, rho, phi, theta):
    norm = rhophi_theta(lib, angle, rho, phi, theta)
    return (lib.nan_to_num(rho / norm, nan=0), phi, theta)


def rhophi_eta(lib, angle, rho, phi, eta):
    norm = rhophi_eta(lib, angle, rho, phi, eta)
    return (lib.nan_to_num(rho / norm, nan=0), phi, eta)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, AzimuthalXY, LongitudinalTheta),
    (AzimuthalXY, LongitudinalEta): (xy_eta, AzimuthalXY, LongitudinalEta),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, AzimuthalRhoPhi, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta,
        AzimuthalRhoPhi,
        LongitudinalTheta,
    ),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, AzimuthalRhoPhi, LongitudinalEta),
}


def dispatch(angle, v):
    function, *returns = dispatch_map[
        aztype(v),
        ltype(v),
    ]
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            function(v.lib, angle, *v.azimuthal.elements, *v.longitudinal.elements),
            returns,
        )
