# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.isclose(self, other, rtol=..., atol=..., equal_nan=...)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.planar import x, y
from vector._compute.spatial import eta, z
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _flavor_of,
    _from_signature,
    _handler_of,
    _lib_of,
    _ltype,
)

# Policy: turn (rho, phi) into (x, y)
#         turn theta into z, eta into z, theta into eta
#         (if not already the same)


# same types
def xy_z_xy_z(lib, rtol, atol, equal_nan, x1, y1, z1, x2, y2, z2):
    return (
        lib.isclose(x1, x2, rtol, atol, equal_nan)
        & lib.isclose(y1, y2, rtol, atol, equal_nan)
        & lib.isclose(z1, z2, rtol, atol, equal_nan)
    )


def xy_z_xy_theta(lib, rtol, atol, equal_nan, x1, y1, z1, x2, y2, theta2):
    return xy_z_xy_z(
        lib, rtol, atol, equal_nan, x1, y1, z1, x2, y2, z.xy_theta(lib, x2, y2, theta2)
    )


def xy_z_xy_eta(lib, rtol, atol, equal_nan, x1, y1, z1, x2, y2, eta2):
    return xy_z_xy_z(
        lib, rtol, atol, equal_nan, x1, y1, z1, x2, y2, z.xy_eta(lib, x2, y2, eta2)
    )


def xy_z_rhophi_z(lib, rtol, atol, equal_nan, x1, y1, z1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def xy_z_rhophi_theta(lib, rtol, atol, equal_nan, x1, y1, z1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_z_rhophi_eta(lib, rtol, atol, equal_nan, x1, y1, z1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def xy_theta_xy_z(lib, rtol, atol, equal_nan, x1, y1, theta1, x2, y2, z2):
    return xy_z_xy_z(
        lib, rtol, atol, equal_nan, x1, y1, z.xy_theta(lib, x1, y1, theta1), x2, y2, z2
    )


# same types
def xy_theta_xy_theta(lib, rtol, atol, equal_nan, x1, y1, theta1, x2, y2, theta2):
    return (
        lib.isclose(x1, x2, rtol, atol, equal_nan)
        & lib.isclose(y1, y2, rtol, atol, equal_nan)
        & lib.isclose(theta1, theta2, rtol, atol, equal_nan)
    )


def xy_theta_xy_eta(lib, rtol, atol, equal_nan, x1, y1, theta1, x2, y2, eta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        eta.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        eta2,
    )


def xy_theta_rhophi_z(lib, rtol, atol, equal_nan, x1, y1, theta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def xy_theta_rhophi_theta(
    lib, rtol, atol, equal_nan, x1, y1, theta1, rho2, phi2, theta2
):
    return xy_theta_xy_theta(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        theta1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        theta2,
    )


def xy_theta_rhophi_eta(lib, rtol, atol, equal_nan, x1, y1, theta1, rho2, phi2, eta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        eta.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        eta2,
    )


def xy_eta_xy_z(lib, rtol, atol, equal_nan, x1, y1, eta1, x2, y2, z2):
    return xy_z_xy_z(
        lib, rtol, atol, equal_nan, x1, y1, z.xy_eta(lib, x1, y1, eta1), x2, y2, z2
    )


def xy_eta_xy_theta(lib, rtol, atol, equal_nan, x1, y1, eta1, x2, y2, theta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        eta1,
        x2,
        y2,
        eta.xy_theta(lib, x2, y2, theta2),
    )


# same types
def xy_eta_xy_eta(lib, rtol, atol, equal_nan, x1, y1, eta1, x2, y2, eta2):
    return (
        lib.isclose(x1, x2, rtol, atol, equal_nan)
        & lib.isclose(y1, y2, rtol, atol, equal_nan)
        & lib.isclose(eta1, eta2, rtol, atol, equal_nan)
    )


def xy_eta_rhophi_z(lib, rtol, atol, equal_nan, x1, y1, eta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def xy_eta_rhophi_theta(lib, rtol, atol, equal_nan, x1, y1, eta1, rho2, phi2, theta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        eta1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        eta.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_eta_rhophi_eta(lib, rtol, atol, equal_nan, x1, y1, eta1, rho2, phi2, eta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x1,
        y1,
        eta1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        eta2,
    )


def rhophi_z_xy_z(lib, rtol, atol, equal_nan, rho1, phi1, z1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x2,
        y2,
        z2,
    )


def rhophi_z_xy_theta(lib, rtol, atol, equal_nan, rho1, phi1, z1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_z_xy_eta(lib, rtol, atol, equal_nan, rho1, phi1, z1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x2,
        y2,
        z.xy_eta(lib, x2, y2, eta2),
    )


# same types
def rhophi_z_rhophi_z(lib, rtol, atol, equal_nan, rho1, phi1, z1, rho2, phi2, z2):
    return (
        lib.isclose(rho1, rho2, rtol, atol, equal_nan)
        & lib.isclose(phi1, phi2, rtol, atol, equal_nan)
        & lib.isclose(z1, z2, rtol, atol, equal_nan)
    )


def rhophi_z_rhophi_theta(
    lib, rtol, atol, equal_nan, rho1, phi1, z1, rho2, phi2, theta2
):
    return rhophi_z_rhophi_z(
        lib,
        rtol,
        atol,
        equal_nan,
        rho1,
        phi1,
        z1,
        rho2,
        phi2,
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def rhophi_z_rhophi_eta(lib, rtol, atol, equal_nan, rho1, phi1, z1, rho2, phi2, eta2):
    return rhophi_z_rhophi_z(
        lib,
        rtol,
        atol,
        equal_nan,
        rho1,
        phi1,
        z1,
        rho2,
        phi2,
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_theta_xy_z(lib, rtol, atol, equal_nan, rho1, phi1, theta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x2,
        y2,
        z2,
    )


def rhophi_theta_xy_theta(
    lib, rtol, atol, equal_nan, rho1, phi1, theta1, x2, y2, theta2
):
    return xy_theta_xy_theta(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        theta1,
        x2,
        y2,
        theta2,
    )


def rhophi_theta_xy_eta(lib, rtol, atol, equal_nan, rho1, phi1, theta1, x2, y2, eta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        eta.rhophi_theta(lib, rho1, phi1, theta1),
        x2,
        y2,
        eta2,
    )


def rhophi_theta_rhophi_z(
    lib, rtol, atol, equal_nan, rho1, phi1, theta1, rho2, phi2, z2
):
    return rhophi_z_rhophi_z(
        lib,
        rtol,
        atol,
        equal_nan,
        rho1,
        phi1,
        z.rhophi_theta(lib, rho1, phi1, theta1),
        rho2,
        phi2,
        z2,
    )


# same types
def rhophi_theta_rhophi_theta(
    lib, rtol, atol, equal_nan, rho1, phi1, theta1, rho2, phi2, theta2
):
    return (
        lib.isclose(rho1, rho2, rtol, atol, equal_nan)
        & lib.isclose(phi1, phi2, rtol, atol, equal_nan)
        & lib.isclose(theta1, theta2, rtol, atol, equal_nan)
    )


def rhophi_theta_rhophi_eta(
    lib, rtol, atol, equal_nan, rho1, phi1, theta1, rho2, phi2, eta2
):
    return rhophi_eta_rhophi_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        rho1,
        phi1,
        eta.rhophi_theta(lib, rho1, phi1, theta1),
        rho2,
        phi2,
        eta2,
    )


def rhophi_eta_xy_z(lib, rtol, atol, equal_nan, rho1, phi1, eta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x2,
        y2,
        z2,
    )


def rhophi_eta_xy_theta(lib, rtol, atol, equal_nan, rho1, phi1, eta1, x2, y2, theta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        eta1,
        x2,
        y2,
        eta.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_eta_xy_eta(lib, rtol, atol, equal_nan, rho1, phi1, eta1, x2, y2, eta2):
    return xy_eta_xy_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        eta1,
        x2,
        y2,
        eta2,
    )


def rhophi_eta_rhophi_z(lib, rtol, atol, equal_nan, rho1, phi1, eta1, rho2, phi2, z2):
    return rhophi_z_rhophi_z(
        lib,
        rtol,
        atol,
        equal_nan,
        rho1,
        phi1,
        z.rhophi_eta(lib, rho1, phi1, eta1),
        rho2,
        phi2,
        z2,
    )


def rhophi_eta_rhophi_theta(
    lib, rtol, atol, equal_nan, rho1, phi1, eta1, rho2, phi2, theta2
):
    return rhophi_eta_rhophi_eta(
        lib,
        rtol,
        atol,
        equal_nan,
        rho1,
        phi1,
        eta1,
        rho2,
        phi2,
        eta.rhophi_theta(lib, rho2, phi2, theta2),
    )


# same types
def rhophi_eta_rhophi_eta(
    lib, rtol, atol, equal_nan, rho1, phi1, eta1, rho2, phi2, eta2
):
    return (
        lib.isclose(rho1, rho2, rtol, atol, equal_nan)
        & lib.isclose(phi1, phi2, rtol, atol, equal_nan)
        & lib.isclose(eta1, eta2, rtol, atol, equal_nan)
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ): (xy_z_xy_z, bool),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        xy_z_xy_theta,
        bool,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalEta): (xy_z_xy_eta, bool),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        xy_z_rhophi_z,
        bool,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_z_rhophi_theta,
        bool,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        xy_z_rhophi_eta,
        bool,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        xy_theta_xy_z,
        bool,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        xy_theta_xy_theta,
        bool,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        xy_theta_xy_eta,
        bool,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_theta_rhophi_z,
        bool,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_theta_rhophi_theta,
        bool,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_theta_rhophi_eta,
        bool,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalZ): (xy_eta_xy_z, bool),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        xy_eta_xy_theta,
        bool,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        xy_eta_xy_eta,
        bool,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_eta_rhophi_z,
        bool,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_eta_rhophi_theta,
        bool,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_eta_rhophi_eta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        rhophi_z_xy_z,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        rhophi_z_xy_theta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalEta): (
        rhophi_z_xy_eta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_z_rhophi_z,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_z_rhophi_theta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_z_rhophi_eta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        rhophi_theta_xy_z,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        rhophi_theta_xy_theta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        rhophi_theta_xy_eta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_theta_rhophi_z,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta_rhophi_theta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_theta_rhophi_eta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalZ): (
        rhophi_eta_xy_z,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        rhophi_eta_xy_theta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        rhophi_eta_xy_eta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_eta_rhophi_z,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_eta_rhophi_theta,
        bool,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_eta_rhophi_eta,
        bool,
    ),
}


def dispatch(
    rtol: typing.Any,
    atol: typing.Any,
    equal_nan: typing.Any,
    v1: typing.Any,
    v2: typing.Any,
) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v1),
            _ltype(v1),
            _aztype(v2),
            _ltype(v2),
        ),
    )
    with numpy.errstate(all="ignore"):
        return _handler_of(v1, v2)._wrap_result(
            _flavor_of(v1, v2),
            function(
                _lib_of(v1, v2),
                rtol,
                atol,
                equal_nan,
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements,
            ),
            returns,
            2,
        )
