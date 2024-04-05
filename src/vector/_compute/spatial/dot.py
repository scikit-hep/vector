# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.dot(self, other)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.planar import x, y
from vector._compute.spatial import theta, z
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


# specialized
def xy_z_xy_z(lib, x1, y1, z1, x2, y2, z2):
    return x1 * x2 + y1 * y2 + z1 * z2


def xy_z_xy_theta(lib, x1, y1, z1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_z(lib, x1, y1, z1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_theta(lib, x2, y2, theta2),
    )


def xy_z_xy_eta(lib, x1, y1, z1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_z(lib, x1, y1, z1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_eta(lib, x2, y2, eta2),
    )


def xy_z_rhophi_z(lib, x1, y1, z1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_z(lib, x1, y1, z1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_z(lib, rho2, phi2, z2),
    )


def xy_z_rhophi_theta(lib, x1, y1, z1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_z(lib, x1, y1, z1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_z_rhophi_eta(lib, x1, y1, z1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_z(lib, x1, y1, z1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def xy_theta_xy_z(lib, x1, y1, theta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_theta(lib, x1, y1, theta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_z(lib, x2, y2, z2),
    )


def xy_theta_xy_theta(lib, x1, y1, theta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_theta(lib, x1, y1, theta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_theta(lib, x2, y2, theta2),
    )


def xy_theta_xy_eta(lib, x1, y1, theta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_theta(lib, x1, y1, theta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_eta(lib, x2, y2, eta2),
    )


def xy_theta_rhophi_z(lib, x1, y1, theta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_z(lib, rho2, phi2, z2),
    )


def xy_theta_rhophi_theta(lib, x1, y1, theta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_theta_rhophi_eta(lib, x1, y1, theta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def xy_eta_xy_z(lib, x1, y1, eta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_eta(lib, x1, y1, eta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_z(lib, x2, y2, z2),
    )


def xy_eta_xy_theta(lib, x1, y1, eta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_eta(lib, x1, y1, eta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_theta(lib, x2, y2, theta2),
    )


def xy_eta_xy_eta(lib, x1, y1, eta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_eta(lib, x1, y1, eta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_eta(lib, x2, y2, eta2),
    )


def xy_eta_rhophi_z(lib, x1, y1, eta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_z(lib, rho2, phi2, z2),
    )


def xy_eta_rhophi_theta(lib, x1, y1, eta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_eta_rhophi_eta(lib, x1, y1, eta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x.xy(lib, x1, y1),
        y.xy(lib, x1, y1),
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_z_xy_z(lib, rho1, phi1, z1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_z(lib, rho1, phi1, z1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_z(lib, x2, y2, z2),
    )


def rhophi_z_xy_theta(lib, rho1, phi1, z1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_z(lib, rho1, phi1, z1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_z_xy_eta(lib, rho1, phi1, z1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_z(lib, rho1, phi1, z1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_eta(lib, x2, y2, eta2),
    )


# specialized
def rhophi_z_rhophi_z(lib, rho1, phi1, z1, rho2, phi2, z2):
    return rho1 * rho2 * lib.cos(phi1 - phi2) + z1 * z2


def rhophi_z_rhophi_theta(lib, rho1, phi1, z1, rho2, phi2, theta2):
    return rhophi_z_rhophi_z(
        lib, rho1, phi1, z1, rho2, phi2, z.rhophi_theta(lib, rho2, phi2, theta2)
    )


def rhophi_z_rhophi_eta(lib, rho1, phi1, z1, rho2, phi2, eta2):
    return rhophi_z_rhophi_z(
        lib, rho1, phi1, z1, rho2, phi2, z.rhophi_eta(lib, rho2, phi2, eta2)
    )


def rhophi_theta_xy_z(lib, rho1, phi1, theta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_z(lib, x2, y2, z2),
    )


def rhophi_theta_xy_theta(lib, rho1, phi1, theta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_theta_xy_eta(lib, rho1, phi1, theta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_eta(lib, x2, y2, eta2),
    )


def rhophi_theta_rhophi_z(lib, rho1, phi1, theta1, rho2, phi2, z2):
    return rhophi_z_rhophi_z(
        lib, rho1, phi1, z.rhophi_theta(lib, rho1, phi1, theta1), rho2, phi2, z2
    )


# specialized
def rhophi_theta_rhophi_theta(lib, rho1, phi1, theta1, rho2, phi2, theta2):
    return (
        rho1 * rho2 * (lib.cos(phi1 - phi2) + 1 / (lib.tan(theta1) * lib.tan(theta2)))
    )


def rhophi_theta_rhophi_eta(lib, rho1, phi1, theta1, rho2, phi2, eta2):
    return rhophi_theta_rhophi_theta(
        lib, rho1, phi1, theta1, rho2, phi2, theta.rhophi_eta(lib, rho2, phi2, eta2)
    )


def rhophi_eta_xy_z(lib, rho1, phi1, eta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_z(lib, x2, y2, z2),
    )


def rhophi_eta_xy_theta(lib, rho1, phi1, eta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_eta_xy_eta(lib, rho1, phi1, eta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x.xy(lib, x2, y2),
        y.xy(lib, x2, y2),
        z.xy_eta(lib, x2, y2, eta2),
    )


def rhophi_eta_rhophi_z(lib, rho1, phi1, eta1, rho2, phi2, z2):
    return rhophi_theta_rhophi_theta(
        lib,
        rho1,
        phi1,
        theta.rhophi_eta(lib, rho1, phi1, eta1),
        rho2,
        phi2,
        theta.rhophi_z(lib, rho2, phi2, z2),
    )


def rhophi_eta_rhophi_theta(lib, rho1, phi1, eta1, rho2, phi2, theta2):
    return rhophi_theta_rhophi_theta(
        lib, rho1, phi1, theta.rhophi_eta(lib, rho1, phi1, eta1), rho2, phi2, theta2
    )


# specialized
def rhophi_eta_rhophi_eta(lib, rho1, phi1, eta1, rho2, phi2, eta2):
    expmeta1 = lib.exp(-eta1)
    expmeta2 = lib.exp(-eta2)
    invtantheta1 = 0.5 * (1 - expmeta1**2) / expmeta1
    invtantheta2 = 0.5 * (1 - expmeta2**2) / expmeta2
    return rho1 * rho2 * (lib.cos(phi1 - phi2) + invtantheta1 * invtantheta2)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ): (xy_z_xy_z, float),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        xy_z_xy_theta,
        float,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalEta): (xy_z_xy_eta, float),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        xy_z_rhophi_z,
        float,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_z_rhophi_theta,
        float,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        xy_z_rhophi_eta,
        float,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        xy_theta_xy_z,
        float,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        xy_theta_xy_theta,
        float,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        xy_theta_xy_eta,
        float,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_theta_rhophi_z,
        float,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_theta_rhophi_theta,
        float,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_theta_rhophi_eta,
        float,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalZ): (xy_eta_xy_z, float),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        xy_eta_xy_theta,
        float,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        xy_eta_xy_eta,
        float,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_eta_rhophi_z,
        float,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_eta_rhophi_theta,
        float,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_eta_rhophi_eta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        rhophi_z_xy_z,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        rhophi_z_xy_theta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalEta): (
        rhophi_z_xy_eta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_z_rhophi_z,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_z_rhophi_theta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_z_rhophi_eta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        rhophi_theta_xy_z,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        rhophi_theta_xy_theta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        rhophi_theta_xy_eta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_theta_rhophi_z,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta_rhophi_theta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_theta_rhophi_eta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalZ): (
        rhophi_eta_xy_z,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        rhophi_eta_xy_theta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        rhophi_eta_xy_eta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_eta_rhophi_z,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_eta_rhophi_theta,
        float,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_eta_rhophi_eta,
        float,
    ),
}


def dispatch(v1: typing.Any, v2: typing.Any) -> typing.Any:
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
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements,
            ),
            returns,
            2,
        )
