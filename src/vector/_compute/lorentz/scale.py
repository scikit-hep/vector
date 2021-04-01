# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Lorentz.scale(self, factor)
"""

import numpy

from vector._compute.spatial import scale as scale3d
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    TemporalT,
    TemporalTau,
    _aztype,
    _flavor_of,
    _from_signature,
    _ltype,
    _ttype,
)


def xy_z_t(lib, factor, x, y, z, t):
    sx, sy, sz = scale3d.xy_z(lib, factor, x, y, z)
    return (sx, sy, sz, t * factor)


def xy_z_tau(lib, factor, x, y, z, tau):
    sx, sy, sz = scale3d.xy_z(lib, factor, x, y, z)
    return (sx, sy, sz, tau * factor)


def xy_theta_t(lib, factor, x, y, theta, t):
    sx, sy, stheta = scale3d.xy_theta(lib, factor, x, y, theta)
    return (sx, sy, stheta, t * factor)


def xy_theta_tau(lib, factor, x, y, theta, tau):
    sx, sy, stheta = scale3d.xy_theta(lib, factor, x, y, theta)
    return (sx, sy, stheta, tau * factor)


def xy_eta_t(lib, factor, x, y, eta, t):
    sx, sy, seta = scale3d.xy_eta(lib, factor, x, y, eta)
    return (sx, sy, seta, t * factor)


def xy_eta_tau(lib, factor, x, y, eta, tau):
    sx, sy, seta = scale3d.xy_eta(lib, factor, x, y, eta)
    return (sx, sy, seta, tau * factor)


def rhophi_z_t(lib, factor, rho, phi, z, t):
    srho, sphi, sz = scale3d.rhophi_z(lib, factor, rho, phi, z)
    return (srho, sphi, sz, t * factor)


def rhophi_z_tau(lib, factor, rho, phi, z, tau):
    srho, sphi, sz = scale3d.rhophi_z(lib, factor, rho, phi, z)
    return (srho, sphi, sz, tau * factor)


def rhophi_theta_t(lib, factor, rho, phi, theta, t):
    srho, sphi, stheta = scale3d.rhophi_theta(lib, factor, rho, phi, theta)
    return (srho, sphi, stheta, t * factor)


def rhophi_theta_tau(lib, factor, rho, phi, theta, tau):
    srho, sphi, stheta = scale3d.rhophi_theta(lib, factor, rho, phi, theta)
    return (srho, sphi, stheta, tau * factor)


def rhophi_eta_t(lib, factor, rho, phi, eta, t):
    srho, sphi, seta = scale3d.rhophi_eta(lib, factor, rho, phi, eta)
    return (srho, sphi, seta, t * factor)


def rhophi_eta_tau(lib, factor, rho, phi, eta, tau):
    srho, sphi, seta = scale3d.rhophi_eta(lib, factor, rho, phi, eta)
    return (srho, sphi, seta, tau * factor)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT): (
        xy_z_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalZ, TemporalTau): (
        xy_z_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (AzimuthalXY, LongitudinalTheta, TemporalT): (
        xy_theta_t,
        AzimuthalXY,
        LongitudinalTheta,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalTheta, TemporalTau): (
        xy_theta_tau,
        AzimuthalXY,
        LongitudinalTheta,
        TemporalTau,
    ),
    (AzimuthalXY, LongitudinalEta, TemporalT): (
        xy_eta_t,
        AzimuthalXY,
        LongitudinalEta,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalEta, TemporalTau): (
        xy_eta_tau,
        AzimuthalXY,
        LongitudinalEta,
        TemporalTau,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalT): (
        rhophi_z_t,
        AzimuthalRhoPhi,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalTau): (
        rhophi_z_tau,
        AzimuthalRhoPhi,
        LongitudinalZ,
        TemporalTau,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalT): (
        rhophi_theta_t,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        TemporalT,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalTau): (
        rhophi_theta_tau,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        TemporalTau,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalT): (
        rhophi_eta_t,
        AzimuthalRhoPhi,
        LongitudinalEta,
        TemporalT,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalTau): (
        rhophi_eta_tau,
        AzimuthalRhoPhi,
        LongitudinalEta,
        TemporalTau,
    ),
}


def dispatch(factor: typing.Any, v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v),
            _ltype(v),
            _ttype(v),
        ),
    )
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(
                v.lib,
                factor,
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements
            ),
            returns,
            1,
        )
