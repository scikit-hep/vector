# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Lorentz.to_beta3(self)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.lorentz import t
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


def xy_z_t(lib, x, y, z, t):
    return (x / t, y / t, z / t)


def xy_z_tau(lib, x, y, z, tau):
    return xy_z_t(lib, x, y, z, t.xy_z_tau(lib, x, y, z, tau))


def xy_theta_t(lib, x, y, theta, t):
    return (x / t, y / t, theta)


def xy_theta_tau(lib, x, y, theta, tau):
    return xy_theta_t(lib, x, y, theta, t.xy_theta_tau(lib, x, y, theta, tau))


def xy_eta_t(lib, x, y, eta, t):
    return (x / t, y / t, eta)


def xy_eta_tau(lib, x, y, eta, tau):
    return xy_eta_t(lib, x, y, eta, t.xy_eta_tau(lib, x, y, eta, tau))


def rhophi_z_t(lib, rho, phi, z, t):
    return (rho / t, phi, z / t)


def rhophi_z_tau(lib, rho, phi, z, tau):
    return rhophi_z_t(lib, rho, phi, z, t.rhophi_z_tau(lib, rho, phi, z, tau))


def rhophi_theta_t(lib, rho, phi, theta, t):
    return (rho / t, phi, theta)


def rhophi_theta_tau(lib, rho, phi, theta, tau):
    return rhophi_theta_t(
        lib, rho, phi, theta, t.rhophi_theta_tau(lib, rho, phi, theta, tau)
    )


def rhophi_eta_t(lib, rho, phi, eta, t):
    return (rho / t, phi, eta)


def rhophi_eta_tau(lib, rho, phi, eta, tau):
    return rhophi_eta_t(lib, rho, phi, eta, t.rhophi_eta_tau(lib, rho, phi, eta, tau))


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT): (xy_z_t, AzimuthalXY, LongitudinalZ, None),
    (AzimuthalXY, LongitudinalZ, TemporalTau): (
        xy_z_tau,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, TemporalT): (
        xy_theta_t,
        AzimuthalXY,
        LongitudinalTheta,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, TemporalTau): (
        xy_theta_tau,
        AzimuthalXY,
        LongitudinalTheta,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, TemporalT): (
        xy_eta_t,
        AzimuthalXY,
        LongitudinalEta,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, TemporalTau): (
        xy_eta_tau,
        AzimuthalXY,
        LongitudinalEta,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalT): (
        rhophi_z_t,
        AzimuthalRhoPhi,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalTau): (
        rhophi_z_tau,
        AzimuthalRhoPhi,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalT): (
        rhophi_theta_t,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalTau): (
        rhophi_theta_tau,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalT): (
        rhophi_eta_t,
        AzimuthalRhoPhi,
        LongitudinalEta,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalTau): (
        rhophi_eta_tau,
        AzimuthalRhoPhi,
        LongitudinalEta,
        None,
    ),
}


def dispatch(v: typing.Any) -> typing.Any:
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
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements,
            ),
            returns,
            1,
        )
