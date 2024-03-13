# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Lorentz.unit(self)
"""

from __future__ import annotations

import typing
from math import inf

import numpy

from vector._compute.lorentz import tau2
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
    squared = tau2.xy_z_t(lib, x, y, z, t)
    norm = lib.sqrt(lib.absolute(squared))
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(z / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(t / norm, nan=0, posinf=inf, neginf=-inf),
    )


def xy_z_tau(lib, x, y, z, tau):
    norm = lib.absolute(tau)
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(z / norm, nan=0, posinf=inf, neginf=-inf),
        lib.copysign(1, tau),
    )


def xy_theta_t(lib, x, y, theta, t):
    squared = tau2.xy_theta_t(lib, x, y, theta, t)
    norm = lib.sqrt(lib.absolute(squared))
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        theta,
        lib.nan_to_num(t / norm, nan=0, posinf=inf, neginf=-inf),
    )


def xy_theta_tau(lib, x, y, theta, tau):
    norm = lib.absolute(tau)
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        theta,
        lib.copysign(1, tau),
    )


def xy_eta_t(lib, x, y, eta, t):
    squared = tau2.xy_eta_t(lib, x, y, eta, t)
    norm = lib.sqrt(lib.absolute(squared))
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        eta,
        lib.nan_to_num(t / norm, nan=0, posinf=inf, neginf=-inf),
    )


def xy_eta_tau(lib, x, y, eta, tau):
    norm = lib.absolute(tau)
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
        eta,
        lib.copysign(1, tau),
    )


def rhophi_z_t(lib, rho, phi, z, t):
    squared = tau2.rhophi_z_t(lib, rho, phi, z, t)
    norm = lib.sqrt(lib.absolute(squared))
    return (
        lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf),
        phi,
        lib.nan_to_num(z / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(t / norm, nan=0, posinf=inf, neginf=-inf),
    )


def rhophi_z_tau(lib, rho, phi, z, tau):
    norm = lib.absolute(tau)
    return (
        lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf),
        phi,
        lib.nan_to_num(z / norm, nan=0, posinf=inf, neginf=-inf),
        lib.copysign(1, tau),
    )


def rhophi_theta_t(lib, rho, phi, theta, t):
    squared = tau2.rhophi_theta_t(lib, rho, phi, theta, t)
    norm = lib.sqrt(lib.absolute(squared))
    return (
        lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf),
        phi,
        theta,
        lib.nan_to_num(t / norm, nan=0, posinf=inf, neginf=-inf),
    )


def rhophi_theta_tau(lib, rho, phi, theta, tau):
    norm = lib.absolute(tau)
    return (
        lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf),
        phi,
        theta,
        lib.copysign(1, tau),
    )


def rhophi_eta_t(lib, rho, phi, eta, t):
    squared = tau2.rhophi_eta_t(lib, rho, phi, eta, t)
    norm = lib.sqrt(lib.absolute(squared))
    return (
        lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf),
        phi,
        eta,
        lib.nan_to_num(t / norm, nan=0, posinf=inf, neginf=-inf),
    )


def rhophi_eta_tau(lib, rho, phi, eta, tau):
    norm = lib.absolute(tau)
    return (
        lib.nan_to_num(rho / norm, nan=0, posinf=inf, neginf=-inf),
        phi,
        eta,
        lib.copysign(1, tau),
    )


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
