# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Lorentz.boostX(self, beta=...)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.lorentz import t
from vector._compute.planar import x, y
from vector._compute.spatial import z
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


def xy_z_t(lib, beta, x, y, z, t):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x
    why = y
    zee = z
    tee = t
    return (gam * exx + bgam * tee, why, zee, bgam * exx + gam * tee)


def xy_z_tau(lib, beta, x, y, z, tau):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x
    why = y
    zee = z
    tee = t.xy_z_tau(lib, x, y, z, tau)
    return (gam * exx + bgam * tee, why, zee, tau)


def xy_theta_t(lib, beta, x, y, theta, t):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x
    why = y
    zee = z.xy_theta(lib, x, y, theta)
    tee = t
    return (gam * exx + bgam * tee, why, zee, bgam * exx + gam * tee)


def xy_theta_tau(lib, beta, x, y, theta, tau):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x
    why = y
    zee = z.xy_theta(lib, x, y, theta)
    tee = t.xy_theta_tau(lib, x, y, theta, tau)
    return (gam * exx + bgam * tee, why, zee, tau)


def xy_eta_t(lib, beta, x, y, eta, t):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x
    why = y
    zee = z.xy_eta(lib, x, y, eta)
    tee = t
    return (gam * exx + bgam * tee, why, zee, bgam * exx + gam * tee)


def xy_eta_tau(lib, beta, x, y, eta, tau):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x
    why = y
    zee = z.xy_eta(lib, x, y, eta)
    tee = t.xy_eta_tau(lib, x, y, eta, tau)
    return (gam * exx + bgam * tee, why, zee, tau)


def rhophi_z_t(lib, beta, rho, phi, z, t):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x.rhophi(lib, rho, phi)
    why = y.rhophi(lib, rho, phi)
    zee = z
    tee = t
    return (gam * exx + bgam * tee, why, zee, bgam * exx + gam * tee)


def rhophi_z_tau(lib, beta, rho, phi, z, tau):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x.rhophi(lib, rho, phi)
    why = y.rhophi(lib, rho, phi)
    zee = z
    tee = t.rhophi_z_tau(lib, rho, phi, z, tau)
    return (gam * exx + bgam * tee, why, zee, tau)


def rhophi_theta_t(lib, beta, rho, phi, theta, t):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x.rhophi(lib, rho, phi)
    why = y.rhophi(lib, rho, phi)
    zee = z.rhophi_theta(lib, rho, phi, theta)
    tee = t
    return (gam * exx + bgam * tee, why, zee, bgam * exx + gam * tee)


def rhophi_theta_tau(lib, beta, rho, phi, theta, tau):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x.rhophi(lib, rho, phi)
    why = y.rhophi(lib, rho, phi)
    zee = z.rhophi_theta(lib, rho, phi, theta)
    tee = t.rhophi_theta_tau(lib, rho, phi, theta, tau)
    return (gam * exx + bgam * tee, why, zee, tau)


def rhophi_eta_t(lib, beta, rho, phi, eta, t):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x.rhophi(lib, rho, phi)
    why = y.rhophi(lib, rho, phi)
    zee = z.rhophi_eta(lib, rho, phi, eta)
    tee = t
    return (gam * exx + bgam * tee, why, zee, bgam * exx + gam * tee)


def rhophi_eta_tau(lib, beta, rho, phi, eta, tau):
    gam = (1 - beta**2) ** -0.5
    bgam = beta * gam
    exx = x.rhophi(lib, rho, phi)
    why = y.rhophi(lib, rho, phi)
    zee = z.rhophi_eta(lib, rho, phi, eta)
    tee = t.rhophi_eta_tau(lib, rho, phi, eta, tau)
    return (gam * exx + bgam * tee, why, zee, tau)


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
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalTheta, TemporalTau): (
        xy_theta_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (AzimuthalXY, LongitudinalEta, TemporalT): (
        xy_eta_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalEta, TemporalTau): (
        xy_eta_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalT): (
        rhophi_z_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalTau): (
        rhophi_z_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalT): (
        rhophi_theta_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalTau): (
        rhophi_theta_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalT): (
        rhophi_eta_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalTau): (
        rhophi_eta_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
}


def dispatch(beta: typing.Any, v: typing.Any) -> typing.Any:
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
                beta,
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements,
            ),
            returns,
            1,
        )
