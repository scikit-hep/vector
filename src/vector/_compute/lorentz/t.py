# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    @property
    Lorentz.t(self)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.lorentz import t2
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
    return t


xy_z_t.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


def xy_z_tau(lib, x, y, z, tau):
    return lib.sqrt(t2.xy_z_tau(lib, x, y, z, tau))


def xy_theta_t(lib, x, y, theta, t):
    return t


xy_theta_t.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


def xy_theta_tau(lib, x, y, theta, tau):
    return lib.sqrt(t2.xy_theta_tau(lib, x, y, theta, tau))


def xy_eta_t(lib, x, y, eta, t):
    return t


xy_eta_t.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


def xy_eta_tau(lib, x, y, eta, tau):
    return lib.sqrt(t2.xy_eta_tau(lib, x, y, eta, tau))


def rhophi_z_t(lib, rho, phi, z, t):
    return t


rhophi_z_t.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


def rhophi_z_tau(lib, rho, phi, z, tau):
    return lib.sqrt(t2.rhophi_z_tau(lib, rho, phi, z, tau))


def rhophi_theta_t(lib, rho, phi, theta, t):
    return t


rhophi_theta_t.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


def rhophi_theta_tau(lib, rho, phi, theta, tau):
    return lib.sqrt(t2.rhophi_theta_tau(lib, rho, phi, theta, tau))


def rhophi_eta_t(lib, rho, phi, eta, t):
    return t


rhophi_eta_t.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


def rhophi_eta_tau(lib, rho, phi, eta, tau):
    return lib.sqrt(t2.rhophi_eta_tau(lib, rho, phi, eta, tau))


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT): (xy_z_t, float),
    (AzimuthalXY, LongitudinalZ, TemporalTau): (xy_z_tau, float),
    (AzimuthalXY, LongitudinalTheta, TemporalT): (xy_theta_t, float),
    (AzimuthalXY, LongitudinalTheta, TemporalTau): (xy_theta_tau, float),
    (AzimuthalXY, LongitudinalEta, TemporalT): (xy_eta_t, float),
    (AzimuthalXY, LongitudinalEta, TemporalTau): (xy_eta_tau, float),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalT): (rhophi_z_t, float),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalTau): (rhophi_z_tau, float),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalT): (rhophi_theta_t, float),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalTau): (rhophi_theta_tau, float),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalT): (rhophi_eta_t, float),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalTau): (rhophi_eta_tau, float),
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
            v._wrap_dispatched_function(function)(
                v.lib,
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements,
            ),
            returns,
            1,
        )
