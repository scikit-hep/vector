# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    @property
    Lorentz.beta(self)
"""

import numpy

from vector._compute.lorentz import t
from vector._compute.spatial import mag
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
    return lib.nan_to_num(mag.xy_z(lib, x, y, z) / t, nan=0)


def xy_z_tau(lib, x, y, z, tau):
    return lib.nan_to_num(mag.xy_z(lib, x, y, z) / t.xy_z_tau(lib, x, y, z, tau), nan=0)


def xy_theta_t(lib, x, y, theta, t):
    return lib.nan_to_num(mag.xy_theta(lib, x, y, theta) / t, nan=0)


def xy_theta_tau(lib, x, y, theta, tau):
    return lib.nan_to_num(
        mag.xy_theta(lib, x, y, theta) / t.xy_theta_tau(lib, x, y, theta, tau), nan=0
    )


def xy_eta_t(lib, x, y, eta, t):
    return lib.nan_to_num(mag.xy_eta(lib, x, y, eta) / t, nan=0)


def xy_eta_tau(lib, x, y, eta, tau):
    return lib.nan_to_num(
        mag.xy_eta(lib, x, y, eta) / t.xy_eta_tau(lib, x, y, eta, tau), nan=0
    )


def rhophi_z_t(lib, rho, phi, z, t):
    return lib.nan_to_num(mag.rhophi_z(lib, rho, phi, z) / t, nan=0)


def rhophi_z_tau(lib, rho, phi, z, tau):
    return lib.nan_to_num(
        mag.rhophi_z(lib, rho, phi, z) / t.rhophi_z_tau(lib, rho, phi, z, tau), nan=0
    )


def rhophi_theta_t(lib, rho, phi, theta, t):
    return lib.nan_to_num(mag.rhophi_theta(lib, rho, phi, theta) / t, nan=0)


def rhophi_theta_tau(lib, rho, phi, theta, tau):
    return lib.nan_to_num(
        mag.rhophi_theta(lib, rho, phi, theta)
        / t.rhophi_theta_tau(lib, rho, phi, theta, tau),
        nan=0,
    )


def rhophi_eta_t(lib, rho, phi, eta, t):
    return lib.nan_to_num(mag.rhophi_eta(lib, rho, phi, eta) / t, nan=0)


def rhophi_eta_tau(lib, rho, phi, eta, tau):
    return lib.nan_to_num(
        mag.rhophi_eta(lib, rho, phi, eta) / t.rhophi_eta_tau(lib, rho, phi, eta, tau),
        nan=0,
    )


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
            function(
                v.lib,
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements
            ),
            returns,
            1,
        )
