# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    @property
    Lorentz.Mt2(self)
"""

import numpy

from vector._compute.lorentz import tau2
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


def xy_z_t(lib, x, y, z, t):
    return t ** 2 - z ** 2


def xy_z_tau(lib, x, y, z, tau):
    return lib.maximum(tau2.xy_z_tau(lib, x, y, z, tau) + x ** 2 + y ** 2, 0)


def xy_theta_t(lib, x, y, theta, t):
    return xy_z_t(lib, x, y, z.xy_theta(lib, x, y, theta), t)


def xy_theta_tau(lib, x, y, theta, tau):
    return lib.maximum(tau2.xy_theta_tau(lib, x, y, theta, tau) + x ** 2 + y ** 2, 0)


def xy_eta_t(lib, x, y, eta, t):
    return xy_z_t(lib, x, y, z.xy_eta(lib, x, y, eta), t)


def xy_eta_tau(lib, x, y, eta, tau):
    return lib.maximum(tau2.xy_eta_tau(lib, x, y, eta, tau) + x ** 2 + y ** 2, 0)


def rhophi_z_t(lib, rho, phi, z, t):
    return t ** 2 - z ** 2


def rhophi_z_tau(lib, rho, phi, z, tau):
    return lib.maximum(tau2.rhophi_z_tau(lib, rho, phi, z, tau) + rho ** 2, 0)


def rhophi_theta_t(lib, rho, phi, theta, t):
    return rhophi_z_t(lib, rho, phi, z.rhophi_theta(lib, rho, phi, theta), t)


def rhophi_theta_tau(lib, rho, phi, theta, tau):
    return lib.maximum(tau2.rhophi_theta_tau(lib, rho, phi, theta, tau) + rho ** 2, 0)


def rhophi_eta_t(lib, rho, phi, eta, t):
    return rhophi_z_t(lib, rho, phi, z.rhophi_eta(lib, rho, phi, eta), t)


def rhophi_eta_tau(lib, rho, phi, eta, tau):
    return lib.maximum(tau2.rhophi_eta_tau(lib, rho, phi, eta, tau) + rho ** 2, 0)


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
