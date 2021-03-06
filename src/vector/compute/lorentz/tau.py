# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.lorentz import tau2
from vector.geometry import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    TemporalT,
    TemporalTau,
    aztype,
    ltype,
    ttype,
)


def xy_z_t(lib, x, y, z, t):
    return lib.sqrt(tau2.xy_z_t(lib, x, y, z, t))


def xy_z_tau(lib, x, y, z, tau):
    return tau


def xy_theta_t(lib, x, y, theta, t):
    return lib.sqrt(tau2.xy_theta_t(lib, x, y, theta, t))


def xy_theta_tau(lib, x, y, theta, tau):
    return tau


def xy_eta_t(lib, x, y, eta, t):
    return lib.sqrt(tau2.xy_eta_t(lib, x, y, eta, t))


def xy_eta_tau(lib, x, y, eta, tau):
    return tau


def rhophi_z_t(lib, rho, phi, z, t):
    return lib.sqrt(tau2.rhophi_z_t(lib, rho, phi, z, t))


def rhophi_z_tau(lib, rho, phi, z, tau):
    return tau


def rhophi_theta_t(lib, rho, phi, theta, t):
    return lib.sqrt(tau2.rhophi_theta_t(lib, rho, phi, theta, t))


def rhophi_theta_tau(lib, rho, phi, theta, tau):
    return tau


def rhophi_eta_t(lib, rho, phi, eta, t):
    return lib.sqrt(tau2.rhophi_eta_t(lib, rho, phi, eta, t))


def rhophi_eta_tau(lib, rho, phi, eta, tau):
    return tau


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT): xy_z_t,
    (AzimuthalXY, LongitudinalZ, TemporalTau): xy_z_tau,
    (AzimuthalXY, LongitudinalTheta, TemporalT): xy_theta_t,
    (AzimuthalXY, LongitudinalTheta, TemporalTau): xy_theta_tau,
    (AzimuthalXY, LongitudinalEta, TemporalT): xy_eta_t,
    (AzimuthalXY, LongitudinalEta, TemporalTau): xy_eta_tau,
    (AzimuthalRhoPhi, LongitudinalZ, TemporalT): rhophi_z_t,
    (AzimuthalRhoPhi, LongitudinalZ, TemporalTau): rhophi_z_tau,
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalT): rhophi_theta_t,
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalTau): rhophi_theta_tau,
    (AzimuthalRhoPhi, LongitudinalEta, TemporalT): rhophi_eta_t,
    (AzimuthalRhoPhi, LongitudinalEta, TemporalTau): rhophi_eta_tau,
}


def dispatch(v):
    with numpy.errstate(all="ignore"):
        return v.lib.nan_to_num(
            dispatch_map[aztype(v), ltype(v), ttype(v),](
                v.lib,
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements
            ),
            nan=0.0,
        )
