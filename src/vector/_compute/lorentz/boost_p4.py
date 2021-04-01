# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Lorentz.boost_p4(self, p4)

or

.. code-block:: python

    Lorentz.boost(self, p4=...)
"""

import numpy

from vector._compute.lorentz import transform4D
from vector._compute.planar import x, y
from vector._compute.spatial import mag2, z
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
    _handler_of,
    _lib_of,
    _ltype,
    _ttype,
)


def cartesian_t(lib, x1, y1, z1, t1, energy, mass, mass2, x2, y2, z2):
    gamma = energy / mass
    mass2_gamma_1 = mass2 * (gamma + 1)
    gbetax = x2 / mass
    gbetay = y2 / mass
    gbetaz = z2 / mass
    xx = 1 + (x2 * x2) / mass2_gamma_1
    yy = 1 + (y2 * y2) / mass2_gamma_1
    zz = 1 + (z2 * z2) / mass2_gamma_1
    xy = (x2 * y2) / mass2_gamma_1
    xz = (x2 * z2) / mass2_gamma_1
    yz = (y2 * z2) / mass2_gamma_1
    yx = xy
    zx = xz
    zy = yz
    xt = gbetax
    yt = gbetay
    zt = gbetaz
    tx = xt
    ty = yt
    tz = zt
    tt = gamma

    # fmt: off
    return transform4D.cartesian_t(
        lib,
        xx, xy, xz, xt,
        yx, yy, yz, yt,
        zx, zy, zz, zt,
        tx, ty, tz, tt,
        x1, y1, z1, t1,
    )
    # fmt: on


def cartesian_tau(lib, x1, y1, z1, tau1, energy, mass, mass2, x2, y2, z2):
    gamma = energy / mass
    mass2_gamma_1 = mass2 * (gamma + 1)
    gbetax = x2 / mass
    gbetay = y2 / mass
    gbetaz = z2 / mass
    xx = 1 + (x2 * x2) / mass2_gamma_1
    yy = 1 + (y2 * y2) / mass2_gamma_1
    zz = 1 + (z2 * z2) / mass2_gamma_1
    xy = (x2 * y2) / mass2_gamma_1
    xz = (x2 * z2) / mass2_gamma_1
    yz = (y2 * z2) / mass2_gamma_1
    yx = xy
    zx = xz
    zy = yz
    xt = gbetax
    yt = gbetay
    zt = gbetaz

    # fmt: off
    return transform4D.cartesian_tau(
        lib,
        xx, xy, xz, xt,
        yx, yy, yz, yt,
        zx, zy, zz, zt,
        x1, y1, z1, tau1,
    )
    # fmt: on


def cartesian_t_xy_z_t(lib, x1, y1, z1, t1, x2, y2, z2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.xy_z(lib, x2, y2, z2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(lib, x1, y1, z1, t1, energy, mass, mass2, x2, y2, z2)


def cartesian_tau_xy_z_t(lib, x1, y1, z1, tau1, x2, y2, z2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.xy_z(lib, x2, y2, z2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(lib, x1, y1, z1, tau1, energy, mass, mass2, x2, y2, z2)


def cartesian_t_xy_z_tau(lib, x1, y1, z1, t1, x2, y2, z2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.xy_z(lib, x2, y2, z2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(lib, x1, y1, z1, t1, energy, mass, mass2, x2, y2, z2)


def cartesian_tau_xy_z_tau(lib, x1, y1, z1, tau1, x2, y2, z2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.xy_z(lib, x2, y2, z2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(lib, x1, y1, z1, tau1, energy, mass, mass2, x2, y2, z2)


def cartesian_t_xy_theta_t(lib, x1, y1, z1, t1, x2, y2, theta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.xy_theta(lib, x2, y2, theta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def cartesian_tau_xy_theta_t(lib, x1, y1, z1, tau1, x2, y2, theta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.xy_theta(lib, x2, y2, theta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def cartesian_t_xy_theta_tau(lib, x1, y1, z1, t1, x2, y2, theta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.xy_theta(lib, x2, y2, theta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def cartesian_tau_xy_theta_tau(lib, x1, y1, z1, tau1, x2, y2, theta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.xy_theta(lib, x2, y2, theta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def cartesian_t_xy_eta_t(lib, x1, y1, z1, t1, x2, y2, eta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.xy_eta(lib, x2, y2, eta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib, x1, y1, z1, t1, energy, mass, mass2, x2, y2, z.xy_eta(lib, x2, y2, eta2)
    )


def cartesian_tau_xy_eta_t(lib, x1, y1, z1, tau1, x2, y2, eta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.xy_eta(lib, x2, y2, eta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib, x1, y1, z1, tau1, energy, mass, mass2, x2, y2, z.xy_eta(lib, x2, y2, eta2)
    )


def cartesian_t_xy_eta_tau(lib, x1, y1, z1, t1, x2, y2, eta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.xy_eta(lib, x2, y2, eta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib, x1, y1, z1, t1, energy, mass, mass2, x2, y2, z.xy_eta(lib, x2, y2, eta2)
    )


def cartesian_tau_xy_eta_tau(lib, x1, y1, z1, tau1, x2, y2, eta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.xy_eta(lib, x2, y2, eta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib, x1, y1, z1, tau1, energy, mass, mass2, x2, y2, z.xy_eta(lib, x2, y2, eta2)
    )


def cartesian_t_rhophi_z_t(lib, x1, y1, z1, t1, rho2, phi2, z2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.rhophi_z(lib, rho2, phi2, z2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def cartesian_tau_rhophi_z_t(lib, x1, y1, z1, tau1, rho2, phi2, z2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.rhophi_z(lib, rho2, phi2, z2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def cartesian_t_rhophi_z_tau(lib, x1, y1, z1, t1, rho2, phi2, z2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.rhophi_z(lib, rho2, phi2, z2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def cartesian_tau_rhophi_z_tau(lib, x1, y1, z1, tau1, rho2, phi2, z2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.rhophi_z(lib, rho2, phi2, z2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def cartesian_t_rhophi_theta_t(lib, x1, y1, z1, t1, rho2, phi2, theta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.rhophi_theta(lib, rho2, phi2, theta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def cartesian_tau_rhophi_theta_t(lib, x1, y1, z1, tau1, rho2, phi2, theta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.rhophi_theta(lib, rho2, phi2, theta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def cartesian_t_rhophi_theta_tau(lib, x1, y1, z1, t1, rho2, phi2, theta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.rhophi_theta(lib, rho2, phi2, theta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def cartesian_tau_rhophi_theta_tau(lib, x1, y1, z1, tau1, rho2, phi2, theta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.rhophi_theta(lib, rho2, phi2, theta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def cartesian_t_rhophi_eta_t(lib, x1, y1, z1, t1, rho2, phi2, eta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.rhophi_eta(lib, rho2, phi2, eta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def cartesian_tau_rhophi_eta_t(lib, x1, y1, z1, tau1, rho2, phi2, eta2, t2):
    energy = t2
    energy2 = energy ** 2
    mass2 = energy2 - mag2.rhophi_eta(lib, rho2, phi2, eta2)
    mass = lib.sqrt(mass2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def cartesian_t_rhophi_eta_tau(lib, x1, y1, z1, t1, rho2, phi2, eta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.rhophi_eta(lib, rho2, phi2, eta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def cartesian_tau_rhophi_eta_tau(lib, x1, y1, z1, tau1, rho2, phi2, eta2, tau2):
    mass = tau2
    mass2 = mass ** 2
    energy2 = mass2 + mag2.rhophi_eta(lib, rho2, phi2, eta2)
    energy = lib.sqrt(energy2)  # NaN for spacelike boosts propagates everywhere!
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        energy,
        mass,
        mass2,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT, AzimuthalXY, LongitudinalZ, TemporalT): (
        cartesian_t_xy_z_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalZ, TemporalTau, AzimuthalXY, LongitudinalZ, TemporalT): (
        cartesian_tau_xy_z_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (AzimuthalXY, LongitudinalZ, TemporalT, AzimuthalXY, LongitudinalZ, TemporalTau): (
        cartesian_t_xy_z_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ): (
        cartesian_tau_xy_z_tau,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalXY,
        LongitudinalTheta,
        TemporalT,
    ): (cartesian_t_xy_theta_t, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalXY,
        LongitudinalTheta,
        TemporalT,
    ): (cartesian_tau_xy_theta_t, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalXY,
        LongitudinalTheta,
        TemporalTau,
    ): (cartesian_t_xy_theta_tau, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalXY,
        LongitudinalTheta,
        TemporalTau,
    ): (cartesian_tau_xy_theta_tau, AzimuthalXY, LongitudinalZ, TemporalTau),
    (AzimuthalXY, LongitudinalZ, TemporalT, AzimuthalXY, LongitudinalEta, TemporalT): (
        cartesian_t_xy_eta_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalXY,
        LongitudinalEta,
        TemporalT,
    ): (
        cartesian_tau_xy_eta_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalXY,
        LongitudinalEta,
        TemporalTau,
    ): (cartesian_t_xy_eta_tau, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalXY,
        LongitudinalEta,
        TemporalTau,
    ): (cartesian_tau_xy_eta_tau, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalZ,
        TemporalT,
    ): (cartesian_t_rhophi_z_t, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalZ,
        TemporalT,
    ): (cartesian_tau_rhophi_z_t, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalZ,
        TemporalTau,
    ): (cartesian_t_rhophi_z_tau, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalZ,
        TemporalTau,
    ): (cartesian_tau_rhophi_z_tau, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        TemporalT,
    ): (cartesian_t_rhophi_theta_t, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        TemporalT,
    ): (cartesian_tau_rhophi_theta_t, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        TemporalTau,
    ): (cartesian_t_rhophi_theta_tau, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalTheta,
        TemporalTau,
    ): (cartesian_tau_rhophi_theta_tau, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalEta,
        TemporalT,
    ): (cartesian_t_rhophi_eta_t, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalEta,
        TemporalT,
    ): (cartesian_tau_rhophi_eta_t, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalEta,
        TemporalTau,
    ): (cartesian_t_rhophi_eta_tau, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalEta,
        TemporalTau,
    ): (cartesian_tau_rhophi_eta_tau, AzimuthalXY, LongitudinalZ, TemporalTau),
}


def make_conversion(
    azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2, temporal2
):
    if (azimuthal1, longitudinal1, temporal1) != (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ):
        if azimuthal1 is AzimuthalXY:
            to_x = x.xy
            to_y = y.xy
            if longitudinal1 is LongitudinalZ:
                to_z = z.xy_z
            elif longitudinal1 is LongitudinalTheta:
                to_z = z.xy_theta
            elif longitudinal1 is LongitudinalEta:
                to_z = z.xy_eta
        elif azimuthal1 is AzimuthalRhoPhi:
            to_x = x.rhophi
            to_y = y.rhophi
            if longitudinal1 is LongitudinalZ:
                to_z = z.rhophi_z
            elif longitudinal1 is LongitudinalTheta:
                to_z = z.rhophi_theta
            elif longitudinal1 is LongitudinalEta:
                to_z = z.rhophi_eta
        cartesian, azout, lout, tout = dispatch_map[
            AzimuthalXY, LongitudinalZ, temporal1, azimuthal2, longitudinal2, temporal2
        ]

        def f(
            lib, coord11, coord12, coord13, coord14, coord21, coord22, coord23, coord24
        ):
            return cartesian(
                lib,
                to_x(lib, coord11, coord12),
                to_y(lib, coord11, coord12),
                to_z(lib, coord11, coord12, coord13),
                coord14,
                coord21,
                coord22,
                coord23,
                coord24,
            )

        dispatch_map[
            azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2, temporal2
        ] = (f, azout, lout, tout)


for azimuthal1 in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal1 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for temporal1 in (TemporalT, TemporalTau):
            for azimuthal2 in (AzimuthalXY, AzimuthalRhoPhi):
                for longitudinal2 in (
                    LongitudinalZ,
                    LongitudinalTheta,
                    LongitudinalEta,
                ):
                    for temporal2 in (TemporalT, TemporalTau):
                        make_conversion(
                            azimuthal1,
                            longitudinal1,
                            temporal1,
                            azimuthal2,
                            longitudinal2,
                            temporal2,
                        )


def dispatch(v1: typing.Any, v2: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v1),
            _ltype(v1),
            _ttype(v1),
            _aztype(v2),
            _ltype(v2),
            _ttype(v2),
        ),
    )
    with numpy.errstate(all="ignore"):
        return _handler_of(v1, v2)._wrap_result(
            _flavor_of(v1, v2),
            function(
                _lib_of(v1, v2),
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v1.temporal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements,
                *v2.temporal.elements,
            ),
            returns,
            1,
        )
