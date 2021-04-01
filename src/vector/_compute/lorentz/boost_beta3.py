# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Lorentz.boost_beta3(self, beta3)

or

.. code-block:: python

    Lorentz.boost(self, beta3=...)
"""

import numpy

from vector._compute.lorentz import transform4D
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
    _handler_of,
    _lib_of,
    _ltype,
    _ttype,
)


def cartesian_t(lib, x1, y1, z1, t1, betax, betay, betaz):
    bp2 = betax ** 2 + betay ** 2 + betaz ** 2
    gamma = 1 / lib.sqrt(1 - bp2)
    bgam = gamma ** 2 / (1 + gamma)
    xx = 1 + bgam * betax * betax
    yy = 1 + bgam * betay * betay
    zz = 1 + bgam * betaz * betaz
    xy = bgam * betax * betay
    xz = bgam * betax * betaz
    yz = bgam * betay * betaz
    yx = xy
    zx = xz
    zy = yz
    xt = gamma * betax
    yt = gamma * betay
    zt = gamma * betaz
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


def cartesian_tau(lib, x1, y1, z1, tau1, betax, betay, betaz):
    bp2 = betax ** 2 + betay ** 2 + betaz ** 2
    gamma = 1 / lib.sqrt(1 - bp2)
    bgam = gamma ** 2 / (1 + gamma)
    xx = 1 + bgam * betax * betax
    yy = 1 + bgam * betay * betay
    zz = 1 + bgam * betaz * betaz
    xy = bgam * betax * betay
    xz = bgam * betax * betaz
    yz = bgam * betay * betaz
    yx = xy
    zx = xz
    zy = yz
    xt = gamma * betax
    yt = gamma * betay
    zt = gamma * betaz

    # fmt: off
    return transform4D.cartesian_tau(
        lib,
        xx, xy, xz, xt,
        yx, yy, yz, yt,
        zx, zy, zz, zt,
        x1, y1, z1, tau1,
    )
    # fmt: on


def cartesian_t_xy_z(lib, x1, y1, z1, t1, x2, y2, z2):
    return cartesian_t(lib, x1, y1, z1, t1, x2, y2, z2)


def cartesian_tau_xy_z(lib, x1, y1, z1, tau1, x2, y2, z2):
    return cartesian_tau(lib, x1, y1, z1, tau1, x2, y2, z2)


def cartesian_t_xy_theta(lib, x1, y1, z1, t1, x2, y2, theta2):
    return cartesian_t(lib, x1, y1, z1, t1, x2, y2, z.xy_theta(lib, x2, y2, theta2))


def cartesian_tau_xy_theta(lib, x1, y1, z1, tau1, x2, y2, theta2):
    return cartesian_tau(lib, x1, y1, z1, tau1, x2, y2, z.xy_theta(lib, x2, y2, theta2))


def cartesian_t_xy_eta(lib, x1, y1, z1, t1, x2, y2, eta2):
    return cartesian_t(lib, x1, y1, z1, t1, x2, y2, z.xy_eta(lib, x2, y2, eta2))


def cartesian_tau_xy_eta(lib, x1, y1, z1, tau1, x2, y2, eta2):
    return cartesian_tau(lib, x1, y1, z1, tau1, x2, y2, z.xy_eta(lib, x2, y2, eta2))


def cartesian_t_rhophi_z(lib, x1, y1, z1, t1, rho2, phi2, z2):
    return cartesian_t(
        lib, x1, y1, z1, t1, x.rhophi(lib, rho2, phi2), y.rhophi(lib, rho2, phi2), z2
    )


def cartesian_tau_rhophi_z(lib, x1, y1, z1, tau1, rho2, phi2, z2):
    return cartesian_tau(
        lib, x1, y1, z1, tau1, x.rhophi(lib, rho2, phi2), y.rhophi(lib, rho2, phi2), z2
    )


def cartesian_t_rhophi_theta(lib, x1, y1, z1, t1, rho2, phi2, theta2):
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def cartesian_tau_rhophi_theta(lib, x1, y1, z1, tau1, rho2, phi2, theta2):
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def cartesian_t_rhophi_eta(lib, x1, y1, z1, t1, rho2, phi2, eta2):
    return cartesian_t(
        lib,
        x1,
        y1,
        z1,
        t1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def cartesian_tau_rhophi_eta(lib, x1, y1, z1, tau1, rho2, phi2, eta2):
    return cartesian_tau(
        lib,
        x1,
        y1,
        z1,
        tau1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT, AzimuthalXY, LongitudinalZ): (
        cartesian_t_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalZ, TemporalTau, AzimuthalXY, LongitudinalZ): (
        cartesian_tau_xy_z,
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
    ): (cartesian_t_xy_theta, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalXY,
        LongitudinalTheta,
    ): (cartesian_tau_xy_theta, AzimuthalXY, LongitudinalZ, TemporalTau),
    (AzimuthalXY, LongitudinalZ, TemporalT, AzimuthalXY, LongitudinalEta): (
        cartesian_t_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (AzimuthalXY, LongitudinalZ, TemporalTau, AzimuthalXY, LongitudinalEta): (
        cartesian_tau_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
    ),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalZ,
    ): (cartesian_t_rhophi_z, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalZ,
    ): (cartesian_tau_rhophi_z, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalTheta,
    ): (cartesian_t_rhophi_theta, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalTheta,
    ): (cartesian_tau_rhophi_theta, AzimuthalXY, LongitudinalZ, TemporalTau),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalEta,
    ): (cartesian_t_rhophi_eta, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalTau,
        AzimuthalRhoPhi,
        LongitudinalEta,
    ): (cartesian_tau_rhophi_eta, AzimuthalXY, LongitudinalZ, TemporalTau),
}


def make_conversion(azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2):
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
            AzimuthalXY, LongitudinalZ, temporal1, azimuthal2, longitudinal2
        ]

        def f(lib, coord11, coord12, coord13, coord14, coord21, coord22, coord23):
            return cartesian(
                lib,
                to_x(lib, coord11, coord12),
                to_y(lib, coord11, coord12),
                to_z(lib, coord11, coord12, coord13),
                coord14,
                coord21,
                coord22,
                coord23,
            )

        dispatch_map[
            azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2
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
                    make_conversion(
                        azimuthal1,
                        longitudinal1,
                        temporal1,
                        azimuthal2,
                        longitudinal2,
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
            ),
            returns,
            1,
        )
