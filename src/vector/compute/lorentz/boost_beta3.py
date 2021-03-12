# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.lorentz import t, transform4D
from vector.compute.planar import x, y
from vector.compute.spatial import z
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


def cartesian(lib, x1, y1, z1, t1, betax, betay, betaz):
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
    return transform4D.cartesian(
        lib,
        xx,
        xy,
        xz,
        xt,
        yx,
        yy,
        yz,
        yt,
        zx,
        zy,
        zz,
        zt,
        tx,
        ty,
        tz,
        tt,
        x1,
        y1,
        z1,
        t1,
    )


def cartesian_xy_z(lib, x1, y1, z1, t1, x2, y2, z2):
    return cartesian(lib, x1, y1, z1, t1, x2, y2, z2)


def cartesian_xy_theta(lib, x1, y1, z1, t1, x2, y2, theta2):
    return cartesian(lib, x1, y1, z1, t1, x2, y2, z.xy_theta(lib, x2, y2, theta2))


def cartesian_xy_eta(lib, x1, y1, z1, t1, x2, y2, eta2):
    return cartesian(lib, x1, y1, z1, t1, x2, y2, z.xy_eta(lib, x2, y2, eta2))


def cartesian_rhophi_z(lib, x1, y1, z1, t1, rho2, phi2, z2):
    return cartesian(
        lib, x1, y1, z1, t1, x.rhophi(lib, rho2, phi2), y.rhophi(lib, rho2, phi2), z2
    )


def cartesian_rhophi_theta(lib, x1, y1, z1, t1, rho2, phi2, theta2):
    return cartesian(
        lib,
        x1,
        y1,
        z1,
        t1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def cartesian_rhophi_eta(lib, x1, y1, z1, t1, rho2, phi2, eta2):
    return cartesian(
        lib,
        x1,
        y1,
        z1,
        t1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT, AzimuthalXY, LongitudinalZ): (
        cartesian_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalXY,
        LongitudinalTheta,
    ): (cartesian_xy_theta, AzimuthalXY, LongitudinalZ, TemporalT),
    (AzimuthalXY, LongitudinalZ, TemporalT, AzimuthalXY, LongitudinalEta): (
        cartesian_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    ),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalZ,
    ): (cartesian_rhophi_z, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalTheta,
    ): (cartesian_rhophi_theta, AzimuthalXY, LongitudinalZ, TemporalT),
    (
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
        AzimuthalRhoPhi,
        LongitudinalEta,
    ): (cartesian_rhophi_eta, AzimuthalXY, LongitudinalZ, TemporalT),
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
                if temporal1 is TemporalT:
                    to_t = t.xy_z_t
                elif temporal1 is TemporalTau:
                    to_t = t.xy_z_tau
            elif longitudinal1 is LongitudinalTheta:
                to_z = z.xy_theta
                if temporal1 is TemporalT:
                    to_t = t.xy_theta_t
                elif temporal1 is TemporalTau:
                    to_t = t.xy_theta_tau
            elif longitudinal1 is LongitudinalEta:
                to_z = z.xy_eta
                if temporal1 is TemporalT:
                    to_t = t.xy_eta_t
                elif temporal1 is TemporalTau:
                    to_t = t.xy_eta_tau
        elif azimuthal1 is AzimuthalRhoPhi:
            to_x = x.rhophi
            to_y = y.rhophi
            if longitudinal1 is LongitudinalZ:
                to_z = z.rhophi_z
                if temporal1 is TemporalT:
                    to_t = t.rhophi_z_t
                elif temporal1 is TemporalTau:
                    to_t = t.rhophi_z_tau
            elif longitudinal1 is LongitudinalTheta:
                to_z = z.rhophi_theta
                if temporal1 is TemporalT:
                    to_t = t.rhophi_theta_t
                elif temporal1 is TemporalTau:
                    to_t = t.rhophi_theta_tau
            elif longitudinal1 is LongitudinalEta:
                to_z = z.rhophi_eta
                if temporal1 is TemporalT:
                    to_t = t.rhophi_eta_t
                elif temporal1 is TemporalTau:
                    to_t = t.rhophi_eta_tau
        cartesian, azout, lout, tout = dispatch_map[
            AzimuthalXY, LongitudinalZ, TemporalT, azimuthal2, longitudinal2
        ]

        def f(lib, coord11, coord12, coord13, coord14, coord21, coord22, coord23):
            return cartesian(
                lib,
                to_x(lib, coord11, coord12),
                to_y(lib, coord11, coord12),
                to_z(lib, coord11, coord12, coord13),
                to_t(lib, coord11, coord12, coord13, coord14),
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


def dispatch(v1, v2):
    if v1.lib is not v2.lib:
        raise TypeError(
            f"cannot use {v1} (requires {v1.lib}) and {v2} (requires {v1.lib}) together"
        )
    function, *returns = dispatch_map[
        aztype(v1),
        ltype(v1),
        ttype(v1),
        aztype(v2),
        ltype(v2),
    ]
    with numpy.errstate(all="ignore"):
        return v1._wrap_result(
            function(
                v1.lib,
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v1.temporal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements,
            ),
            returns,
        )
