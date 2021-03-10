# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.planar import x, y
from vector.compute.spatial import z
from vector.geometry import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    aztype,
    ltype,
)

# Rotation is only computed in Cartesian coordinates; the rest are conversions.


def xy_z_xy_z(lib, angle, x1, y1, z1, x2, y2, z2):
    norm = lib.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
    ux = x1 / norm
    uy = y1 / norm
    uz = z1 / norm
    c = lib.cos(angle)
    s = lib.sin(angle)
    c1 = 1 - c
    xp = (
        (c + ux ** 2 * c1) * x2
        + (ux * uy * c1 - uz * s) * y2
        + (ux * uz * c1 + uy * s) * z2
    )
    yp = (
        (ux * uy * c1 + uz * s) * x2
        + (c + uy ** 2 * c1) * y2
        + (uy * uz * c1 - ux * s) * z2
    )
    zp = (
        (ux * uz * c1 - uy * s) * x2
        + (uy * uz * c1 + ux * s) * y2
        + (c + uz ** 2 * c1) * z2
    )
    return (xp, yp, zp)


def xy_z_xy_theta(lib, angle, x1, y1, z1, x2, y2, theta2):
    return xy_z_xy_z(lib, angle, x1, y1, z1, x2, y2, z.xy_theta(lib, x2, y2, theta2))


def xy_z_xy_eta(lib, angle, x1, y1, z1, x2, y2, eta2):
    return xy_z_xy_z(lib, angle, x1, y1, z1, x2, y2, z.xy_eta(lib, x2, y2, eta2))


def xy_z_rhophi_z(lib, angle, x1, y1, z1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib, angle, x1, y1, z1, x.rhophi(lib, rho2, phi2), y.rhophi(lib, rho2, phi2), z2
    )


def xy_z_rhophi_theta(lib, angle, x1, y1, z1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_z_rhophi_eta(lib, angle, x1, y1, z1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def xy_theta_xy_z(lib, angle, x1, y1, theta1, x2, y2, z2):
    return xy_z_xy_z(lib, angle, x1, y1, z.xy_theta(lib, x1, y1, theta1), x2, y2, z2)


def xy_theta_xy_theta(lib, angle, x1, y1, theta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def xy_theta_xy_eta(lib, angle, x1, y1, theta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        z.xy_eta(lib, x2, y2, eta2),
    )


def xy_theta_rhophi_z(lib, angle, x1, y1, theta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def xy_theta_rhophi_theta(lib, angle, x1, y1, theta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_theta_rhophi_eta(lib, angle, x1, y1, theta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def xy_eta_xy_z(lib, angle, x1, y1, eta1, x2, y2, z2):
    return xy_z_xy_z(lib, angle, x1, y1, z.xy_eta(lib, x1, y1, eta1), x2, y2, z2)


def xy_eta_xy_theta(lib, angle, x1, y1, eta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def xy_eta_xy_eta(lib, angle, x1, y1, eta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x2,
        y2,
        z.xy_eta(lib, x2, y2, eta2),
    )


def xy_eta_rhophi_z(lib, angle, x1, y1, eta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def xy_eta_rhophi_theta(lib, angle, x1, y1, eta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_eta_rhophi_eta(lib, angle, x1, y1, eta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_z_xy_z(lib, angle, rho1, phi1, z1, x2, y2, z2):
    return xy_z_xy_z(
        lib, angle, x.rhophi(lib, rho1, phi1), y.rhophi(lib, rho1, phi1), z1, x2, y2, z2
    )


def rhophi_z_xy_theta(lib, angle, rho1, phi1, z1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_z_xy_eta(lib, angle, rho1, phi1, z1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x2,
        y2,
        z.xy_eta(lib, x2, y2, eta2),
    )


def rhophi_z_rhophi_z(lib, angle, rho1, phi1, z1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def rhophi_z_rhophi_theta(lib, angle, rho1, phi1, z1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def rhophi_z_rhophi_eta(lib, angle, rho1, phi1, z1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_theta_xy_z(lib, angle, rho1, phi1, theta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x2,
        y2,
        z2,
    )


def rhophi_theta_xy_theta(lib, angle, rho1, phi1, theta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_theta_xy_eta(lib, angle, rho1, phi1, theta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x2,
        y2,
        z.xy_eta(lib, x2, y2, eta2),
    )


def rhophi_theta_rhophi_z(lib, angle, rho1, phi1, theta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def rhophi_theta_rhophi_theta(lib, angle, rho1, phi1, theta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def rhophi_theta_rhophi_eta(lib, angle, rho1, phi1, theta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_theta(lib, rho1, phi1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_eta_xy_z(lib, angle, rho1, phi1, eta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x2,
        y2,
        z2,
    )


def rhophi_eta_xy_theta(lib, angle, rho1, phi1, eta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x2,
        y2,
        z.xy_theta(lib, x2, y2, theta2),
    )


def rhophi_eta_xy_eta(lib, angle, rho1, phi1, eta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x2,
        y2,
        z.xy_eta(lib, x2, y2, eta2),
    )


def rhophi_eta_rhophi_z(lib, angle, rho1, phi1, eta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def rhophi_eta_rhophi_theta(lib, angle, rho1, phi1, eta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def rhophi_eta_rhophi_eta(lib, angle, rho1, phi1, eta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        angle,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.rhophi_eta(lib, rho1, phi1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        xy_z_xy_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        xy_z_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalEta): (
        xy_z_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        xy_z_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_z_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        xy_z_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        xy_theta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        xy_theta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        xy_theta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_theta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_theta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_theta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalZ): (
        xy_eta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        xy_eta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        xy_eta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_eta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_eta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_eta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        rhophi_z_xy_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        rhophi_z_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalEta): (
        rhophi_z_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_z_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_z_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_z_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        rhophi_theta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        rhophi_theta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        rhophi_theta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_theta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_theta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalZ): (
        rhophi_eta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        rhophi_eta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        rhophi_eta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_eta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_eta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_eta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
    ),
}


def dispatch(angle, v1, v2):
    function, *returns = dispatch_map[
        aztype(v1),
        ltype(v1),
        aztype(v2),
        ltype(v2),
    ]
    with numpy.errstate(all="ignore"):
        return v1._wrap_result(
            function(
                v1.lib,
                angle,
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements
            ),
            returns,
        )
