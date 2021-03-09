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


# Cross-product is only computed in Cartesian coordinates; the rest are conversions.


def xy_z_xy_z(lib, x1, y1, z1, x2, y2, z2):
    return (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)


def xy_z_xy_theta(lib, x1, y1, z1, x2, y2, theta2):
    return xy_z_xy_z(lib, x1, y1, z1, x2, y2, *z.xy_theta(lib, x2, y2, theta2))


def xy_z_xy_eta(lib, x1, y1, z1, x2, y2, eta2):
    return xy_z_xy_z(lib, x1, y1, z1, x2, y2, *z.xy_eta(lib, x2, y2, eta2))


def xy_z_rhophi_z(lib, x1, y1, z1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib, x1, y1, z1, x.rhophi(lib, rho2, phi2), y.rhophi(lib, rho2, phi2), z2
    )


def xy_z_rhophi_theta(lib, x1, y1, z1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_z_rhophi_eta(lib, x1, y1, z1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def xy_theta_xy_z(lib, x1, y1, theta1, x2, y2, z2):
    return xy_z_xy_z(lib, x1, y1, z.xy_theta(lib, x1, y1, theta1), x2, y2, z2)


def xy_theta_xy_theta(lib, x1, y1, theta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        *z.xy_theta(lib, x2, y2, theta2)
    )


def xy_theta_xy_eta(lib, x1, y1, theta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        *z.xy_eta(lib, x2, y2, eta2)
    )


def xy_theta_rhophi_z(lib, x1, y1, theta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def xy_theta_rhophi_theta(lib, x1, y1, theta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_theta_rhophi_eta(lib, x1, y1, theta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def xy_eta_xy_z(lib, x1, y1, eta1, x2, y2, z2):
    return xy_z_xy_z(lib, x1, y1, z.xy_eta(lib, x1, y1, eta1), x2, y2, z2)


def xy_eta_xy_theta(lib, x1, y1, eta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x2,
        y2,
        *z.xy_theta(lib, x2, y2, theta2)
    )


def xy_eta_xy_eta(lib, x1, y1, eta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib, x1, y1, z.xy_eta(lib, x1, y1, eta1), x2, y2, *z.xy_eta(lib, x2, y2, eta2)
    )


def xy_eta_rhophi_z(lib, x1, y1, eta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def xy_eta_rhophi_theta(lib, x1, y1, eta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def xy_eta_rhophi_eta(lib, x1, y1, eta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x1,
        y1,
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_z_xy_z(lib, rho1, phi1, z1, x2, y2, z2):
    return xy_z_xy_z(
        lib, x.rhophi(lib, rho1, phi1), y.rhophi(lib, rho1, phi1), z1, x2, y2, z2
    )


def rhophi_z_xy_theta(lib, rho1, phi1, z1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x2,
        y2,
        *z.xy_theta(lib, x2, y2, theta2)
    )


def rhophi_z_xy_eta(lib, rho1, phi1, z1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x2,
        y2,
        *z.xy_eta(lib, x2, y2, eta2)
    )


def rhophi_z_rhophi_z(lib, rho1, phi1, z1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def rhophi_z_rhophi_theta(lib, rho1, phi1, z1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def rhophi_z_rhophi_eta(lib, rho1, phi1, z1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z1,
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_theta_xy_z(lib, rho1, phi1, theta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        z2,
    )


def rhophi_theta_xy_theta(lib, rho1, phi1, theta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        *z.xy_theta(lib, x2, y2, theta2)
    )


def rhophi_theta_xy_eta(lib, rho1, phi1, theta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_theta(lib, x1, y1, theta1),
        x2,
        y2,
        *z.xy_eta(lib, x2, y2, eta2)
    )


def rhophi_theta_rhophi_z(lib, rho1, phi1, theta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def rhophi_theta_rhophi_theta(lib, rho1, phi1, theta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def rhophi_theta_rhophi_eta(lib, rho1, phi1, theta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_theta(lib, x1, y1, theta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


def rhophi_eta_xy_z(lib, rho1, phi1, eta1, x2, y2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_eta(lib, x1, y1, eta1),
        x2,
        y2,
        z2,
    )


def rhophi_eta_xy_theta(lib, rho1, phi1, eta1, x2, y2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_eta(lib, x1, y1, eta1),
        x2,
        y2,
        *z.xy_theta(lib, x2, y2, theta2)
    )


def rhophi_eta_xy_eta(lib, rho1, phi1, eta1, x2, y2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_eta(lib, x1, y1, eta1),
        x2,
        y2,
        *z.xy_eta(lib, x2, y2, eta2)
    )


def rhophi_eta_rhophi_z(lib, rho1, phi1, eta1, rho2, phi2, z2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z2,
    )


def rhophi_eta_rhophi_theta(lib, rho1, phi1, eta1, rho2, phi2, theta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_theta(lib, rho2, phi2, theta2),
    )


def rhophi_eta_rhophi_eta(lib, rho1, phi1, eta1, rho2, phi2, eta2):
    return xy_z_xy_z(
        lib,
        x.rhophi(lib, rho1, phi1),
        y.rhophi(lib, rho1, phi1),
        z.xy_eta(lib, x1, y1, eta1),
        x.rhophi(lib, rho2, phi2),
        y.rhophi(lib, rho2, phi2),
        z.rhophi_eta(lib, rho2, phi2, eta2),
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        xy_z_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        xy_z_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalEta): (
        xy_z_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        xy_z_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_z_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        xy_z_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        xy_theta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        xy_theta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        xy_theta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_theta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_theta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_theta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalZ): (
        xy_eta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        xy_eta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        xy_eta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        xy_eta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        xy_eta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalXY, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        xy_eta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        rhophi_z_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalTheta): (
        rhophi_z_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalXY, LongitudinalEta): (
        rhophi_z_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_z_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_z_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalZ, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_z_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalZ): (
        rhophi_theta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalTheta): (
        rhophi_theta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalXY, LongitudinalEta): (
        rhophi_theta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_theta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_theta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalZ): (
        rhophi_eta_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalTheta): (
        rhophi_eta_xy_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalXY, LongitudinalEta): (
        rhophi_eta_xy_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_eta_rhophi_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_eta_rhophi_theta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_eta_rhophi_eta,
        AzimuthalXY,
        LongitudinalZ,
        None,
    ),
}


def dispatch(v1, v2):
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
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements
            ),
            returns,
        )
