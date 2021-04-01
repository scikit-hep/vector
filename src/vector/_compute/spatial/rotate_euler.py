# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Spatial.rotate_euler(self, phi, theta, psi, order=...)
"""

import numpy

from vector._compute.planar import x, y
from vector._compute.spatial import z
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _flavor_of,
    _from_signature,
    _ltype,
)

# Rotation is only computed in Cartesian coordinates; the rest are conversions.


# Matrices copied from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
#
# ROOT uses the same names, phi, theta, psi, but takes the arguments in the
# opposite order (i.e. first psi, then theta, finally phi) and takes their
# direction to be the opposite way (e.g. Wikipedia's psi is ROOT's -psi for all
# three). I've left the matrices in terms of the same c1, c2, c3, s1, s2, s3,
# but changed the definitions of c1, c2, c3, s1, s2, s3 to agree with ROOT.
#
# Also, ROOT's angle order convention is "zxz", so that's our default.
#
# https://github.com/root-project/root/blob/f8efb11a51cbe5b5152ebef19a4f7b78744ca2fa/math/genvector/src/3DConversions.cxx#L420-L436


def cartesian_xzx(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c2) * x + (-c3 * s2) * y + (s2 * s3) * z
    yp = (c1 * s2) * x + (c1 * c2 * c3 - s1 * s3) * y + (-c3 * s1 - c1 * c2 * s3) * z
    zp = (s1 * s2) * x + (c1 * s3 + c2 * c3 * s1) * y + (c1 * c3 - c2 * s1 * s3) * z
    return (xp, yp, zp)


def cartesian_xyx(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c2) * x + (s2 * s3) * y + (c3 * s2) * z
    yp = (s1 * s2) * x + (c1 * c3 - c2 * s1 * s3) * y + (-c1 * s3 - c2 * c3 * s1) * z
    zp = (-c1 * s2) * x + (c3 * s1 + c1 * c2 * s3) * y + (c1 * c2 * c3 - s1 * s3) * z
    return (xp, yp, zp)


def cartesian_yxy(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c3 - c2 * s1 * s3) * x + (s1 * s2) * y + (c1 * s3 + c2 * c3 * s1) * z
    yp = (s2 * s3) * x + (c2) * y + (-c3 * s2) * z
    zp = (-c3 * s1 - c1 * c2 * s3) * x + (c1 * s2) * y + (c1 * c2 * c3 - s1 * s3) * z
    return (xp, yp, zp)


def cartesian_yzy(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c2 * c3 - s1 * s3) * x + (-c1 * s2) * y + (c3 * s1 + c1 * c2 * s3) * z
    yp = (c3 * s2) * x + (c2) * y + (s2 * s3) * z
    zp = (-c1 * s3 - c2 * c3 * s1) * x + (s1 * s2) * y + (c1 * c3 - c2 * s1 * s3) * z
    return (xp, yp, zp)


def cartesian_zyz(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c2 * c3 - s1 * s3) * x + (-c3 * s1 - c1 * c2 * s3) * y + (c1 * s2) * z
    yp = (c1 * s3 + c2 * c3 * s1) * x + (c1 * c3 - c2 * s1 * s3) * y + (s1 * s2) * z
    zp = (-c3 * s2) * x + (s2 * s3) * y + (c2) * z
    return (xp, yp, zp)


def cartesian_zxz(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c3 - c2 * s1 * s3) * x + (-c1 * s3 - c2 * c3 * s1) * y + (s1 * s2) * z
    yp = (c3 * s1 + c1 * c2 * s3) * x + (c1 * c2 * c3 - s1 * s3) * y + (-c1 * s2) * z
    zp = (s2 * s3) * x + (c3 * s2) * y + (c2) * z
    return (xp, yp, zp)


def cartesian_xzy(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c2 * c3) * x + (-s2) * y + (c2 * s3) * z
    yp = (s1 * s3 + c1 * c3 * s2) * x + (c1 * c2) * y + (c1 * s2 * s3 - c3 * s1) * z
    zp = (c3 * s1 * s2 - c1 * s3) * x + (c2 * s1) * y + (c1 * c3 + s1 * s2 * s3) * z
    return (xp, yp, zp)


def cartesian_xyz(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c2 * c3) * x + (-c2 * s3) * y + (s2) * z
    yp = (c1 * s3 + c3 * s1 * s2) * x + (c1 * c3 - s1 * s2 * s3) * y + (-c2 * s1) * z
    zp = (s1 * s3 - c1 * c3 * s2) * x + (c3 * s1 + c1 * s2 * s3) * y + (c1 * c2) * z
    return (xp, yp, zp)


def cartesian_yxz(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c3 + s1 * s2 * s3) * x + (c3 * s1 * s2 - c1 * s3) * y + (c2 * s1) * z
    yp = (c2 * s3) * x + (c2 * c3) * y + (-s2) * z
    zp = (c1 * s2 * s3 - c3 * s1) * x + (c1 * c3 * s2 + s1 * s3) * y + (c1 * c2) * z
    return (xp, yp, zp)


def cartesian_yzx(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c2) * x + (s1 * s3 - c1 * c3 * s2) * y + (c3 * s1 + c1 * s2 * s3) * z
    yp = (s2) * x + (c2 * c3) * y + (-c2 * s3) * z
    zp = (-c2 * s1) * x + (c1 * s3 + c3 * s1 * s2) * y + (c1 * c3 - s1 * s2 * s3) * z
    return (xp, yp, zp)


def cartesian_zyx(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c2) * x + (c1 * s2 * s3 - c3 * s1) * y + (s1 * s3 + c1 * c3 * s2) * z
    yp = (c2 * s1) * x + (c1 * c3 + s1 * s2 * s3) * y + (c3 * s1 * s2 - c1 * s3) * z
    zp = (-s2) * x + (c2 * s3) * y + (c2 * c3) * z
    return (xp, yp, zp)


def cartesian_zxy(lib, phi, theta, psi, x, y, z):
    c1 = lib.cos(psi)
    s1 = -lib.sin(psi)
    c2 = lib.cos(theta)
    s2 = -lib.sin(theta)
    c3 = lib.cos(phi)
    s3 = -lib.sin(phi)
    xp = (c1 * c3 - s1 * s2 * s3) * x + (-c2 * s1) * y + (c1 * s3 + c3 * s1 * s2) * z
    yp = (c3 * s1 + c1 * s2 * s3) * x + (c1 * c2) * y + (s1 * s3 - c1 * c3 * s2) * z
    zp = (-c2 * s3) * x + (s2) * y + (c2 * c3) * z
    return (xp, yp, zp)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, "xzx"): (cartesian_xzx, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "xyx"): (cartesian_xyx, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "yxy"): (cartesian_yxy, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "yzy"): (cartesian_yzy, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "zyz"): (cartesian_zyz, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "zxz"): (cartesian_zxz, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "xzy"): (cartesian_xzy, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "xyz"): (cartesian_xyz, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "yxz"): (cartesian_yxz, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "yzx"): (cartesian_yzx, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "zyx"): (cartesian_zyx, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalZ, "zxy"): (cartesian_zxy, AzimuthalXY, LongitudinalZ),
}


def make_conversion(azimuthal, longitudinal, order):
    if (azimuthal, longitudinal) != (AzimuthalXY, LongitudinalZ):
        if azimuthal is AzimuthalXY:
            to_x = x.xy
            to_y = y.xy
            if longitudinal is LongitudinalZ:
                to_z = z.xy_z
            elif longitudinal is LongitudinalTheta:
                to_z = z.xy_theta
            elif longitudinal is LongitudinalEta:
                to_z = z.xy_eta
        elif azimuthal is AzimuthalRhoPhi:
            to_x = x.rhophi
            to_y = y.rhophi
            if longitudinal is LongitudinalZ:
                to_z = z.rhophi_z
            elif longitudinal is LongitudinalTheta:
                to_z = z.rhophi_theta
            elif longitudinal is LongitudinalEta:
                to_z = z.rhophi_eta
        cartesian, azout, lout = dispatch_map[AzimuthalXY, LongitudinalZ, order]

        def f(lib, phi, theta, psi, coord1, coord2, coord3):
            return cartesian(
                lib,
                phi,
                theta,
                psi,
                to_x(lib, coord1, coord2),
                to_y(lib, coord1, coord2),
                to_z(lib, coord1, coord2, coord3),
            )

        dispatch_map[azimuthal, longitudinal, order] = (f, azout, lout)


for azimuthal in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for order in (
            "xzx",
            "xyx",
            "yxy",
            "yzy",
            "zyz",
            "zxz",
            "xzy",
            "xyz",
            "yxz",
            "yzx",
            "zyx",
            "zxy",
        ):
            make_conversion(azimuthal, longitudinal, order)


def dispatch(
    phi: typing.Any,
    theta: typing.Any,
    psi: typing.Any,
    order: typing.Any,
    v: typing.Any,
) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v),
            _ltype(v),
            order,
        ),
    )
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(
                v.lib, phi, theta, psi, *v.azimuthal.elements, *v.longitudinal.elements
            ),
            returns,
            1,
        )
