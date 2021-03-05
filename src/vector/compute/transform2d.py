# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import vector.compute.planar.x
import vector.compute.planar.y
import vector.geometry


def apply_xy(lib, x1, y1, xx, xy, yx, yy):
    return (xx * x1 + xy * y1, yx * x1 + yy * y1)


def apply_rhophi(lib, rho1, phi1, xx, xy, yx, yy):
    return apply_xy(
        lib,
        vector.compute.planar.x.rhophi(lib, rho1, phi1),
        vector.compute.planar.y.rhophi(lib, rho1, phi1),
        xx,
        xy,
        yx,
        yy,
    )


dispatch_map = {
    (vector.geometry.AzimuthalXY,): apply_xy,
    (vector.geometry.AzimuthalRhoPhi,): apply_rhophi,
}


def dispatch(v, t):
    if v.lib is not t.lib:
        raise TypeError(f"cannot use a {t.lib} transform on a {v.lib} vector")
    return dispatch_map[
        vector.geometry.aztype(v),
    ](v.lib, *(v.azimuthal.elements + t.elements))


def from_AzimuthalRotation(lib, angle):
    c = lib.cos(angle)
    s = lib.sin(angle)
    return (c, -s, s, c)
