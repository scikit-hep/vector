# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing
import sympy

"""
.. code-block:: python

    Spatial.transform3D(self, obj)

where ``obj` has ``obj["xx"]``, ``obj["xy"]``, etc.
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


def cartesian(lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, x, y, z):
    xp = xx * x + xy * y + xz * z
    yp = yx * x + yy * y + yz * z
    zp = zx * x + zy * y + zz * z
    return (xp, yp, zp)


def xy_theta(lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, x, y, theta):
    return cartesian(
        lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, x, y, z.xy_theta(lib, x, y, theta)
    )


def xy_eta(lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, x, y, eta):
    return cartesian(
        lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, x, y, z.xy_eta(lib, x, y, eta)
    )


def rhophi_z(lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, rho, phi, z):
    return cartesian(
        lib,
        xx,
        xy,
        xz,
        yx,
        yy,
        yz,
        zx,
        zy,
        zz,
        x.rhophi(lib, rho, phi),
        y.rhophi(lib, rho, phi),
        z,
    )


def rhophi_theta(lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, rho, phi, theta):
    return cartesian(
        lib,
        xx,
        xy,
        xz,
        yx,
        yy,
        yz,
        zx,
        zy,
        zz,
        x.rhophi(lib, rho, phi),
        y.rhophi(lib, rho, phi),
        z.rhophi_theta(lib, rho, phi, theta),
    )


def rhophi_eta(lib, xx, xy, xz, yx, yy, yz, zx, zy, zz, rho, phi, eta):
    return cartesian(
        lib,
        xx,
        xy,
        xz,
        yx,
        yy,
        yz,
        zx,
        zy,
        zz,
        x.rhophi(lib, rho, phi),
        y.rhophi(lib, rho, phi),
        z.rhophi_eta(lib, rho, phi, eta),
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (cartesian, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalEta): (xy_eta, AzimuthalXY, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, AzimuthalXY, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalTheta): (rhophi_theta, AzimuthalXY, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, AzimuthalXY, LongitudinalZ),
}


def dispatch(obj: typing.Any, v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v),
            _ltype(v),
        ),
    )
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(
                v.lib,
                obj["xx"],
                obj["xy"],
                obj["xz"],
                obj["yx"],
                obj["yy"],
                obj["yz"],
                obj["zx"],
                obj["zy"],
                obj["zz"],
                *v.azimuthal.elements,
                *v.longitudinal.elements
            ),
            returns,
            1,
        )

# Contravariant transformation of vector components
# 
# enter the x, y and z compoents of your vector and the new coordinate system
# as a function of the old one (x, y and z). 
# 
# Example:
# 
# x_prime = 2*x
# y_prime = 4*y+2*x
# z_prime = 2*x+z*z
# 
# This method the returns the vector in the new coordinate system
# 
# This could also be done with the Jacobian and the dot product, but this works
# aswell (maybe the Jacobian can be added later)

def contravariant(lib, x1, y1, z1, x_prime, y_prime, z_prime):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    new_x1 = sympy.Derivative(x_prime, x) * x1 + sympy.Derivative(x_prime, y) * y1 + sympy.Derivative(x_prime, z) * z1 
    new_y1 = sympy.Derivative(y_prime, x) * x1 + sympy.Derivative(y_prime, y) * y1 + sympy.Derivative(y_prime, z) * z1 
    new_z1 = sympy.Derivative(z_prime, x) * x1 + sympy.Derivative(z_prime, y) * y1 + sympy.Derivative(z_prime, z) * z1 

    return (new_x1, new_y1, new_z1)
