# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Spatial.rotateY(self, angle)
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


def xy_z(lib, angle, x, y, z):
    s = lib.sin(angle)
    c = lib.cos(angle)
    return (c * x + s * z, y, -s * x + c * z)


def xy_theta(lib, angle, x, y, theta):
    return xy_z(lib, angle, x, y, z.xy_theta(lib, x, y, theta))


def xy_eta(lib, angle, x, y, eta):
    return xy_z(lib, angle, x, y, z.xy_eta(lib, x, y, eta))


def rhophi_z(lib, angle, rho, phi, z):
    return xy_z(lib, angle, x.rhophi(lib, rho, phi), y.rhophi(lib, rho, phi), z)


def rhophi_theta(lib, angle, rho, phi, theta):
    return xy_z(
        lib,
        angle,
        x.rhophi(lib, rho, phi),
        y.rhophi(lib, rho, phi),
        z.rhophi_theta(lib, rho, phi, theta),
    )


def rhophi_eta(lib, angle, rho, phi, eta):
    return xy_z(
        lib,
        angle,
        x.rhophi(lib, rho, phi),
        y.rhophi(lib, rho, phi),
        z.rhophi_eta(lib, rho, phi, eta),
    )


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalEta): (xy_eta, AzimuthalXY, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, AzimuthalXY, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalTheta): (rhophi_theta, AzimuthalXY, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, AzimuthalXY, LongitudinalZ),
}


def dispatch(angle: typing.Any, v: typing.Any) -> typing.Any:
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
            function(v.lib, angle, *v.azimuthal.elements, *v.longitudinal.elements),
            returns,
            1,
        )
