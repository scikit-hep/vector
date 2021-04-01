# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Spatial.rotate_quaternion(self, u, i, j, k)
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


# Follows ROOT's conventions.
#
# https://github.com/root-project/root/blob/f8efb11a51cbe5b5152ebef19a4f7b78744ca2fa/math/genvector/src/3DConversions.cxx#L478-L502
#
# I don't know how this relates to Wikipedia's representation:
#
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix


def cartesian(lib, u, i, j, k, x, y, z):
    q00 = u * u
    q01 = u * i
    q02 = u * j
    q03 = u * k
    q11 = i * i
    q12 = i * j
    q13 = i * k
    q22 = j * j
    q23 = j * k
    q33 = k * k
    xp = (q00 + q11 - q22 - q33) * x + (2 * (q12 - q03)) * y + (2 * (q02 + q13)) * z
    yp = (2 * (q12 + q03)) * x + (q00 - q11 + q22 - q33) * y + (2 * (q23 - q01)) * z
    zp = (2 * (q13 - q02)) * x + (2 * (q23 + q01)) * y + (q00 - q11 - q22 + q33) * z
    return (xp, yp, zp)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (cartesian, AzimuthalXY, LongitudinalZ),
}


def make_conversion(azimuthal, longitudinal):
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
        cartesian, azout, lout = dispatch_map[AzimuthalXY, LongitudinalZ]

        def f(lib, u, i, j, k, coord1, coord2, coord3):
            return cartesian(
                lib,
                u,
                i,
                j,
                k,
                to_x(lib, coord1, coord2),
                to_y(lib, coord1, coord2),
                to_z(lib, coord1, coord2, coord3),
            )

        dispatch_map[azimuthal, longitudinal] = (f, azout, lout)


for azimuthal in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        make_conversion(azimuthal, longitudinal)


def dispatch(
    u: typing.Any, i: typing.Any, j: typing.Any, k: typing.Any, vec: typing.Any
) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(vec),
            _ltype(vec),
        ),
    )
    with numpy.errstate(all="ignore"):
        return vec._wrap_result(
            _flavor_of(vec),
            function(
                vec.lib, u, i, j, k, *vec.azimuthal.elements, *vec.longitudinal.elements
            ),
            returns,
            1,
        )
