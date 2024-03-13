# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.rotate_axis(self, axis, angle)
"""

from __future__ import annotations

import typing

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
    _handler_of,
    _lib_of,
    _ltype,
)

# Rotation is only computed in Cartesian coordinates; the rest are conversions.


def cartesian(lib, angle, x1, y1, z1, x2, y2, z2):
    norm = lib.sqrt(x1**2 + y1**2 + z1**2)
    ux = x1 / norm
    uy = y1 / norm
    uz = z1 / norm
    c = lib.cos(angle)
    s = lib.sin(angle)
    c1 = 1 - c
    xp = (
        (c + ux**2 * c1) * x2
        + (ux * uy * c1 - uz * s) * y2
        + (ux * uz * c1 + uy * s) * z2
    )
    yp = (
        (ux * uy * c1 + uz * s) * x2
        + (c + uy**2 * c1) * y2
        + (uy * uz * c1 - ux * s) * z2
    )
    zp = (
        (ux * uz * c1 - uy * s) * x2
        + (uy * uz * c1 + ux * s) * y2
        + (c + uz**2 * c1) * z2
    )
    return (xp, yp, zp)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        cartesian,
        AzimuthalXY,
        LongitudinalZ,
    ),
}


def make_conversion(azimuthal1, longitudinal1, azimuthal2, longitudinal2):
    if (azimuthal1, longitudinal1, azimuthal2, longitudinal2) != (
        AzimuthalXY,
        LongitudinalZ,
        AzimuthalXY,
        LongitudinalZ,
    ):
        if azimuthal1 is AzimuthalXY:
            to_x1 = x.xy
            to_y1 = y.xy
            if longitudinal1 is LongitudinalZ:
                to_z1 = z.xy_z
            elif longitudinal1 is LongitudinalTheta:
                to_z1 = z.xy_theta
            elif longitudinal1 is LongitudinalEta:
                to_z1 = z.xy_eta
        elif azimuthal1 is AzimuthalRhoPhi:
            to_x1 = x.rhophi
            to_y1 = y.rhophi
            if longitudinal1 is LongitudinalZ:
                to_z1 = z.rhophi_z
            elif longitudinal1 is LongitudinalTheta:
                to_z1 = z.rhophi_theta
            elif longitudinal1 is LongitudinalEta:
                to_z1 = z.rhophi_eta
        if azimuthal2 is AzimuthalXY:
            to_x2 = x.xy
            to_y2 = y.xy
            if longitudinal2 is LongitudinalZ:
                to_z2 = z.xy_z
            elif longitudinal2 is LongitudinalTheta:
                to_z2 = z.xy_theta
            elif longitudinal2 is LongitudinalEta:
                to_z2 = z.xy_eta
        elif azimuthal2 is AzimuthalRhoPhi:
            to_x2 = x.rhophi
            to_y2 = y.rhophi
            if longitudinal2 is LongitudinalZ:
                to_z2 = z.rhophi_z
            elif longitudinal2 is LongitudinalTheta:
                to_z2 = z.rhophi_theta
            elif longitudinal2 is LongitudinalEta:
                to_z2 = z.rhophi_eta
        cartesian, azout, lout = dispatch_map[
            AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ
        ]

        def f(lib, angle, coord11, coord12, coord13, coord21, coord22, coord23):
            return cartesian(
                lib,
                angle,
                to_x1(lib, coord11, coord12),
                to_y1(lib, coord11, coord12),
                to_z1(lib, coord11, coord12, coord13),
                to_x2(lib, coord21, coord22),
                to_y2(lib, coord21, coord22),
                to_z2(lib, coord21, coord22, coord23),
            )

        dispatch_map[azimuthal1, longitudinal1, azimuthal2, longitudinal2] = (
            f,
            azout,
            lout,
        )


for azimuthal1 in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal1 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for azimuthal2 in (AzimuthalXY, AzimuthalRhoPhi):
            for longitudinal2 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
                make_conversion(azimuthal1, longitudinal1, azimuthal2, longitudinal2)


def dispatch(angle: typing.Any, v1: typing.Any, v2: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v1),  # v1 is the axis about which we're rotating
            _ltype(v1),
            _aztype(v2),  # v2 is the primary vector, the one being rotated
            _ltype(v2),
        ),
    )
    with numpy.errstate(all="ignore"):
        return _handler_of(v2)._wrap_result(  # note: _handler_of(v2)
            _flavor_of(v2),  # note: _flavor_of(v2)
            function(
                _lib_of(v1, v2),
                angle,
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements,
            ),
            returns,
            1,
        )
