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


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ): (
        xy_z_xy_z,
        AzimuthalXY,
        LongitudinalZ,
        None,
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
        cartesian, azout, lout, tout = dispatch_map[
            AzimuthalXY, LongitudinalZ, AzimuthalXY, LongitudinalZ
        ]

        def f(lib, coord11, coord12, coord13, coord21, coord22, coord23):
            return cartesian(
                lib,
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
            tout,
        )


for azimuthal1 in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal1 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for azimuthal2 in (AzimuthalXY, AzimuthalRhoPhi):
            for longitudinal2 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
                make_conversion(azimuthal1, longitudinal1, azimuthal2, longitudinal2)


def dispatch(v1, v2):
    if v1.lib is not v2.lib:
        raise TypeError(
            f"cannot use {v1} (requires {v1.lib}) and {v2} (requires {v1.lib}) together"
        )
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
                *v2.longitudinal.elements,
            ),
            returns,
        )
