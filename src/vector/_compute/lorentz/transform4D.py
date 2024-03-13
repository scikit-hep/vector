# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Lorentz.transform4D(self, obj)

where ``obj`` has ``obj["xx"]``, ``obj["xy"]``, etc.
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.lorentz import t
from vector._compute.planar import x, y
from vector._compute.spatial import z
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    TemporalT,
    TemporalTau,
    _aztype,
    _flavor_of,
    _from_signature,
    _ltype,
    _ttype,
)

# Rotation is only computed in Cartesian coordinates; the rest are conversions.


def cartesian_t(
    lib, xx, xy, xz, xt, yx, yy, yz, yt, zx, zy, zz, zt, tx, ty, tz, tt, x, y, z, t
):
    xp = xx * x + xy * y + xz * z + xt * t
    yp = yx * x + yy * y + yz * z + yt * t
    zp = zx * x + zy * y + zz * z + zt * t
    tp = tx * x + ty * y + tz * z + tt * t
    return (xp, yp, zp, tp)


def cartesian_tau(lib, xx, xy, xz, xt, yx, yy, yz, yt, zx, zy, zz, zt, x, y, z, tau):
    tee = t.xy_z_tau(lib, x, y, z, tau)
    xp = xx * x + xy * y + xz * z + xt * tee
    yp = yx * x + yy * y + yz * z + yt * tee
    zp = zx * x + zy * y + zz * z + zt * tee
    return (xp, yp, zp, tau)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT): (
        cartesian_t,
        AzimuthalXY,
        LongitudinalZ,
        TemporalT,
    )
}


def make_conversion(azimuthal, longitudinal, temporal):
    if (azimuthal, longitudinal, temporal) != (AzimuthalXY, LongitudinalZ, TemporalT):
        if azimuthal is AzimuthalXY:
            to_x = x.xy
            to_y = y.xy
            if longitudinal is LongitudinalZ:
                to_z = z.xy_z
                if temporal is TemporalT:
                    to_t = t.xy_z_t
                elif temporal is TemporalTau:
                    to_t = t.xy_z_tau
            elif longitudinal is LongitudinalTheta:
                to_z = z.xy_theta
                if temporal is TemporalT:
                    to_t = t.xy_theta_t
                elif temporal is TemporalTau:
                    to_t = t.xy_theta_tau
            elif longitudinal is LongitudinalEta:
                to_z = z.xy_eta
                if temporal is TemporalT:
                    to_t = t.xy_eta_t
                elif temporal is TemporalTau:
                    to_t = t.xy_eta_tau
        elif azimuthal is AzimuthalRhoPhi:
            to_x = x.rhophi
            to_y = y.rhophi
            if longitudinal is LongitudinalZ:
                to_z = z.rhophi_z
                if temporal is TemporalT:
                    to_t = t.rhophi_z_t
                elif temporal is TemporalTau:
                    to_t = t.rhophi_z_tau
            elif longitudinal is LongitudinalTheta:
                to_z = z.rhophi_theta
                if temporal is TemporalT:
                    to_t = t.rhophi_theta_t
                elif temporal is TemporalTau:
                    to_t = t.rhophi_theta_tau
            elif longitudinal is LongitudinalEta:
                to_z = z.rhophi_eta
                if temporal is TemporalT:
                    to_t = t.rhophi_eta_t
                elif temporal is TemporalTau:
                    to_t = t.rhophi_eta_tau
        cartesian, azout, lout, tout = dispatch_map[
            AzimuthalXY, LongitudinalZ, TemporalT
        ]

        def f(
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
            coord1,
            coord2,
            coord3,
            coord4,
        ):
            return cartesian(
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
                to_x(lib, coord1, coord2),
                to_y(lib, coord1, coord2),
                to_z(lib, coord1, coord2, coord3),
                to_t(lib, coord1, coord2, coord3, coord4),
            )

        dispatch_map[azimuthal, longitudinal, temporal] = (f, azout, lout, tout)


for azimuthal in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for temporal in (TemporalT, TemporalTau):
            make_conversion(azimuthal, longitudinal, temporal)


def dispatch(obj: typing.Any, v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v),
            _ltype(v),
            _ttype(v),
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
                obj["xt"],
                obj["yx"],
                obj["yy"],
                obj["yz"],
                obj["yt"],
                obj["zx"],
                obj["zy"],
                obj["zz"],
                obj["zt"],
                obj["tx"],
                obj["ty"],
                obj["tz"],
                obj["tt"],
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements,
            ),
            returns,
            1,
        )
