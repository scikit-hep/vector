# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Lorentz.dot(self, other)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.lorentz import t
from vector._compute.spatial import dot
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
    _handler_of,
    _lib_of,
    _ltype,
    _ttype,
)

dispatch_map = {}


def make_conversion(
    azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2, temporal2
):
    spatial_dot, _ = dot.dispatch_map[
        azimuthal1, longitudinal1, azimuthal2, longitudinal2
    ]

    if azimuthal1 is AzimuthalXY:
        if longitudinal1 is LongitudinalZ:
            if temporal1 is TemporalT:
                to_t1 = t.xy_z_t
            elif temporal1 is TemporalTau:
                to_t1 = t.xy_z_tau
        elif longitudinal1 is LongitudinalTheta:
            if temporal1 is TemporalT:
                to_t1 = t.xy_theta_t
            elif temporal1 is TemporalTau:
                to_t1 = t.xy_theta_tau
        elif longitudinal1 is LongitudinalEta:
            if temporal1 is TemporalT:
                to_t1 = t.xy_eta_t
            elif temporal1 is TemporalTau:
                to_t1 = t.xy_eta_tau
    elif azimuthal1 is AzimuthalRhoPhi:
        if longitudinal1 is LongitudinalZ:
            if temporal1 is TemporalT:
                to_t1 = t.rhophi_z_t
            elif temporal1 is TemporalTau:
                to_t1 = t.rhophi_z_tau
        elif longitudinal1 is LongitudinalTheta:
            if temporal1 is TemporalT:
                to_t1 = t.rhophi_theta_t
            elif temporal1 is TemporalTau:
                to_t1 = t.rhophi_theta_tau
        elif longitudinal1 is LongitudinalEta:
            if temporal1 is TemporalT:
                to_t1 = t.rhophi_eta_t
            elif temporal1 is TemporalTau:
                to_t1 = t.rhophi_eta_tau

    if azimuthal2 is AzimuthalXY:
        if longitudinal2 is LongitudinalZ:
            if temporal2 is TemporalT:
                to_t2 = t.xy_z_t
            elif temporal2 is TemporalTau:
                to_t2 = t.xy_z_tau
        elif longitudinal2 is LongitudinalTheta:
            if temporal2 is TemporalT:
                to_t2 = t.xy_theta_t
            elif temporal2 is TemporalTau:
                to_t2 = t.xy_theta_tau
        elif longitudinal2 is LongitudinalEta:
            if temporal2 is TemporalT:
                to_t2 = t.xy_eta_t
            elif temporal2 is TemporalTau:
                to_t2 = t.xy_eta_tau
    elif azimuthal2 is AzimuthalRhoPhi:
        if longitudinal2 is LongitudinalZ:
            if temporal2 is TemporalT:
                to_t2 = t.rhophi_z_t
            elif temporal2 is TemporalTau:
                to_t2 = t.rhophi_z_tau
        elif longitudinal2 is LongitudinalTheta:
            if temporal2 is TemporalT:
                to_t2 = t.rhophi_theta_t
            elif temporal2 is TemporalTau:
                to_t2 = t.rhophi_theta_tau
        elif longitudinal2 is LongitudinalEta:
            if temporal2 is TemporalT:
                to_t2 = t.rhophi_eta_t
            elif temporal2 is TemporalTau:
                to_t2 = t.rhophi_eta_tau

    def f(lib, coord11, coord12, coord13, coord14, coord21, coord22, coord23, coord24):
        return (
            to_t1(lib, coord11, coord12, coord13, coord14)
            * to_t2(lib, coord21, coord22, coord23, coord24)
        ) - spatial_dot(lib, coord11, coord12, coord13, coord21, coord22, coord23)

    dispatch_map[
        azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2, temporal2
    ] = (f, float)


for azimuthal1 in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal1 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for temporal1 in (TemporalT, TemporalTau):
            for azimuthal2 in (AzimuthalXY, AzimuthalRhoPhi):
                for longitudinal2 in (
                    LongitudinalZ,
                    LongitudinalTheta,
                    LongitudinalEta,
                ):
                    for temporal2 in (TemporalT, TemporalTau):
                        make_conversion(
                            azimuthal1,
                            longitudinal1,
                            temporal1,
                            azimuthal2,
                            longitudinal2,
                            temporal2,
                        )


def dispatch(v1: typing.Any, v2: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v1),
            _ltype(v1),
            _ttype(v1),
            _aztype(v2),
            _ltype(v2),
            _ttype(v2),
        ),
    )
    with numpy.errstate(all="ignore"):
        handler = _handler_of(v1, v2)
        return handler._wrap_result(
            _flavor_of(v1, v2),
            handler._wrap_dispatched_function(function)(
                _lib_of(v1, v2),
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v1.temporal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements,
                *v2.temporal.elements,
            ),
            returns,
            2,
        )
