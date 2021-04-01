# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Lorentz.subtract(self, other)
"""

import numpy

from vector._compute.lorentz import t, tau
from vector._compute.spatial import subtract
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
    spatial_subtract, azimuthal, longitudinal = subtract.dispatch_map[
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

    if temporal1 is TemporalT or temporal2 is TemporalT:

        def f(
            lib, coord11, coord12, coord13, coord14, coord21, coord22, coord23, coord24
        ):
            coord1, coord2, coord3 = spatial_subtract(
                lib, coord11, coord12, coord13, coord21, coord22, coord23
            )
            t1 = to_t1(lib, coord11, coord12, coord13, coord14)
            t2 = to_t2(lib, coord21, coord22, coord23, coord24)
            return (coord1, coord2, coord3, t1 - t2)

        dispatch_map[
            azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2, temporal2
        ] = (f, azimuthal, longitudinal, TemporalT)

    else:
        if azimuthal is AzimuthalXY:
            if longitudinal is LongitudinalZ:
                to_tau = tau.xy_z_t
            elif longitudinal is LongitudinalTheta:
                to_tau = tau.xy_theta_t
            elif longitudinal is LongitudinalEta:
                to_tau = tau.xy_eta_t
        elif azimuthal is AzimuthalRhoPhi:
            if longitudinal is LongitudinalZ:
                to_tau = tau.rhophi_z_t
            elif longitudinal is LongitudinalTheta:
                to_tau = tau.rhophi_theta_t
            elif longitudinal is LongitudinalEta:
                to_tau = tau.rhophi_eta_t

        def f(
            lib, coord11, coord12, coord13, coord14, coord21, coord22, coord23, coord24
        ):
            coord1, coord2, coord3 = spatial_subtract(
                lib, coord11, coord12, coord13, coord21, coord22, coord23
            )
            t1 = to_t1(lib, coord11, coord12, coord13, coord14)
            t2 = to_t2(lib, coord21, coord22, coord23, coord24)
            return (
                coord1,
                coord2,
                coord3,
                to_tau(lib, coord1, coord2, coord3, t1 - t2),
            )

        dispatch_map[
            azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2, temporal2
        ] = (
            f,
            azimuthal,
            longitudinal,
            TemporalTau,
        )  # type: ignore


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
        return _handler_of(v1, v2)._wrap_result(
            _flavor_of(v1, v2),
            function(
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
