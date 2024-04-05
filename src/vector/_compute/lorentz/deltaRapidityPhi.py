# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.deltaRapidityPhi(self, other)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.lorentz import deltaRapidityPhi2
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
    lorentz_deltaRapidityPhi2, _ = deltaRapidityPhi2.dispatch_map[
        azimuthal1, longitudinal1, temporal1, azimuthal2, longitudinal2, temporal2
    ]

    def f(
        lib,
        coord11,
        coord12,
        coord13,
        coord14,
        coord21,
        coord22,
        coord23,
        coord24,
    ):
        return lib.sqrt(
            lorentz_deltaRapidityPhi2(
                lib,
                coord11,
                coord12,
                coord13,
                coord14,
                coord21,
                coord22,
                coord23,
                coord24,
            )
        )

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


def dispatch(
    v1: typing.Any,
    v2: typing.Any,
) -> typing.Any:
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
