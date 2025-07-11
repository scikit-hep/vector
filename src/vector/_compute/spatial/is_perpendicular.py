# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.is_perpendicular(self, other, tolerance=...)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.spatial import dot, mag
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

dispatch_map = {}


def make_function(azimuthal1, longitudinal1, azimuthal2, longitudinal2):
    dot_function, _ = dot.dispatch_map[
        azimuthal1, longitudinal1, azimuthal2, longitudinal2
    ]
    mag1_function, _ = mag.dispatch_map[azimuthal1, longitudinal1]
    mag2_function, _ = mag.dispatch_map[azimuthal2, longitudinal2]

    def f(lib, tolerance, coord11, coord12, coord13, coord21, coord22, coord23):
        return dot_function(
            lib, coord11, coord12, coord13, coord21, coord22, coord23
        ) < lib.absolute(tolerance) * mag1_function(
            lib, coord11, coord12, coord13
        ) * mag2_function(lib, coord21, coord22, coord23)

    dispatch_map[azimuthal1, longitudinal1, azimuthal2, longitudinal2] = (f, bool)


for azimuthal1 in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal1 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for azimuthal2 in (AzimuthalXY, AzimuthalRhoPhi):
            for longitudinal2 in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
                make_function(azimuthal1, longitudinal1, azimuthal2, longitudinal2)


def dispatch(tolerance: typing.Any, v1: typing.Any, v2: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v1),
            _ltype(v1),
            _aztype(v2),
            _ltype(v2),
        ),
    )
    with numpy.errstate(all="ignore"):
        handler = _handler_of(v1, v2)
        return handler._wrap_result(
            _flavor_of(v1, v2),
            handler._wrap_dispatched_function(function)(
                _lib_of(v1, v2),
                tolerance,
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
                *v2.azimuthal.elements,
                *v2.longitudinal.elements,
            ),
            returns,
            2,
        )
