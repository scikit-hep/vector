# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Planar.is_perpendicular(self, other, tolerance=...)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.planar import dot, rho
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
    _handler_of,
    _lib_of,
)

dispatch_map = {}


def make_function(azimuthal1, azimuthal2):
    dot_function, _ = dot.dispatch_map[azimuthal1, azimuthal2]
    rho1_function, _ = rho.dispatch_map[azimuthal1,]
    rho2_function, _ = rho.dispatch_map[azimuthal2,]

    def f(lib, tolerance, coord11, coord12, coord21, coord22):
        return dot_function(lib, coord11, coord12, coord21, coord22) < lib.absolute(
            tolerance
        ) * rho1_function(lib, coord11, coord12) * rho2_function(lib, coord21, coord22)

    dispatch_map[azimuthal1, azimuthal2] = (f, bool)


for azimuthal1 in (AzimuthalXY, AzimuthalRhoPhi):
    for azimuthal2 in (AzimuthalXY, AzimuthalRhoPhi):
        make_function(azimuthal1, azimuthal2)


def dispatch(tolerance: typing.Any, v1: typing.Any, v2: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v1),
            _aztype(v2),
        ),
    )
    with numpy.errstate(all="ignore"):
        return _handler_of(v1, v2)._wrap_result(
            _flavor_of(v1, v2),
            function(
                _lib_of(v1, v2),
                tolerance,
                *v1.azimuthal.elements,
                *v2.azimuthal.elements,
            ),
            returns,
            2,
        )
