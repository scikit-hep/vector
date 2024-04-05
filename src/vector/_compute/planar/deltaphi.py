# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.
"""
.. code-block:: python

    Planar.deltaphi(self, other)
"""

from __future__ import annotations

import typing

import numpy

from vector._compute.planar import phi
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
    _handler_of,
    _lib_of,
)


def rectify(lib, phi):
    return (phi + lib.pi) % (2 * lib.pi) - lib.pi


def xy_xy(lib, x1, y1, x2, y2):
    return rectify(lib, phi.xy(lib, x1, y1) - phi.xy(lib, x2, y2))


def xy_rhophi(lib, x1, y1, rho2, phi2):
    return rectify(lib, phi.xy(lib, x1, y1) - phi2)


def rhophi_xy(lib, rho1, phi1, x2, y2):
    return rectify(lib, phi1 - phi.xy(lib, x2, y2))


def rhophi_rhophi(lib, rho1, phi1, rho2, phi2):
    return rectify(lib, phi1 - phi2)


dispatch_map = {
    (AzimuthalXY, AzimuthalXY): (xy_xy, float),
    (AzimuthalXY, AzimuthalRhoPhi): (xy_rhophi, float),
    (AzimuthalRhoPhi, AzimuthalXY): (rhophi_xy, float),
    (AzimuthalRhoPhi, AzimuthalRhoPhi): (rhophi_rhophi, float),
}


def dispatch(v1: typing.Any, v2: typing.Any) -> typing.Any:
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
            function(_lib_of(v1, v2), *v1.azimuthal.elements, *v2.azimuthal.elements),
            returns,
            2,
        )
