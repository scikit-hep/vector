# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Planar.add(self, other)
"""

import numpy

from vector._compute.planar import x, y
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


# specialized
def xy_xy(lib, x1, y1, x2, y2):
    return (x1 + x2, y1 + y2)


def xy_rhophi(lib, x1, y1, rho2, phi2):
    return xy_xy(lib, x1, y1, x.rhophi(lib, rho2, phi2), y.rhophi(lib, rho2, phi2))


def rhophi_xy(lib, rho1, phi1, x2, y2):
    return xy_xy(lib, x.rhophi(lib, rho1, phi1), y.rhophi(lib, rho1, phi1), x2, y2)


# specialized
def rhophi_rhophi(lib, rho1, phi1, rho2, phi2):
    diff = phi2 - phi1
    u = rho2 * lib.cos(diff)
    v = rho2 * lib.sin(diff)
    return (
        lib.sqrt((rho1 + u) ** 2 + v ** 2),
        rectify(lib, phi1 + lib.arctan2(v, rho1 + u)),
    )


dispatch_map = {
    (AzimuthalXY, AzimuthalXY): (xy_xy, AzimuthalXY),
    (AzimuthalXY, AzimuthalRhoPhi): (xy_rhophi, AzimuthalXY),
    (AzimuthalRhoPhi, AzimuthalXY): (rhophi_xy, AzimuthalXY),
    (AzimuthalRhoPhi, AzimuthalRhoPhi): (rhophi_rhophi, AzimuthalRhoPhi),
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
