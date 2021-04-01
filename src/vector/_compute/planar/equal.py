# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Planar.equal(self, other)
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

# Policy: turn (rho, phi) into (x, y)
#         (if not already the same)


# same types
def xy_xy(lib, x1, y1, x2, y2):
    return (x1 == x2) & (y1 == y2)


def xy_rhophi(lib, x1, y1, rho2, phi2):
    return xy_xy(lib, x1, y1, x.rhophi(lib, rho2, phi2), y.rhophi(lib, rho2, phi2))


def rhophi_xy(lib, rho1, phi1, x2, y2):
    return xy_xy(lib, x.rhophi(lib, rho1, phi1), y.rhophi(lib, rho1, phi1), x2, y2)


# same types
def rhophi_rhophi(lib, rho1, phi1, rho2, phi2):
    return (rho1 == rho2) & (phi1 == phi2)


dispatch_map = {
    (AzimuthalXY, AzimuthalXY): (xy_xy, bool),
    (AzimuthalXY, AzimuthalRhoPhi): (xy_rhophi, bool),
    (AzimuthalRhoPhi, AzimuthalXY): (rhophi_xy, bool),
    (AzimuthalRhoPhi, AzimuthalRhoPhi): (rhophi_rhophi, bool),
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
            function(
                _lib_of(v1, v2),
                *v1.azimuthal.elements,
                *v2.azimuthal.elements,
            ),
            returns,
            2,
        )
