# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Planar.rotateZ(self, angle)
"""

from __future__ import annotations

import typing

import numpy

from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
)


def rectify(lib, phi):
    return (phi + lib.pi) % (2 * lib.pi) - lib.pi


def xy(lib, angle, x, y):
    s = lib.sin(angle)
    c = lib.cos(angle)
    return c * x - s * y, s * x + c * y


def rhophi(lib, angle, rho, phi):
    return rho, rectify(lib, phi + angle)


dispatch_map = {
    (AzimuthalXY,): (xy, AzimuthalXY),
    (AzimuthalRhoPhi,): (rhophi, AzimuthalRhoPhi),
}


def dispatch(angle: typing.Any, v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(__name__, dispatch_map, (_aztype(v),))
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            v._wrap_dispatched_function(function)(v.lib, angle, *v.azimuthal.elements),
            returns,
            1,
        )
