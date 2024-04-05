# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Planar.scale(self, factor)
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


def xy(lib, factor, x, y):
    return (x * factor, y * factor)


def rhophi(lib, factor, rho, phi):
    absfactor = lib.absolute(factor)
    sign = lib.sign(factor)
    turn_if_negative = -0.5 * (sign - 1) * lib.pi
    return (rho * absfactor, rectify(lib, phi + turn_if_negative))


dispatch_map = {
    (AzimuthalXY,): (xy, AzimuthalXY),
    (AzimuthalRhoPhi,): (rhophi, AzimuthalRhoPhi),
}


def dispatch(factor: typing.Any, v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(__name__, dispatch_map, (_aztype(v),))
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(v.lib, factor, *v.azimuthal.elements),
            returns,
            1,
        )
