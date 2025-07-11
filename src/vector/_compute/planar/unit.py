# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Planar.unit(self)
"""

from __future__ import annotations

import typing
from math import inf

import numpy

from vector._compute.planar import rho
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
)


def xy(lib, x, y):
    norm = rho.xy(lib, x, y)
    return (
        lib.nan_to_num(x / norm, nan=0, posinf=inf, neginf=-inf),
        lib.nan_to_num(y / norm, nan=0, posinf=inf, neginf=-inf),
    )


def rhophi(lib, rho, phi):
    return (1, phi)


rhophi.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


dispatch_map = {
    (AzimuthalXY,): (xy, AzimuthalXY),
    (AzimuthalRhoPhi,): (rhophi, AzimuthalRhoPhi),
}


def dispatch(v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(__name__, dispatch_map, (_aztype(v),))
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            v._wrap_dispatched_function(function)(v.lib, *v.azimuthal.elements),
            returns,
            1,
        )
