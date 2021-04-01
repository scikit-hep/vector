# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    @property
    Planar.rho(self)
"""

import numpy

from vector._compute.planar import rho2
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
)


def xy(lib, x, y):
    return lib.sqrt(rho2.xy(lib, x, y))


def rhophi(lib, rho, phi):
    return rho


dispatch_map = {
    (AzimuthalXY,): (xy, float),
    (AzimuthalRhoPhi,): (rhophi, float),
}


def dispatch(v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(__name__, dispatch_map, (_aztype(v),))
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v), function(v.lib, *v.azimuthal.elements), returns, 1
        )
