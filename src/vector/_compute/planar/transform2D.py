# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Planar.transform2D(self, obj)

where ``obj`` has ``obj["xx"]``, ``obj["xy"]``, etc.
"""

import numpy

from vector._compute.planar import x, y
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
)

# Rotation is only computed in Cartesian coordinates; the rest are conversions.


def cartesian(lib, xx, xy, yx, yy, x, y):
    return (xx * x + xy * y, yx * x + yy * y)


def rhophi(lib, xx, xy, yx, yy, rho, phi):
    return cartesian(
        lib, xx, xy, yx, yy, x.rhophi(lib, rho, phi), y.rhophi(lib, rho, phi)
    )


dispatch_map = {
    (AzimuthalXY,): (cartesian, AzimuthalXY),
    (AzimuthalRhoPhi,): (rhophi, AzimuthalXY),
}


def dispatch(obj: typing.Any, v: typing.Any) -> typing.Any:
    function, *returns = _from_signature(__name__, dispatch_map, (_aztype(v),))
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(
                v.lib, obj["xx"], obj["xy"], obj["yx"], obj["yy"], *v.azimuthal.elements
            ),
            returns,
            1,
        )
